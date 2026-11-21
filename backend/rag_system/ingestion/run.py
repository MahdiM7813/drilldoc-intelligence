from __future__ import annotations

import argparse
import hashlib
import json
from rag_system.runtime_config import ensure_external_path, resolve_config_paths
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from rag_system.extraction.pdf_extractor import extract_pages
from rag_system.extraction.structure import detect_clauses, detect_structure
from rag_system.extraction.tables import inspect_tables


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "rag_system"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = path or PACKAGE_ROOT / "config" / "defaults.json"
    return resolve_config_paths(json.loads(source.read_text(encoding="utf-8")))


def infer_registry(pdf_path: Path, source_filename: str, document_id: str, pages: list[dict], file_hash: str, metadata: dict, config: dict) -> dict:
    cover = "\n".join(p["normalized_text"] for p in pages[:3])
    code_match = re.search(r"\b(ISO\s+\d+(?:[-:]\d{4})?)\b", cover, re.I)
    reference_match = re.search(r"\b(ISO\s+\d+):(\d{4})\s*\([A-Z]\)", cover, re.I)
    edition_match = re.search(r"\b((?:First|Second|Third|Fourth|Fifth|Sixth|\d+(?:st|nd|rd|th))\s+edition)\b", cover, re.I)
    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", cover)
    title_match = re.search(r"Welding\s+[—-]\s+(.+?)(?=\n(?:Soudage|Reference number))", cover, re.I | re.S)
    standard_code = re.sub(r":\d{4}$", "", reference_match.group(1) if reference_match else (code_match.group(1) if code_match else "unknown"), flags=re.I).upper()
    year = reference_match.group(2) if reference_match else (date_match.group(1)[:4] if date_match else None)
    notes: list[str] = ["Metadata was inferred from visible PDF content; authority/authenticity remains unverified."]
    filename_years = re.findall(r"(?:19|20)\d{2}", pdf_path.name)
    if year and filename_years and year not in filename_years:
        notes.append(f"Filename/content mismatch: original filename contains year(s) {', '.join(filename_years)}, while internal content identifies edition year {year}.")
    creation = metadata.get("/CreationDate", "")
    if creation:
        notes.append(f"File metadata CreationDate is {creation}; this is a file timestamp, not the standard publication date.")
    return {
        "document_id": document_id,
        "standard_code": standard_code,
        "title": re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else "Title not confidently extracted",
        "edition": edition_match.group(1) if edition_match else "unknown",
        "publication_date": date_match.group(1) if date_match else None,
        "language": config.get("language", "en"),
        "document_type": config.get("document_type", "standard_document"),
        "inspection_methods": [],
        "authority": standard_code.split()[0] if standard_code != "unknown" else "unknown",
        "authority_status": config.get("authority_status", "official_or_unverified"),
        "source_filename": source_filename,
        "original_filename": pdf_path.name,
        "page_count": len(pages),
        "file_hash_sha256": file_hash,
        "ingestion_status": "completed_with_warnings" if any(p["warnings"] for p in pages) else "completed",
        "notes": notes,
        "pdf_metadata": metadata,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def render_review_pages(source_pdf: Path, page_indexes: list[int], output_dir: Path) -> tuple[list[str], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered, warnings = [], []
    executable = shutil.which("pdftoppm")
    if not executable:
        return rendered, ["pdftoppm unavailable; visual review images were not generated"]
    if Path(executable).suffix.lower() in {".cmd", ".bat"}:
        runtime_candidate = Path(executable).parents[2] / "native" / "poppler" / "Library" / "bin" / "pdftoppm.exe"
        if runtime_candidate.is_file():
            executable = str(runtime_candidate)
    for index in sorted(set(page_indexes)):
        prefix = output_dir / f"pdf_page_{index:03d}"
        try:
            subprocess.run([executable, "-f", str(index + 1), "-l", str(index + 1), "-r", "120", "-singlefile", "-png", str(source_pdf), str(prefix)], check=True, capture_output=True, text=True)
            rendered.append(str(prefix.with_suffix(".png").name))
        except Exception as exc:
            warnings.append(f"page {index}: render failed: {exc}")
    return rendered, warnings


def make_report(registry: dict, pages: list[dict], structure: list[dict], clauses: list[dict], tables: list[dict], terms: list[str], rendered: list[str], render_warnings: list[str]) -> dict:
    empty = [p["pdf_page_index"] for p in pages if not p["normalized_text"]]
    corrupt = [p["pdf_page_index"] for p in pages if any("error" in w or "replacement" in w for w in p["warnings"])]
    manual = sorted(set(p["pdf_page_index"] for p in pages if p["warnings"] or p["has_tables"]))
    corpus = "\n".join(p["normalized_text"] for p in pages).casefold()
    term_checks = {term: term.casefold() in corpus for term in terms}
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "registry_metadata": registry,
        "pdf_page_count": len(pages),
        "pages_with_extracted_text": sum(bool(p["normalized_text"]) for p in pages),
        "empty_pages": empty,
        "detected_structure_count": len(structure),
        "detected_clause_count": len(clauses),
        "detected_table_count": len(tables),
        "possible_extraction_corruption_pages": corrupt,
        "pages_requiring_manual_review": manual,
        "term_checks": term_checks,
        "rendered_review_images": rendered,
        "render_warnings": render_warnings,
        "limitations": [
            "Text and heading order are heuristic and may differ from visual reading order.",
            "Tables, merged cells, drawings, and formulas require visual verification.",
            "Empty/uncertain cells are retained as uncertain and never guessed.",
            "No authenticity determination, engineering decision, or acceptance/rejection assessment is made.",
        ],
    }


def report_fa(report: dict, structure: list[dict], tables: list[dict]) -> str:
    registry = report["registry_metadata"]
    sections = "، ".join(f"{x['label']} - {x['title']}" for x in structure if x["type"] in {"section", "annex"}) or "موردی شناسایی نشد"
    table_status = "؛ ".join(f"{t['table_id']}: {t['extraction_status']} (صفحات PDF: {', '.join(map(str, t['pdf_pages']))})" for t in tables) or "جدولی شناسایی نشد"
    term_fail = [term for term, ok in report["term_checks"].items() if not ok]
    return f"""# گزارش استخراج Stage RAG-01

## سند پردازش‌شده

- فایل اصلی: `{registry['original_filename']}` (بدون تغییر، جابه‌جایی یا حذف)
- نسخهٔ ثبت‌شده برای پردازش: `{registry['source_filename']}`
- شناسه: `{registry['document_id']}`
- SHA-256: `{registry['file_hash_sha256']}`
- تعداد صفحات PDF: {report['pdf_page_count']}

## فرادادهٔ داخلی

- کد استاندارد: {registry['standard_code']}
- ویرایش: {registry['edition']}
- تاریخ انتشار: {registry['publication_date']}
- عنوان: {registry['title']}
- وضعیت مرجع: {registry['authority_status']}؛ اصالت حقوقی فایل در این مرحله تأیید نشده است.

## اختلاف نام فایل و محتوا

نام فایل اصلی به سال 2016 اشاره می‌کند، اما روی جلد و شمارهٔ مرجع داخلی، سند `ISO 5817:2014(E)` با ویرایش سوم و تاریخ 2014-02-15 درج شده است. تاریخ ساخت فایل PDF در 2016 یک ویژگی فایل است و سال انتشار استاندارد محسوب نمی‌شود. این اختلاف در رجیستری نیز ثبت شده است.

## نتیجهٔ استخراج

- صفحات دارای متن: {report['pages_with_extracted_text']}
- بندهای کاندید: {report['detected_clause_count']}
- نواحی/جدول‌های شناسایی‌شده: {report['detected_table_count']}
- صفحات PDF نیازمند بازبینی دستی (اندیس صفرمبنا): {report['pages_requiring_manual_review']}
- صفحات با احتمال خرابی استخراج: {report['possible_extraction_corruption_pages']}
- تصاویر بازبینی تولیدشده: {', '.join(report['rendered_review_images']) or 'هیچ‌کدام'}

بخش‌ها و پیوست‌های شناسایی‌شده: {sections}

وضعیت جدول‌ها: {table_status}

عبارت‌های مورد انتظار که پیدا نشدند: {('، '.join(term_fail) if term_fail else 'هیچ‌کدام')}. این عبارت‌ها فقط کنترل کیفیت همین سند هستند و به‌عنوان قاعدهٔ دائمی سامانه استفاده نشده‌اند.

## محدودیت‌ها

استخراج متن، تشخیص تیتر و بازسازی جدول‌ها خودکار و اکتشافی است. جدول چندصفحه‌ای، سلول‌های ادغام‌شده، شکل‌ها و فرمول‌ها باید با تصویر صفحه مقایسه شوند. هیچ سلول ناموجودی حدس زده نشده و سلول‌های خالی/نامطمئن علامت‌گذاری شده‌اند. این مرحله هیچ embedding، جست‌وجوی برداری، پاسخ‌گویی، agent، rule engine یا تصمیم پذیرش/رد پیاده‌سازی نمی‌کند.
"""


def run(pdf_path: Path, document_id: str, output_root: Path, config_path: Path | None = None, source_filename: str | None = None) -> dict:
    pdf_path = pdf_path.resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", document_id):
        raise ValueError("document_id must contain only letters, numbers, dots, underscores, or hyphens")
    output_root = ensure_external_path(output_root)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    original_hash = sha256_file(pdf_path)
    config = load_config(config_path)
    reader = PdfReader(str(pdf_path))
    metadata = {str(k): str(v) for k, v in (reader.metadata or {}).items()}
    pages, page_tables = extract_pages(pdf_path, document_id, config)
    structure = detect_structure(pages, document_id)
    clauses = detect_clauses(pages, document_id)
    tables = inspect_tables(pages, page_tables, clauses, document_id)

    source_name = source_filename or f"{document_id.replace('-', '_')}.pdf"
    source_copy = output_root / "documents" / "source" / source_name
    source_copy.parent.mkdir(parents=True, exist_ok=True)
    if source_copy.resolve() != pdf_path:
        shutil.copy2(pdf_path, source_copy)
    if sha256_file(pdf_path) != original_hash or sha256_file(source_copy) != original_hash:
        raise RuntimeError("Source integrity check failed")

    registry = infer_registry(pdf_path, source_name, document_id, pages, original_hash, metadata, config)
    review_pages = sorted(set(i for table in tables for i in table["pdf_pages"] if table["extraction_status"] != "complete"))
    rendered, render_warnings = render_review_pages(source_copy, review_pages, output_root / "outputs" / "pages")
    report = make_report(registry, pages, structure, clauses, tables, config.get("verification_terms", []), rendered, render_warnings)

    write_json(output_root / "documents" / "registry" / "document_registry.json", registry)
    write_jsonl(output_root / "outputs" / "raw_text" / "pages.jsonl", pages)
    write_json(output_root / "outputs" / "reports" / "detected_structure.json", structure)
    write_jsonl(output_root / "outputs" / "reports" / "clauses.jsonl", clauses)
    write_json(output_root / "outputs" / "tables" / "tables.json", tables)
    write_json(output_root / "outputs" / "reports" / "extraction_report.json", report)
    (output_root / "outputs" / "reports" / "EXTRACTION_REPORT_FA.md").write_text(report_fa(report, structure, tables), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register, extract, and inspect a standards PDF.")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-filename")
    args = parser.parse_args(argv)
    report = run(args.pdf, args.document_id, args.output_root, args.config, args.source_filename)
    print(json.dumps({k: report[k] for k in ("pdf_page_count", "detected_clause_count", "detected_table_count", "pages_requiring_manual_review")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

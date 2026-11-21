from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_system.chunking.hierarchy import extract_definition_candidates, resolve_hierarchy
from rag_system.chunking.table_chunker import build_table_chunks
from rag_system.chunking.text_chunker import build_definition_chunks, build_prose_chunks
from rag_system.chunking.validator import validate_chunks
from rag_system.runtime_config import ensure_external_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "rag_system"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_simple_yaml(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        key, value = (x.strip() for x in stripped.split(":", 1))
        if value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
        elif value.isdigit():
            result[key] = int(value)
        else:
            result[key] = value.strip("'\"")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in values)
    path.write_text(content, encoding="utf-8")


def _report_fa(report: dict) -> str:
    counts = report["counts_by_content_type"]
    return f"""# گزارش Stage RAG-02 - ساخت chunk ساختارآگاه

## خلاصه

- کل chunkها: {report['total_chunks']}
- chunkهای نثر/بند: {report['prose_chunks']}
- تعریف‌ها: {report['definition_chunks']}
- chunkهای جدول (والد و قطعه): {report['table_chunks']}
- chunkهای پیوست: {report['annex_chunks']}
- میانگین تخمین token: {report['average_token_estimate']}
- بیشترین تخمین token: {report['maximum_token_estimate']}
- نیازمند بازبینی دستی: {report['chunks_requiring_manual_review']}
- سلسله‌مراتب نامطمئن: {report['uncertain_hierarchy_count']}
- تکرار دقیق متن نرمال‌شده: {report['duplicate_chunk_count']}

## شمارش بر اساس نوع

`{json.dumps(counts, ensure_ascii=False, sort_keys=True)}`

## جدول‌ها

`{json.dumps(report['table_segment_statistics'], ensure_ascii=False, sort_keys=True)}`

## کنترل اعتبار

- معتبر: {report['validation']['valid']}
- خطاها: `{json.dumps(report['validation']['errors'], ensure_ascii=False)}`
- هشدارهای عمومی: `{json.dumps(report['validation']['warnings'], ensure_ascii=False)}`
- دسته‌بندی هشدارها: `{json.dumps(report['validation']['warnings_by_category'], ensure_ascii=False, sort_keys=True)}`

## پیکربندی

`{json.dumps(report['configuration'], ensure_ascii=False, sort_keys=True)}`

## محدودیت‌ها

تعریف‌های صفحهٔ چاپی ۲ به علت درهم‌آمیختگی شماره‌ها با متن در استخراج PDF با confidence محدود و نیازمند بازبینی ثبت شده‌اند. جدول‌های چندصفحه‌ای شامل سلول‌های ادغام‌شده، شکل‌ها و فرمول‌ها هستند؛ قطعه‌ها ساختار را حفظ می‌کنند اما قاعده یا مقدار ناموجودی را حدس نمی‌زنند. تخمین token محلی و قطعی است و tokenizer مدل خاصی نیست.

این مرحله embedding، پایگاه دادهٔ برداری، semantic retrieval، پاسخ‌گویی، agent، rule engine یا تصمیم پذیرش/رد پیاده‌سازی نمی‌کند.
"""


def _sample_block(title: str, chunk: dict | None) -> str:
    if not chunk:
        return f"## {title}\n\nنمونهٔ قابل‌اعتماد پیدا نشد.\n"
    return f"""## {title}

- chunk_id: `{chunk['chunk_id']}`
- type: `{chunk['content_type']}`
- hierarchy: `{chunk.get('section_number')} / {chunk.get('clause_number')}`
- pages: PDF {chunk['pdf_page_start']}-{chunk['pdf_page_end']}; printed {chunk.get('printed_page_start')}-{chunk.get('printed_page_end')}
- citation: {chunk['citation']['display']}
- warnings: `{json.dumps(chunk['warnings'], ensure_ascii=False)}`

### Source text

```text
{chunk['source_text']}
```

### Normalized text

```text
{chunk['normalized_text']}
```

### Embedding text

```text
{chunk['embedding_text']}
```
"""


def _samples(chunks: list[dict]) -> str:
    pick = lambda fn: next((x for x in chunks if fn(x)), None)
    table1 = [x for x in chunks if x["content_type"] == "table_segment" and x["table_id"].endswith("TABLE-1")]
    selections = [
        ("Scope", pick(lambda x: x.get("clause_number") == "1")),
        ("Normative reference", pick(lambda x: x.get("clause_number") == "2" and "ISO 6520-1" in x.get("source_text", "")) or pick(lambda x: x.get("clause_number") == "2")),
        ("Definition", pick(lambda x: x["content_type"] == "definition")),
        ("Section 5", pick(lambda x: x.get("clause_number") == "5")),
        ("Table 1 - normal segment", next((x for x in table1 if x.get("structured_rows")), None)),
        ("Table 1 - uncertain segment", next((x for x in table1 if not x.get("structured_rows")), table1[-1] if table1 else None)),
        ("Annex", pick(lambda x: x["content_type"] == "annex")),
    ]
    return "# نمونه‌های بازرسی انسانی Stage RAG-02\n\n" + "\n".join(_sample_block(title, chunk) for title, chunk in selections)


def run(registry_path: Path, pages_path: Path, structure_path: Path, clauses_path: Path, tables_path: Path, config_path: Path, output_root: Path) -> dict:
    output_root = ensure_external_path(output_root)
    registry, pages, structure, clauses, tables = load_json(registry_path), load_jsonl(pages_path), load_json(structure_path), load_jsonl(clauses_path), load_json(tables_path)
    config = load_simple_yaml(config_path)
    table_pages = {i for table in tables for i in table["pdf_pages"]}
    hierarchy = resolve_hierarchy(pages, clauses, table_pages)
    definitions = extract_definition_candidates(pages, hierarchy)
    prose = build_prose_chunks(registry, pages, hierarchy, table_pages, config)
    definition_chunks = build_definition_chunks(registry, pages, definitions, config)
    table_parents, table_children = build_table_chunks(registry, pages, tables, config)
    chunk_objects = prose + definition_chunks + table_parents + table_children
    chunks = [x.to_dict() for x in chunk_objects]
    chunks.sort(key=lambda x: (x["pdf_page_start"], x["content_type"], x["chunk_id"]))
    validation = validate_chunks(chunks, registry, tables, config)
    if not validation["valid"]:
        raise ValueError("Chunk validation failed: " + "; ".join(validation["errors"]))
    counts = Counter(x["content_type"] for x in chunks)
    token_values = [x["token_estimate"] for x in chunks]
    table_stats = {table["table_id"]: sum(x["content_type"] == "table_segment" and x["table_id"] == table["table_id"] for x in chunks) for table in tables}
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "document_id": registry["document_id"],
        "total_chunks": len(chunks),
        "counts_by_content_type": dict(sorted(counts.items())),
        "prose_chunks": sum(counts[x] for x in ("paragraph", "clause", "front_matter")),
        "definition_chunks": counts["definition"],
        "table_chunks": counts["table"] + counts["table_segment"],
        "annex_chunks": counts["annex"],
        "average_token_estimate": round(sum(token_values) / len(token_values), 2) if token_values else 0,
        "maximum_token_estimate": max(token_values, default=0),
        "chunks_requiring_manual_review": sum(x["requires_manual_review"] for x in chunks),
        "uncertain_hierarchy_count": sum((x.get("hierarchy_confidence") or 1) < .8 for x in chunks),
        "duplicate_chunk_count": validation["duplicate_chunk_count"],
        "table_segment_statistics": table_stats,
        "hierarchy_candidates": [x.__dict__ for x in hierarchy],
        "configuration": config,
        "validation": validation,
        "limitations": ["Definition numbering is layout-contaminated in Stage RAG-01 text and requires manual review.", "Multi-page tables remain partial; no cell values or alignment were guessed.", "Token counts use a deterministic offline lexical estimate."],
    }
    out = output_root
    write_jsonl(out / "chunks" / "chunks.jsonl", chunks)
    write_jsonl(out / "chunks" / "definitions.jsonl", [x for x in chunks if x["content_type"] == "definition"])
    write_jsonl(out / "chunks" / "table_chunks.jsonl", [x for x in chunks if x["content_type"] in {"table", "table_segment"}])
    write_json(out / "reports" / "chunking_report.json", report)
    (out / "reports" / "CHUNKING_REPORT_FA.md").write_text(_report_fa(report), encoding="utf-8")
    (out / "reports" / "chunk_samples.md").write_text(_samples(chunks), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic, structure-aware chunks from Stage RAG-01 outputs.")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--pages", required=True, type=Path)
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--clauses", required=True, type=Path)
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run(args.registry, args.pages, args.structure, args.clauses, args.tables, args.config, args.output_root)
    print(json.dumps({k: report[k] for k in ("total_chunks", "counts_by_content_type", "average_token_estimate", "maximum_token_estimate", "chunks_requiring_manual_review", "uncertain_hierarchy_count", "duplicate_chunk_count")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

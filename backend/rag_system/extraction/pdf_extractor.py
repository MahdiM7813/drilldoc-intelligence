from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import pdfplumber


PRINTED_PAGE_RE = re.compile(r"(?m)^\s*(?:Page\s+)?([ivxlcdm]+|\d+)\s*$", re.I)


def normalize_text(text: str) -> str:
    """Normalize compatibility forms while preserving mathematical Unicode."""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()).strip()


def printed_page_number(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[-4:] + lines[:3]:
        match = PRINTED_PAGE_RE.fullmatch(line)
        if match:
            return match.group(1)
    # ISO headers/footers often put the number beside copyright text.
    match = re.search(r"(?m)^\s*([ivxlcdm]+|\d+)\s+©\s+ISO\b", text, re.I)
    if match:
        return match.group(1)
    match = re.search(r"(?mi)^.*©\s+ISO.*?\b([ivxlcdm]+|\d+)\s*$", text)
    return match.group(1) if match else None


def extract_pages(pdf_path: Path, document_id: str, config: dict[str, Any]) -> tuple[list[dict], list[list]]:
    pages: list[dict] = []
    page_tables: list[list] = []
    low_threshold = int(config.get("low_text_character_threshold", 120))
    empty_threshold = int(config.get("empty_page_character_threshold", 20))
    table_settings = config.get("table_detection", {})
    with pdfplumber.open(str(pdf_path)) as pdf:
        for index, page in enumerate(pdf.pages):
            warnings: list[str] = []
            try:
                raw = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
            except Exception as exc:
                raw = ""
                warnings.append(f"text_extraction_error: {type(exc).__name__}: {exc}")
            normalized = normalize_text(raw)
            try:
                found = page.find_tables(table_settings=table_settings)
                extracted_tables = [table.extract() for table in found]
            except Exception as exc:
                found, extracted_tables = [], []
                warnings.append(f"table_detection_error: {type(exc).__name__}: {exc}")
            if len(normalized) < empty_threshold:
                warnings.append("empty_or_nearly_empty_text; manual review required")
            elif len(normalized) < low_threshold:
                warnings.append("low_text_volume; inspect page visually")
            replacement_count = normalized.count("�")
            if replacement_count:
                warnings.append(f"unicode_replacement_characters={replacement_count}")
            if found:
                warnings.append("table_layout_present; linear text order may be unreliable")
            pages.append({
                "document_id": document_id,
                "pdf_page_index": index,
                "printed_page_number": printed_page_number(raw),
                "raw_text": raw,
                "normalized_text": normalized,
                "has_images": bool(page.images),
                "has_tables": bool(found),
                "page_width_points": round(float(page.width), 3),
                "page_height_points": round(float(page.height), 3),
                "extraction_method": "pdfplumber.layout_text",
                "warnings": warnings,
            })
            page_tables.append(extracted_tables)
    return pages, page_tables

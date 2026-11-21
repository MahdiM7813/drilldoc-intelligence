from __future__ import annotations

from typing import Any


FIELDS = (
    "chunk_id", "document_id", "standard_code", "edition", "publication_date", "language",
    "content_type", "section_number", "section_title", "clause_number", "clause_title",
    "parent_clause", "table_id", "figure_id", "pdf_page_start", "pdf_page_end",
    "printed_page_start", "printed_page_end", "extraction_status", "requires_manual_review",
)


def _safe(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def flatten_metadata(chunk: dict, source_hash: str) -> dict[str, str | int | float | bool]:
    metadata = {field: _safe(chunk.get(field)) for field in FIELDS}
    citation = chunk.get("citation") or {}
    metadata["citation_display"] = _safe(citation.get("display"))
    metadata["citation_clause"] = _safe(citation.get("clause"))
    metadata["citation_table"] = _safe(citation.get("table"))
    metadata["source_hash"] = source_hash
    metadata["warnings"] = " | ".join(str(x) for x in chunk.get("warnings", []))
    return metadata

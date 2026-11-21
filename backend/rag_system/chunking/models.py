from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_CONTENT_TYPES = {
    "paragraph", "definition", "clause", "table", "table_segment",
    "figure_caption", "annex", "front_matter",
}


@dataclass
class HierarchyItem:
    number: str
    title: str
    pdf_page: int
    printed_page: str | None
    item_type: str
    detection_source: str
    confidence: float
    warnings: list[str] = field(default_factory=list)
    parent_clause: str | None = None


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    standard_code: str
    edition: str
    publication_date: str | None
    language: str
    content_type: str
    section_number: str | None
    section_title: str | None
    clause_number: str | None
    clause_title: str | None
    parent_clause: str | None
    pdf_page_start: int
    pdf_page_end: int
    printed_page_start: str | None
    printed_page_end: str | None
    source_text: str
    normalized_text: str
    embedding_text: str
    table_id: str | None
    figure_id: str | None
    citation: dict[str, Any]
    source_record_ids: list[str]
    token_estimate: int
    character_count: int
    extraction_status: str
    warnings: list[str]
    requires_manual_review: bool
    detection_source: str | None = None
    hierarchy_confidence: float | None = None
    term: str | None = None
    definition_text: str | None = None
    segment_id: str | None = None
    segment_type: str | None = None
    row_labels: list[str] | None = None
    structured_rows: list[dict[str, Any]] | None = None
    continuation_of: str | None = None
    parent_table_chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

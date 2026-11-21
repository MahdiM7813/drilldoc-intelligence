from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    content_type: str
    document_id: str
    clause_number: str
    clause_title: str
    table_id: str
    pdf_page_start: int
    pdf_page_end: int
    citation: dict[str, Any]
    distance: float
    source_hash: str
    collection: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_RESULT_FIELDS = {
    "chunk_id", "content_type", "document_id", "clause_number", "clause_title",
    "table_id", "pdf_page_start", "pdf_page_end", "citation", "distance", "source_hash",
}

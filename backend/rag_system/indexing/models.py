from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TEXT_TYPES = {"clause", "paragraph", "definition", "annex", "front_matter", "figure_caption"}
TABLE_TYPES = {"table", "table_segment", "table_header", "table_row_group", "table_continuation", "table_unstructured_segment"}


@dataclass
class IndexableRecord:
    chunk_id: str
    document_id: str
    embedding_text: str
    content_type: str
    metadata: dict[str, Any]
    source_hash: str
    collection: str
    index_status: str = "pending"
    warnings: list[str] = field(default_factory=list)
    original_character_length: int = 0
    effective_processed_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

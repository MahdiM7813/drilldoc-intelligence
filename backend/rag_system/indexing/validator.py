from __future__ import annotations

from .models import IndexableRecord


def validate_records(records: list[IndexableRecord], registry: dict) -> None:
    if not records:
        raise ValueError("No valid indexable records remain")
    seen: set[str] = set()
    for record in records:
        if record.chunk_id in seen:
            raise ValueError(f"Duplicate prepared chunk ID: {record.chunk_id}")
        seen.add(record.chunk_id)
        if record.document_id != registry.get("document_id"):
            raise ValueError(f"Document mismatch: {record.chunk_id}")
        if not record.embedding_text.strip():
            raise ValueError(f"Empty embedding_text: {record.chunk_id}")
        if not record.metadata.get("citation_display"):
            raise ValueError(f"Missing citation metadata: {record.chunk_id}")
        if len(record.source_hash) != 64:
            raise ValueError(f"Invalid source hash: {record.chunk_id}")

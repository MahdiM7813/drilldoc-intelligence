from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .metadata import flatten_metadata
from .models import IndexableRecord, TABLE_TYPES, TEXT_TYPES


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source_hash(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL at {path}:{number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{number}")
        records.append(value)
    return records


def assign_collection(content_type: str, text_collection: str, table_collection: str) -> str:
    if content_type in TEXT_TYPES:
        return text_collection
    if content_type in TABLE_TYPES:
        return table_collection
    raise ValueError(f"Unsupported indexable content_type: {content_type}")


def prepare_records(paths: list[Path], registry: dict, config: dict) -> tuple[list[IndexableRecord], dict]:
    loaded: list[tuple[dict, str]] = []
    for path in paths:
        loaded.extend((record, str(path)) for record in load_jsonl(path))
    by_id: dict[str, tuple[dict, str, str]] = {}
    duplicate_records: list[dict] = []
    conflicts: list[dict] = []
    for record, origin in loaded:
        chunk_id = record.get("chunk_id")
        text = record.get("embedding_text")
        if not chunk_id:
            conflicts.append({"chunk_id": None, "origin": origin, "reason": "missing chunk_id"}); continue
        if record.get("document_id") != registry.get("document_id"):
            raise ValueError(f"Registry/chunk document_id conflict for {chunk_id}")
        digest = source_hash(text) if isinstance(text, str) else ""
        if chunk_id in by_id:
            previous, previous_origin, previous_hash = by_id[chunk_id]
            if previous_hash != digest or previous != record:
                conflicts.append({"chunk_id": chunk_id, "origins": [previous_origin, origin], "hashes": [previous_hash, digest], "reason": "same chunk_id with different content"})
            else:
                duplicate_records.append({"chunk_id": chunk_id, "source_hash": digest, "kept_from": previous_origin, "excluded_from": origin, "reason": "identical chunk_id and content"})
            continue
        by_id[chunk_id] = (record, origin, digest)
    if conflicts and config["indexing"].get("fail_on_conflicting_ids", True):
        raise ValueError(f"Conflicting records detected: {len(conflicts)}")

    prepared: list[IndexableRecord] = []
    skipped: list[dict] = []
    for chunk_id in sorted(by_id):
        chunk, origin, digest = by_id[chunk_id]
        text = chunk.get("embedding_text")
        if not isinstance(text, str) or not text.strip():
            skipped.append({"chunk_id": chunk_id, "reason": "empty embedding_text", "origin": origin}); continue
        if chunk.get("content_type") == "table" and not config["indexing"].get("index_table_parents", False):
            skipped.append({"chunk_id": chunk_id, "reason": "table parent excluded", "origin": origin}); continue
        try:
            collection = assign_collection(chunk.get("content_type", ""), config["vector_store"]["text_collection"], config["vector_store"]["table_collection"])
        except ValueError as exc:
            skipped.append({"chunk_id": chunk_id, "reason": str(exc), "origin": origin}); continue
        prepared.append(IndexableRecord(
            chunk_id=chunk_id, document_id=chunk["document_id"], embedding_text=text,
            content_type=chunk["content_type"], metadata=flatten_metadata(chunk, digest),
            source_hash=digest, collection=collection, warnings=list(chunk.get("warnings", [])),
            original_character_length=len(text),
        ))
    if not prepared:
        raise ValueError("No valid indexable records remain")
    return prepared, {
        "total_loaded_records": len(loaded), "unique_chunk_ids": len(by_id),
        "duplicate_records": duplicate_records, "duplicate_count": len(duplicate_records),
        "conflicts": conflicts, "conflict_count": len(conflicts), "skipped": skipped,
    }

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def deterministic_index_id(model_config: dict, input_files: list[dict], vector_config: dict) -> str:
    payload = {
        "embedding": model_config,
        "input_files": sorted(({"name": Path(x["path"]).name, "sha256": x["sha256"]} for x in input_files), key=lambda x: x["name"]),
        "collections": {"text": vector_config["text_collection"], "tables": vector_config["table_collection"], "distance_metric": vector_config["distance_metric"]},
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"local-rag-index-{digest[:24]}"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def build_manifest(*, config: dict, input_files: list[dict], embedder: Any, collection_counts: dict, load_report: dict, sync_stats: dict, truncated: list[dict], previous: dict | None = None, rebuild: bool = False) -> dict:
    now = datetime.now().astimezone().isoformat()
    index_id = deterministic_index_id(config["embedding"], input_files, config["vector_store"])
    collections = {name: {"record_count": count, "distance_metric": config["vector_store"]["distance_metric"]} for name, count in collection_counts.items()}
    return {
        "index_id": index_id,
        "created_at": now if rebuild or not previous else previous.get("created_at", now),
        "updated_at": now,
        "embedding_model": config["embedding"]["model_name"],
        "embedding_library": "sentence-transformers",
        "embedding_dimension": embedder.dimension,
        "normalized_embeddings": bool(config["embedding"].get("normalize_embeddings", True)),
        "embedding_precision": config["embedding"].get("precision", "float32"),
        "distance_metric": config["vector_store"]["distance_metric"],
        "device": embedder.device,
        "model_max_sequence_length": embedder.max_sequence_length,
        "truncation_behavior": "sentence-transformers model tokenizer truncation at model_max_sequence_length",
        "chroma_version": package_version("chromadb"),
        "sentence_transformers_version": package_version("sentence-transformers"),
        "python_version": platform.python_version(),
        "input_files": input_files,
        "collections": collections,
        "indexed_chunk_count": sum(collection_counts.values()),
        "skipped_chunk_count": len(load_report["skipped"]) + sync_stats["unchanged"],
        "failed_chunk_count": sync_stats["failed"],
        "duplicate_chunk_count": load_report["duplicate_count"],
        "conflict_count": load_report["conflict_count"],
        "manual_review_chunk_count": sum(bool(record.metadata.get("requires_manual_review")) for record in load_report.get("prepared_records", [])),
        "truncated_record_count": len(truncated),
        "changes": {key: sync_stats[key] for key in ("added", "updated", "unchanged", "removed", "failed")},
    }


def backup_manifest(path: Path) -> Path | None:
    if not path.is_file():
        return None
    backup = path.with_name("index_manifest.backup.json")
    backup.write_bytes(path.read_bytes())
    return backup

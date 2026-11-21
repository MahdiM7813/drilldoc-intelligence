from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import IndexableRecord


class ChromaStore:
    def __init__(self, persist_directory: Path, config: dict[str, Any]):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required; no vector-store fallback will be used") from exc
        self.chromadb = chromadb
        self.persist_directory = persist_directory.resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

    @property
    def collection_names(self) -> list[str]:
        return [self.config["text_collection"], self.config["table_collection"]]

    def _existing_names(self) -> set[str]:
        return {x.name if hasattr(x, "name") else str(x) for x in self.client.list_collections()}

    def rebuild_collections(self) -> None:
        existing = self._existing_names()
        for name in self.collection_names:
            if name in existing:
                self.client.delete_collection(name)

    def ensure_collections(self) -> dict[str, Any]:
        return {
            name: self.client.get_or_create_collection(name=name, metadata={"hnsw:space": self.config.get("distance_metric", "cosine")})
            for name in self.collection_names
        }

    def sync(self, records: list[IndexableRecord], embedder: Any, prune: bool = False) -> dict:
        collections = self.ensure_collections()
        grouped: dict[str, list[IndexableRecord]] = defaultdict(list)
        for record in records:
            grouped[record.collection].append(record)
        stats = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "failed": 0, "failed_records": []}
        for name, collection in collections.items():
            target = grouped.get(name, [])
            existing_data = collection.get(include=["metadatas"])
            existing = {identifier: metadata or {} for identifier, metadata in zip(existing_data.get("ids", []), existing_data.get("metadatas", []))}
            changed: list[IndexableRecord] = []
            for record in target:
                previous = existing.get(record.chunk_id)
                if previous is None:
                    record.index_status = "indexed"; changed.append(record); stats["added"] += 1
                elif previous.get("source_hash") == record.source_hash:
                    record.index_status = "skipped"; stats["unchanged"] += 1
                else:
                    record.index_status = "indexed"; changed.append(record); stats["updated"] += 1
            if changed:
                try:
                    vectors = embedder.encode([record.embedding_text for record in changed])
                    collection.upsert(
                        ids=[record.chunk_id for record in changed], embeddings=vectors,
                        documents=[record.embedding_text for record in changed],
                        metadatas=[record.metadata for record in changed],
                    )
                except Exception as exc:
                    stats["failed"] += len(changed)
                    for record in changed:
                        record.index_status = "failed"
                        stats["failed_records"].append({"chunk_id": record.chunk_id, "collection": name, "error": str(exc)})
                    raise
            if prune:
                desired = {record.chunk_id for record in target}
                removable = sorted(set(existing) - desired)
                if removable:
                    collection.delete(ids=removable); stats["removed"] += len(removable)
        return stats

    def counts(self) -> dict[str, int]:
        collections = self.ensure_collections()
        return {name: collection.count() for name, collection in collections.items()}

    def verify(self, expected: dict[str, int], dimension: int, sample_ids: dict[str, str]) -> dict:
        checks: dict[str, Any] = {"collections": {}, "persistence_restart": False}
        collections = self.ensure_collections()
        for name, collection in collections.items():
            count = collection.count()
            if count != expected[name]:
                raise RuntimeError(f"Collection count mismatch for {name}: {count}!={expected[name]}")
            sample_id = sample_ids.get(name)
            fetched = collection.get(ids=[sample_id], include=["embeddings", "documents", "metadatas"]) if sample_id else {"ids": []}
            if sample_id and fetched.get("ids") != [sample_id]:
                raise RuntimeError(f"Could not fetch {sample_id} from {name}")
            vectors = fetched.get("embeddings")
            if sample_id and (vectors is None or len(vectors[0]) != dimension):
                raise RuntimeError(f"Vector dimension mismatch in {name}")
            metadata_filter_count = collection.count() if not sample_id else len(collection.get(where={"document_id": fetched["metadatas"][0]["document_id"]})["ids"])
            checks["collections"][name] = {"count": count, "sample_id": sample_id, "vector_dimension": dimension, "metadata_filter_matches": metadata_filter_count, "document_readback": bool(fetched.get("documents", [None])[0]) if sample_id else True}
        # A fresh client proves on-disk persistence.
        fresh = self.chromadb.PersistentClient(path=str(self.persist_directory))
        for name, expected_count in expected.items():
            if fresh.get_collection(name).count() != expected_count:
                raise RuntimeError(f"Persistence restart count mismatch for {name}")
        checks["persistence_restart"] = True
        return checks

    def smoke_query(self, collection_name: str, query: str, query_vector: list[float], top_k: int) -> dict:
        collection = self.ensure_collections()[collection_name]
        limit = min(top_k, collection.count())
        if limit == 0:
            return {"query": query, "collection": collection_name, "results": []}
        result = collection.query(query_embeddings=[query_vector], n_results=limit, include=["distances", "metadatas"])
        rows = []
        for identifier, distance, metadata in zip(result["ids"][0], result["distances"][0], result["metadatas"][0]):
            rows.append({"chunk_id": identifier, "distance": float(distance), "citation": metadata.get("citation_display", ""), "content_type": metadata.get("content_type", "")})
        return {"query": query, "collection": collection_name, "results": rows}

    def close(self) -> None:
        """Release Chroma's process-wide shared systems (important on Windows)."""
        try:
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        finally:
            self.client = None

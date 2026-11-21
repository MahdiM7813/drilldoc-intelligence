from __future__ import annotations

import json
import gc
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from .citation import citation_from_metadata
from .query_processor import process_query
from .schemas import RetrievalResult


class LocalRetriever:
    instances_created = 0
    """Embed queries locally and search existing Chroma collections read-only."""

    def __init__(self, config: dict[str, Any], manifest_path: Path, model: Any | None = None, client: Any | None = None):
        self.config = config
        self.manifest_path = manifest_path.resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        embedding = config["embedding"]
        if embedding["model_name"] != self.manifest["embedding_model"]:
            raise ValueError("Retrieval model does not match index manifest")
        if config["retrieval"]["distance_metric"] != self.manifest["distance_metric"]:
            raise ValueError("Retrieval distance metric does not match index manifest")
        self.persist_directory = Path(config["vector_store"]["persist_directory"]).resolve()
        if not (self.persist_directory / "chroma.sqlite3").is_file():
            raise FileNotFoundError(f"Existing Chroma store not found: {self.persist_directory}")
        self._temporary_root: Path | None = None
        client_directory = self.persist_directory
        if client is None and config["retrieval"].get("read_only_copy", True):
            self._temporary_root = Path(tempfile.mkdtemp(prefix="weld_rag_retrieval_"))
            client_directory = self._temporary_root / "chroma"
            shutil.copytree(self.persist_directory, client_directory)
        self.client = client or chromadb.PersistentClient(path=str(client_directory))
        self.collection_names = {
            "text": config["vector_store"]["text_collection"],
            "tables": config["vector_store"]["table_collection"],
        }
        self.collections = {key: self.client.get_collection(name) for key, name in self.collection_names.items()}
        requested = embedding.get("device", "cpu")
        device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
        self.model = model or SentenceTransformer(
            embedding["model_name"], device=device,
            trust_remote_code=bool(embedding.get("trust_remote_code", False)),
            local_files_only=True,
        )
        get_dimension = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)
        dimension = int(get_dimension())
        if dimension != int(self.manifest["embedding_dimension"]):
            raise ValueError(f"Query embedding dimension {dimension} does not match index dimension {self.manifest['embedding_dimension']}")
        self.normalize_embeddings = bool(embedding.get("normalize_embeddings", True))
        self.batch_size = int(embedding.get("batch_size", 2))
        type(self).instances_created += 1

    def embed_query(self, query: str) -> list[float]:
        processed = process_query(query)
        vector = self.model.encode(
            [processed], batch_size=self.batch_size, show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings, convert_to_numpy=True,
        )[0]
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        if len(values) != int(self.manifest["embedding_dimension"]):
            raise ValueError("Generated query vector has unexpected dimension")
        return [float(x) for x in values]

    def retrieve(self, query: str, top_k: int | None = None, collections: Iterable[str] = ("text", "tables")) -> list[RetrievalResult]:
        processed = process_query(query)
        limit = int(self.config["retrieval"]["top_k"] if top_k is None else top_k)
        if limit < 1:
            raise ValueError("top_k must be at least 1")
        selected = list(collections)
        unknown = set(selected) - set(self.collections)
        if unknown:
            raise ValueError(f"Unknown collection selectors: {sorted(unknown)}")
        vector = self.embed_query(processed)
        candidates: list[RetrievalResult] = []
        for selector in selected:
            collection = self.collections[selector]
            count = collection.count()
            if not count:
                continue
            response = collection.query(
                query_embeddings=[vector], n_results=min(limit, count),
                include=["distances", "metadatas"],
            )
            for chunk_id, distance, metadata in zip(response["ids"][0], response["distances"][0], response["metadatas"][0]):
                metadata = metadata or {}
                candidates.append(RetrievalResult(
                    chunk_id=str(chunk_id), content_type=str(metadata.get("content_type") or ""),
                    document_id=str(metadata.get("document_id") or ""),
                    clause_number=str(metadata.get("clause_number") or ""),
                    clause_title=str(metadata.get("clause_title") or ""),
                    table_id=str(metadata.get("table_id") or ""),
                    pdf_page_start=int(metadata.get("pdf_page_start", 0)),
                    pdf_page_end=int(metadata.get("pdf_page_end", 0)),
                    citation=citation_from_metadata(metadata), distance=float(distance),
                    source_hash=str(metadata.get("source_hash") or ""),
                    collection=self.collection_names[selector],
                ))
        candidates.sort(key=lambda item: (item.distance, item.chunk_id))
        return candidates[:limit]

    def close(self) -> None:
        try:
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        finally:
            model = self.model
            self.model = None
            self.collections = {}
            self.client = None
            if self._temporary_root is not None:
                shutil.rmtree(self._temporary_root, ignore_errors=True)
                self._temporary_root = None
            del model
            gc.collect()

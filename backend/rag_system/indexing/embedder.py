from __future__ import annotations

from importlib import metadata
from typing import Any


class LocalBGEEmbedder:
    def __init__(self, config: dict[str, Any]):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers and torch are required; no fallback model will be used") from exc
        requested = config.get("device", "auto")
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
        self.model_name = config["model_name"]
        self.batch_size = int(config.get("batch_size", 8))
        self.precision = config.get("precision", "float32")
        self.normalize = bool(config.get("normalize_embeddings", True))
        self.show_progress = bool(config.get("show_progress", True))
        try:
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(self.precision)
            if dtype is None:
                raise ValueError(f"Unsupported embedding precision: {self.precision}")
            self.model = SentenceTransformer(
                self.model_name, device=self.device,
                trust_remote_code=bool(config.get("trust_remote_code", False)),
                local_files_only=bool(config.get("local_files_only", False)),
                model_kwargs={"torch_dtype": dtype},
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load exact embedding model {self.model_name!r}; no substitute was used: {exc}") from exc
        self.max_sequence_length = int(getattr(self.model, "max_seq_length", 0) or 0)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.library_version = metadata.version("sentence-transformers")

    def analyze_lengths(self, records: list[IndexableRecord]) -> list[dict]:
        texts = [record.embedding_text for record in records]
        tokenized = self.model.tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)
        affected: list[dict] = []
        for record, ids in zip(records, tokenized["input_ids"]):
            count = len(ids)
            if self.max_sequence_length and count > self.max_sequence_length:
                record.warnings.append(f"embedding input exceeds model maximum tokens: {count}>{self.max_sequence_length}; model will truncate")
                record.metadata["warnings"] = " | ".join(record.warnings)
                record.effective_processed_length = self.max_sequence_length
                affected.append({"chunk_id": record.chunk_id, "original_character_length": record.original_character_length, "original_token_count": count, "effective_token_count": self.max_sequence_length})
        return affected

    def encode(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=self.show_progress, normalize_embeddings=self.normalize, convert_to_numpy=True)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(f"Embedding failed with out-of-memory on {self.device}; reduce batch_size (currently {self.batch_size})") from exc
            raise
        return vectors.tolist()

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self.encode(texts)


# Circular type import kept after class definition for lightweight module import in tests.
from .models import IndexableRecord  # noqa: E402

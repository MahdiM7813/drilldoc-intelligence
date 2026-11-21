from __future__ import annotations
from dataclasses import dataclass, field
from rag_system.retrieval.schemas import RetrievalResult

@dataclass(frozen=True)
class QueryAnalysis:
    original: str
    expanded: str
    tokens: tuple[str, ...]
    phrases: tuple[str, ...]
    language: str
    intent: str
    preferred_content_types: tuple[str, ...] = ()

@dataclass
class Candidate:
    result: RetrievalResult
    text: str = ""
    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_score: float = 0.0
    keyword_score: float = 0.0
    metadata_score: float = 0.0
    phrase_score: float = 0.0
    final_score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    def diagnostics(self) -> dict:
        return {"chunk_id": self.result.chunk_id, "vector_rank": self.vector_rank, "keyword_rank": self.keyword_rank,
                "vector_score": round(self.vector_score, 6), "keyword_score": round(self.keyword_score, 6),
                "metadata_score": round(self.metadata_score, 6), "phrase_score": round(self.phrase_score, 6),
                "final_score": round(self.final_score, 6), "matched_terms": self.matched_terms}

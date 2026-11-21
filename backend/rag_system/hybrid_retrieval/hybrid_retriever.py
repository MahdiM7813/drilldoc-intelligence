from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
from rag_system.retrieval.retriever import LocalRetriever
from rag_system.retrieval.schemas import RetrievalResult
from .bm25_retriever import BM25Retriever
from .query_analyzer import QueryAnalyzer
from .reranker import HybridReranker
from .schemas import Candidate

class HybridRetriever:
    """RAG-04-compatible vector + BM25 retriever with deterministic reranking."""
    def __init__(self, config: dict[str, Any], manifest_path: Path, model: Any | None = None, client: Any | None = None):
        self.config = config
        self.vector_retriever = LocalRetriever(config["rag04"], manifest_path, model=model, client=client)
        self.analyzer = QueryAnalyzer()
        paths = [Path(path).resolve() for path in config["bm25"]["chunk_paths"]]
        self.bm25 = BM25Retriever(paths, self.vector_retriever.collection_names, config["bm25"].get("k1", 1.5), config["bm25"].get("b", .75))
        self.reranker = HybridReranker(config["reranking"])
        self.last_diagnostics: list[dict] = []
    def retrieve(self, query: str, top_k: int | None = None, collections: Iterable[str] = ("text", "tables")) -> list[RetrievalResult]:
        limit = int(self.config["hybrid"]["top_k"] if top_k is None else top_k)
        if limit < 1: raise ValueError("top_k must be at least 1")
        selected = tuple(collections)
        analysis = self.analyzer.analyze(query)
        pool = max(limit, int(self.config["hybrid"].get("candidate_pool", 15)))
        vector = self.vector_retriever.retrieve(analysis.expanded, top_k=pool, collections=selected)
        keyword = self.bm25.search(analysis.tokens, pool)
        allowed = {self.vector_retriever.collection_names[x] for x in selected}
        keyword = [item for item in keyword if item[0].collection in allowed]
        merged: dict[str, Candidate] = {}
        for rank, result in enumerate(vector, 1):
            merged[result.chunk_id] = Candidate(result=result, vector_rank=rank, vector_score=1/(1+result.distance))
        max_bm25 = max((score for _, score, _ in keyword), default=1.0)
        for rank, (result, score, text) in enumerate(keyword, 1):
            candidate = merged.setdefault(result.chunk_id, Candidate(result=result))
            candidate.keyword_rank, candidate.keyword_score, candidate.text = rank, score/max_bm25, text
        ranked = self.reranker.rerank(analysis, list(merged.values()))
        self.last_diagnostics = [x.diagnostics() for x in ranked[:limit]]
        return [x.result for x in ranked[:limit]]
    def embed_query(self, query: str) -> list[float]: return self.vector_retriever.embed_query(query)
    @property
    def collections(self): return self.vector_retriever.collections
    @property
    def model(self): return self.vector_retriever.model
    def close(self) -> None:
        self.vector_retriever.close()
        self.bm25 = None
        self.reranker = None
        self.last_diagnostics = []

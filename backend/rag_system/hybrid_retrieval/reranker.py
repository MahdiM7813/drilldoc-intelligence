from __future__ import annotations
from .query_analyzer import normalize_text
from .schemas import Candidate, QueryAnalysis

class HybridReranker:
    def __init__(self, config: dict): self.config = config
    def rerank(self, analysis: QueryAnalysis, candidates: list[Candidate]) -> list[Candidate]:
        rrf_k = float(self.config.get("rrf_k", 60))
        for candidate in candidates:
            rrf = 0.0
            if candidate.vector_rank is not None: rrf += self.config.get("vector_weight", 1.0)/(rrf_k+candidate.vector_rank)
            if candidate.keyword_rank is not None: rrf += self.config.get("keyword_weight", 1.2)/(rrf_k+candidate.keyword_rank)
            haystack = normalize_text(" ".join([candidate.text, candidate.result.clause_title, candidate.result.clause_number]))
            candidate.phrase_score = sum(normalize_text(p) in haystack for p in analysis.phrases)*self.config.get("phrase_boost", .05)
            candidate.matched_terms = sorted({t for t in analysis.tokens if t in set(haystack.split())})
            candidate.metadata_score = self._metadata_score(analysis, candidate)
            candidate.final_score = rrf + candidate.phrase_score + candidate.metadata_score
        return sorted(candidates, key=lambda x: (-x.final_score, x.result.distance, x.result.chunk_id))
    def _metadata_score(self, analysis: QueryAnalysis, candidate: Candidate) -> float:
        result, score = candidate.result, 0.0
        if analysis.intent == "definition" and result.content_type == "definition": score += self.config.get("definition_boost", .035)
        if any(normalize_text(p) in normalize_text(result.clause_title) for p in analysis.phrases): score += self.config.get("title_boost", .06)
        if {"lack", "fusion", "porosity", "pore", "تخلخل", "ذوب"}.intersection(analysis.tokens) and result.content_type == "table_segment":
            score += self.config.get("table_defect_boost", .02)
        return score

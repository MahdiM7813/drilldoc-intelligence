from __future__ import annotations
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from rag_system.retrieval.schemas import RetrievalResult
from .query_analyzer import tokenize

class BM25Retriever:
    """In-memory BM25 over existing chunks; no corpus writes."""
    def __init__(self, paths: list[Path], collection_names: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.collection_names = float(k1), float(b), collection_names
        records: dict[str, dict] = {}
        for path in paths:
            if not path.is_file(): raise FileNotFoundError(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get("content_type") != "table": records.setdefault(str(record["chunk_id"]), record)
        self.records = records; self.ids = sorted(records)
        self.tokens = {cid: tokenize(self._search_text(records[cid])) for cid in self.ids}
        self.term_freq = {cid: Counter(tokens) for cid, tokens in self.tokens.items()}
        self.lengths = {cid: len(tokens) for cid, tokens in self.tokens.items()}
        self.avgdl = sum(self.lengths.values()) / max(len(self.lengths), 1)
        df: Counter[str] = Counter()
        for tokens in self.tokens.values(): df.update(set(tokens))
        n = len(self.ids); self.idf = {term: math.log(1 + (n-freq+0.5)/(freq+0.5)) for term, freq in df.items()}

    @staticmethod
    def _search_text(record: dict) -> str:
        values = [record.get(x, "") for x in ("embedding_text", "normalized_text", "term", "definition_text",
                                                "clause_title", "clause_number", "table_id")]
        values.append(" ".join(record.get("row_labels") or []))
        return " ".join(str(x) for x in values if x)

    def search(self, query_tokens: tuple[str, ...], top_k: int) -> list[tuple[RetrievalResult, float, str]]:
        scores = []
        for cid in self.ids:
            score, dl = 0.0, self.lengths[cid]
            for term in query_tokens:
                tf = self.term_freq[cid].get(term, 0)
                if tf:
                    denom = tf + self.k1*(1-self.b+self.b*dl/max(self.avgdl, 1.0))
                    score += self.idf.get(term, 0.0)*tf*(self.k1+1)/denom
            if score > 0: scores.append((score, cid))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [(self._result(self.records[cid], score), score, self._search_text(self.records[cid]))
                for score, cid in scores[:top_k]]

    def _result(self, record: dict, score: float) -> RetrievalResult:
        content_type = str(record.get("content_type") or "")
        selector = "tables" if content_type.startswith("table") else "text"
        text = str(record.get("embedding_text") or "")
        return RetrievalResult(str(record["chunk_id"]), content_type, str(record.get("document_id") or ""),
            str(record.get("clause_number") or ""), str(record.get("clause_title") or ""), str(record.get("table_id") or ""),
            int(record.get("pdf_page_start", 0)), int(record.get("pdf_page_end", 0)), record.get("citation") or {},
            float(1/(1+score)), hashlib.sha256(text.encode("utf-8")).hexdigest(), self.collection_names[selector])

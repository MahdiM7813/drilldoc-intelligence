from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import shutil
import tempfile
import math
import re
from typing import Any, Callable


TOKEN = re.compile(r"\d+(?:[./-]\d+)*|[a-z]+(?:-[a-z0-9]+)*|[\u0600-\u06ff]+", re.I)


def tokenize(value: str) -> list[str]:
    return TOKEN.findall(str(value).replace("ي", "ی").replace("ك", "ک").casefold())


class Stage5EvidenceRetriever:
    """Read-only Stage-5 adapter using the frozen best-channel RRF policy."""
    def __init__(self, records: list[dict[str, Any]], search: Callable[[str], tuple[list[str], list[str]]], depth: int = 100):
        self.records = {x["record_id"]: x for x in records}
        self.search = search
        self.depth = depth

    def retrieve(self, query: str, document_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        return self.retrieve_variants([query], document_id, top_k)

    def retrieve_variants(self, queries: list[str], document_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        if document_id not in {x.get("document_id") for x in self.records.values()}:
            return []
        scores = Counter()
        for query in dict.fromkeys(x.strip() for x in queries if x.strip()):
            vector, bm25 = self.search(query)
            for weight, ids in ((1.0, vector), (1.15, bm25)):
                for rank, rid in enumerate(ids[:self.depth], 1):
                    if rid in self.records and self.records[rid].get("document_id") == document_id:
                        scores[rid] = max(scores[rid], weight / (20 + rank))
        ranked = sorted(scores, key=lambda rid: (-scores[rid], rid))[:top_k]
        return [{**self.records[rid], "retrieval_score": scores[rid]} for rid in ranked]


class Stage5SearchBackend:
    def __init__(self, corpus_path: Path, chroma_path: Path):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.records = [json.loads(x) for x in corpus_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        self.ids = [x["record_id"] for x in self.records]
        self.docs = [Counter(tokenize(x["retrieval_text"])) for x in self.records]
        self.df = Counter(token for doc in self.docs for token in doc)
        self.avgdl = sum(sum(doc.values()) for doc in self.docs) / len(self.docs)
        self.model = SentenceTransformer("BAAI/bge-m3", device="cpu", local_files_only=True)
        self.model.max_seq_length = 256
        self._temporary_root = Path(tempfile.mkdtemp(prefix='weld_stage5_retrieval_'))
        client_directory = self._temporary_root / 'chroma'
        shutil.copytree(chroma_path, client_directory)
        client = chromadb.PersistentClient(path=str(client_directory))
        self.collections = [client.get_collection(x) for x in ("text_accuracy_candidate", "table_accuracy_candidate", "visual_accuracy_candidate")]

    def search(self, query: str) -> tuple[list[str], list[str]]:
        embedding = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).tolist()[0]
        vector_rows = []
        for collection in self.collections:
            result = collection.query(query_embeddings=[embedding], n_results=collection.count())
            vector_rows.extend((float(distance), rid) for rid, distance in zip(result["ids"][0], result["distances"][0]))
        vector = [rid for _, rid in sorted(vector_rows, key=lambda x: (x[0], x[1]))]
        terms = Counter(tokenize(query)); query_tokens = tokenize(query)
        ngrams = [" ".join(query_tokens[i:i+n]) for n in range(2, min(6, len(query_tokens)+1)) for i in range(len(query_tokens)-n+1)]
        lexical = []
        for rid, doc, record in zip(self.ids, self.docs, self.records):
            dl, score = sum(doc.values()), 0.0
            for term, qf in terms.items():
                tf = doc[term]
                if not tf: continue
                idf = math.log(1 + (len(self.docs) - self.df[term] + .5) / (self.df[term] + .5))
                score += qf * idf * tf * 2.5 / (tf + 1.5 * (1 - .75 + .75 * dl / self.avgdl))
            normalized_doc = " ".join(tokenize(record["retrieval_text"]))
            score += sum(2.0 * len(x.split()) for x in ngrams if x in normalized_doc)
            if score: lexical.append((score, rid))
        bm25 = [rid for _, rid in sorted(lexical, key=lambda x: (-x[0], x[1]))]
        return vector, bm25

    def retriever(self) -> Stage5EvidenceRetriever:
        return Stage5EvidenceRetriever(self.records, self.search)

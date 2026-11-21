from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

from rag_system.retrieval.retriever import LocalRetriever
from rag_system.retrieval.schemas import RetrievalResult

from .citation_formatter import citation_for_result, preserve_allowed_citations
from .llm_provider import LLMProvider
from .prompt_template import FORMATTER_SYSTEM, REASONING_SYSTEM, formatter_prompt, reasoning_prompt
from .schemas import AnswerResponse


class AnswerGenerator:
    def __init__(
        self,
        retriever: LocalRetriever,
        reasoning_factory: Callable[[], LLMProvider],
        formatter_factory: Callable[[], LLMProvider],
        config: dict[str, Any],
    ):
        self.retriever = retriever
        self.reasoning_factory = reasoning_factory
        self.formatter_factory = formatter_factory
        self.config = config

    def _evidence(self, results: list[RetrievalResult]) -> list[dict[str, Any]]:
        collection_by_name = {value: key for key, value in self.retriever.collection_names.items()}
        blocks: list[dict[str, Any]] = []
        used = 0
        maximum = int(self.config["answering"].get("max_evidence_characters", 18000))
        for result in results:
            selector = collection_by_name[result.collection]
            fetched = self.retriever.collections[selector].get(ids=[result.chunk_id], include=["documents"])
            if fetched.get("ids") != [result.chunk_id] or not fetched.get("documents"):
                raise RuntimeError(f"Evidence document could not be read: {result.chunk_id}")
            text = str(fetched["documents"][0])
            if blocks and used + len(text) > maximum:
                continue
            blocks.append({
                "chunk_id": result.chunk_id,
                "citation": result.citation["display"],
                "content_type": result.content_type,
                "text": text,
            })
            used += len(text)
        if not blocks:
            raise RuntimeError("No evidence documents available")
        return blocks

    @staticmethod
    def _dispose(provider: Any) -> None:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
        del provider
        gc.collect()

    def answer_many(self, questions: list[str], top_k: int | None = None) -> list[AnswerResponse]:
        packages = []
        for question in questions:
            results = self.retriever.retrieve(question, top_k=top_k)
            packages.append({
                "question": question,
                "results": results,
                "evidence": self._evidence(results),
                "allowed": [citation_for_result(item) for item in results],
                "standard": results[0].citation["display"].split(",", 1)[0] if results else "",
            })

        # Release BGE and the temporary Chroma copy before loading either LLM.
        self.retriever.close()
        if hasattr(self.retriever, "model"):
            self.retriever.model = None
        gc.collect()

        reasoning = self.reasoning_factory()
        try:
            drafts = [reasoning.generate_text(reasoning_prompt(item["question"], item["evidence"]), REASONING_SYSTEM) for item in packages]
        finally:
            self._dispose(reasoning)

        formatter = self.formatter_factory()
        try:
            candidates = [
                formatter.generate_json(
                    formatter_prompt(item["question"], draft, item["standard"], item["allowed"]), FORMATTER_SYSTEM
                )
                for item, draft in zip(packages, drafts)
            ]
        finally:
            self._dispose(formatter)

        responses = []
        for item, candidate in zip(packages, candidates):
            compact_ids = candidate.pop("citation_chunk_ids", None)
            if compact_ids is not None:
                candidate["citations"] = [{"chunk_id": str(chunk_id)} for chunk_id in compact_ids]
            candidate["question"] = item["question"]
            candidate["standard"] = item["standard"]
            candidate["citations"] = preserve_allowed_citations(candidate.get("citations", []), item["results"])
            response = AnswerResponse.model_validate(candidate)
            if self.config["answering"].get("require_citations", True) and response.confidence != "insufficient" and not response.citations:
                raise ValueError("Formatter produced a supported answer without an allowed citation")
            responses.append(response)
        return responses

    def answer(self, question: str, top_k: int | None = None) -> AnswerResponse:
        return self.answer_many([question], top_k=top_k)[0]

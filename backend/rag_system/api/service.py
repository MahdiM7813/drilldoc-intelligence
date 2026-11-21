from __future__ import annotations
from rag_system.runtime_config import data_root, resolve_config_paths

import asyncio
import gc
import json
import logging
from pathlib import Path
import re
import time
import os
import threading
import tempfile
import uuid
from typing import Any, Callable, Protocol

from rag_system.hybrid_retrieval.hybrid_retriever import HybridRetriever
from rag_system.hybrid_retrieval.run import DEFAULT_CONFIG as DEFAULT_HYBRID_CONFIG
from rag_system.hybrid_retrieval.run import load_config as load_hybrid_config
from rag_system.retrieval.schemas import RetrievalResult

from .schemas import EvidenceItem, GroundedAnswer, RetrievalFilters, RetrievalResponse
from rag_system.answering.llm_provider import LLMProvider
from rag_system.answering.context_selection import select_context
from rag_system.answering.evidence_packing import pack_context
from rag_system.answering.prompt_template import ANALYTICAL_SYSTEM, FORMATTER_SYSTEM, REASONING_SYSTEM, analytical_prompt, formatter_prompt, reasoning_prompt

LOGGER = logging.getLogger(__name__)


INSUFFICIENT_ANSWER = "Insufficient evidence in the retrieved standard excerpts."
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?\s*%?")
ORDERED_LIST_MARKER_PATTERN = re.compile(r"(?m)^\s*\d+[.)]\s+")
QUALITY_LEVEL_PATTERN = re.compile(r"\bquality\s+level\s+([BCD])\b", re.IGNORECASE)
ANALYTICAL_PATTERN = re.compile(r"(?:most|least|which\s+(?:section|formation)|highest|lowest|compare|summari[sz]e|main\s+problems|frequent|downtime|NPT|بیشترین|کمترین|کدام\s+(?:بخش|لایه|سازند)|مجموع|مقایسه|مشکلات\s+(?:اصلی|عملیاتی)|جمع[‌ ]?بندی)", re.I)
PROBLEM_PATTERN = re.compile(r"stuck|tight\s+hole|loss(?:es)?|instab|cav(?:ing|ings)|washout|failure|repair|waiting|npt|back\s*ream|circulation|low\s*gpm|problem|گیر|مشکل|خرابی|تعمیر|انتظار|اتلاف|فشار", re.I)
SECTION_PATTERN = re.compile(r"(?:\d+\s*[-/]?\s*\d+\s*/\s*\d+|\d+\s*[-/]?\s*\d+)(?:\s*[\"”']|\s+inch|\s+in)", re.I)


def is_analytical_query(query: str) -> bool:
    return bool(ANALYTICAL_PATTERN.search(query))


def effective_generation_tokens(config: dict[str, Any], english_only: bool = False) -> int:
    """Return the configured completion budget without silently truncating it."""
    del english_only
    return max(1, int(config.get("max_new_tokens", 128)))


def semantic_number_text(text: str) -> str:
    """Remove presentation-only ordered-list markers before numeric grounding."""
    return ORDERED_LIST_MARKER_PATTERN.sub("", text)


def build_analytical_package(evidence: list[EvidenceItem], limit: int = 6) -> tuple[str, list[EvidenceItem]]:
    """Compact source-backed event summaries without inventing classifications."""
    selected = [item for item in evidence if PROBLEM_PATTERN.search(item.text)]
    selected = selected[:min(limit, 4)] or evidence[:min(limit, 4)]
    groups: dict[str, dict[str, Any]] = {}
    lines = ["ANALYTICAL SUMMARY", "Problem records are counted only when source wording contains a problem/event term."]
    for index, item in enumerate(selected, 1):
        source_id = f"S{index}"
        text = re.sub(r"\s+", " ", item.text).strip()
        section = next(iter(SECTION_PATTERN.findall(text)), "section not explicit")
        group = groups.setdefault(section, {"count": 0, "ids": []})
        group["count"] += 1
        group["ids"].append(source_id)
        excerpt = text[:300]
        lines.append(f"{source_id} (chunk_id={item.chunk_id}): hole section={section}; event_records={group['count']}; excerpt={excerpt}")
    lines.append("SECTION AGGREGATION")
    for section, value in groups.items():
        lines.append(f"{section}: source-backed problem records={value['count']}; evidence={', '.join(value['ids'])}")
    lines.append("FORMATION MAPPING: insufficient unless an evidence excerpt explicitly names the formation and its association.")
    return "\n".join(lines), selected


class InvalidQuery(ValueError):
    pass


class RetrievalUnavailable(RuntimeError):
    pass


class GenerationBusy(RuntimeError):
    pass


class RetrievalServiceProtocol(Protocol):
    async def query(self, query: str, top_k: int | None = None, filters: RetrievalFilters | None = None,
                    standard_id: str | None = None) -> RetrievalResponse: ...


class Stage5RetrievalService:
    """API adapter over the frozen Stage-5 606-record corpus and index."""
    def __init__(self, corpus_path: Path, chroma_path: Path, top_k: int = 10, document_id: str = "drilldoc_default"): 
        from rag_system.answering.stage5_retriever import Stage5SearchBackend
        self.backend = Stage5SearchBackend(corpus_path, chroma_path).retriever()
        self.top_k = top_k
        self.document_id = document_id

    async def query(self, query: str, top_k: int | None = None, filters: RetrievalFilters | None = None, standard_id: str | None = None) -> RetrievalResponse:
        if standard_id is not None: raise InvalidQuery("standard_id routing is not configured")
        initial_pool = int(top_k or self.top_k)
        multi_intent = sum(bool(re.search(pattern, query, re.I)) for pattern in (r"12\s*[¼1/]?\s*4|12\s*[-/]?\s*1\s*(?:[/\s]*4)?", r"hole\s+condition|tight\s+hole|hard\s+back\s*ream", r"stuck|sticking", r"back[- ]?off|sidetrack", r"\bnpt\b", r"drilling\s+fluid", r"drilling\s+data")) >= 2
        effective_pool = max(initial_pool, 160) if multi_intent else initial_pool
        self.last_candidate_pool_initial = initial_pool
        self.last_candidate_pool_effective = effective_pool
        rows = await asyncio.to_thread(self.backend.retrieve, query, self.document_id, effective_pool)
        evidence=[]
        for row in rows:
            page = row.get("pdf_page_number") or (row.get("source_pages") or [None])[0]
            modality = row.get("modality", "TEXT")
            evidence.append(EvidenceItem(chunk_id=str(row["record_id"]), document=str(row["document_id"]), text=str(row.get("retrieval_text", "")), score=float(row.get("retrieval_score", 0.0)), metadata={**row, "modality": modality, "pdf_page_start": page, "record_id": row["record_id"]}, citation={"display": f"page {page}"}))
        return RetrievalResponse(query=query, evidence=evidence)

    def close(self):
        close=getattr(self.backend, "close", None)
        if callable(close): close()


class ChunkTextStore:
    """Read-only lookup for text belonging to retrieved chunk IDs."""

    def __init__(self, chunk_paths: list[Path]):
        self._text: dict[str, str] = {}
        for path in chunk_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                chunk_id = str(record.get("chunk_id") or "")
                if chunk_id and chunk_id not in self._text:
                    self._text[chunk_id] = str(record.get("normalized_text") or record.get("source_text") or "")

    def get(self, chunk_id: str) -> str:
        if chunk_id not in self._text:
            raise RetrievalUnavailable(f"Retrieved chunk text is unavailable: {chunk_id}")
        return self._text[chunk_id]


class RetrievalService:
    """Thin adapter: API request -> existing HybridRetriever -> citation-preserved response."""

    def __init__(self, retriever: HybridRetriever, text_store: ChunkTextStore, default_top_k: int, timeout_seconds: float):
        self.retriever = retriever
        self.text_store = text_store
        self.default_top_k = int(default_top_k)
        self.timeout_seconds = float(timeout_seconds)

    async def query(self, query: str, top_k: int | None = None, filters: RetrievalFilters | None = None,
                    standard_id: str | None = None) -> RetrievalResponse:
        if standard_id is not None:
            raise InvalidQuery("standard_id routing is not configured for this retrieval service")
        normalized = query.strip() if isinstance(query, str) else ""
        if not normalized:
            raise InvalidQuery("query must not be empty")
        limit = self.default_top_k if top_k is None else top_k
        collections = tuple(filters.collections) if filters and filters.collections else ("text", "tables")
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(self.retriever.retrieve, normalized, limit, collections),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise RetrievalUnavailable("retrieval timed out") from exc
        except (InvalidQuery, RetrievalUnavailable):
            raise
        except Exception as exc:
            raise RetrievalUnavailable("retrieval backend unavailable") from exc
        evidence = [self._map_result(result) for result in results]
        return RetrievalResponse(query=normalized, evidence=evidence)

    def _map_result(self, result: RetrievalResult) -> EvidenceItem:
        return EvidenceItem(
            chunk_id=result.chunk_id,
            document=result.document_id,
            text=self.text_store.get(result.chunk_id),
            score=result.distance,
            metadata={
                "score_type": "cosine_distance",
                "content_type": result.content_type,
                "clause_number": result.clause_number,
                "clause_title": result.clause_title,
                "table_id": result.table_id,
                "pdf_page_start": result.pdf_page_start,
                "pdf_page_end": result.pdf_page_end,
                "source_hash": result.source_hash,
                "collection": result.collection,
            },
            citation=dict(result.citation),
        )

    def close(self) -> None:
        self.retriever.close()


class LLMEnhancedRetrievalService:
    """Add grounded reasoning and structured normalization to production retrieval."""

    def __init__(self, base: RetrievalServiceProtocol,
                 reasoning_factory: Callable[[], LLMProvider],
                 structure_factory: Callable[[], LLMProvider],
                 plain_text_only: bool = False,
                 retrieval_factory: Callable[[], RetrievalServiceProtocol] | None = None,
                 release_each_request: bool = False,
                 generation_timeout_seconds: float = 120.0,
                 english_only: bool = False,
                 trace_enabled: bool = False) -> None:
        self.base = base
        self.reasoning_factory = reasoning_factory
        self.structure_factory = structure_factory
        self.plain_text_only = plain_text_only
        self._reasoning_provider: LLMProvider | None = None
        self._structure_provider: LLMProvider | None = None
        self.retrieval_factory = retrieval_factory
        self.release_each_request = release_each_request
        self.generation_timeout_seconds = generation_timeout_seconds
        self.english_only = english_only
        self.trace_enabled = trace_enabled
        self._trace_output_disabled = False
        self.last_trace: dict[str, Any] = {}
        self._generation_gate = threading.Lock()

    def _trace_flush(self) -> None:
        if not self.trace_enabled or self._trace_output_disabled:
            return
        path = data_root() / "traces" / f"trace_{uuid.uuid4().hex}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.last_trace, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            self._trace_output_disabled = True
            LOGGER.warning("Trace output disabled after a write error: %s", type(exc).__name__)

    def _run_generation(self, response: RetrievalResponse, request_id: str) -> GroundedAnswer:
        try:
            return self._answer(response)
        finally:
            if self.trace_enabled:
                self.last_trace.update({"worker_exit": True, "worker_exit_time": time.time(), "request_id": request_id})
                self._trace_flush()
            self._generation_gate.release()

    async def query(self, query: str, top_k: int | None = None,
                    filters: RetrievalFilters | None = None,
                    standard_id: str | None = None) -> RetrievalResponse:
        if self.english_only and any("\u0600" <= char <= "\u06ff" for char in query):
            raise InvalidQuery("English-only forensic mode does not accept Persian questions")
        total_started = time.perf_counter()
        request_id = str(uuid.uuid4())
        if self.base is None:
            if self.retrieval_factory is None:
                raise RetrievalUnavailable("retrieval backend unavailable")
            self.base = self.retrieval_factory()
        base = self.base
        LOGGER.info("RAG_STATE retrieval_started")
        try:
            retrieval_started = time.perf_counter()
            response = await base.query(query, top_k, filters, standard_id)
            retrieval_seconds = time.perf_counter() - retrieval_started
            if self.trace_enabled:
                self.last_trace = {
                    "question": query,
                    "qwen_call_count": 0,
                    "llama_used": False,
                    "fallback_used": False,
                    "retrieval_latency_seconds": retrieval_seconds,
                    "retrieved_candidates": [item.model_dump() for item in response.evidence],
                    "candidate_pool_initial": getattr(base, "last_candidate_pool_initial", top_k),
                    "candidate_pool_effective": getattr(base, "last_candidate_pool_effective", top_k),
                }
            LOGGER.info("RAG_STATE evidence_materialized evidence=%d", len(response.evidence))
            if self.release_each_request:
                self._close(base)
                self.base = None
                base = None
                gc.collect()
                LOGGER.info("RAG_STATE retrieval_released")
            if not self._generation_gate.acquire(blocking=False):
                raise GenerationBusy("LLM generation busy; retry after the active generation worker exits")
            self._active_request_id = request_id
            answer = await asyncio.wait_for(
                asyncio.to_thread(self._run_generation, response, request_id),
                timeout=self.generation_timeout_seconds,
            )
        except Exception as exc:
            LOGGER.exception("RAG_FAILURE request_failed")
            if self.release_each_request and self.base is base:
                self._close(base)
                self.base = None
                base = None
                gc.collect()
            if isinstance(exc, GenerationBusy):
                raise
            if isinstance(exc, asyncio.TimeoutError):
                raise RetrievalUnavailable("LLM generation timed out") from exc
            raise RetrievalUnavailable("LLM-grounded answering unavailable") from exc
        finally:
            if self.release_each_request:
                self._close(self._reasoning_provider)
                self._reasoning_provider = None
                LOGGER.info("RAG_STATE llm_released")
        if self.trace_enabled:
            self.last_trace["total_latency_seconds"] = time.perf_counter() - total_started
        return response.model_copy(update={"answer": answer})

    def _answer(self, response: RetrievalResponse) -> GroundedAnswer:
        started = time.perf_counter()
        if self.english_only and self._reasoning_provider is None:
            self._reasoning_provider = self.reasoning_factory()
        # Stage-5 may return a broad candidate set. Keep generation context
        # bounded independently of retrieval depth and never send raw table
        # payloads wholesale to the model.
        context_items = select_context(response.query, response.evidence)
        tokenizer = getattr(self._reasoning_provider, "tokenize", None)
        packed_context = pack_context(response.query, context_items, budget_tokens=400, tokenizer=tokenizer)
        blocks = [{
            "chunk_id": item.chunk_id,
            "citation": item.citation.get("display") or json.dumps(item.citation, ensure_ascii=False),
            "content_type": str(item.metadata.get("content_type", "")),
            "text": packed.text,
        } for packed in packed_context.blocks for item in [packed.item]]
        packed_items = [packed.item for packed in packed_context.blocks]
        if self.trace_enabled:
            self.last_trace["selected_record_ids"] = [item.chunk_id for item in context_items]
            self.last_trace["selected_context"] = json.dumps(blocks, ensure_ascii=False, indent=2)
            self.last_trace["packed_context"] = "\n".join(block["text"] for block in blocks)
            self.last_trace["packed_context_tokens"] = packed_context.estimated_tokens
            self.last_trace["packed_context_budget_tokens"] = packed_context.budget_tokens
            self.last_trace["packed_omitted_record_ids"] = [
                item.chunk_id for item in context_items
                if item.chunk_id not in {packed.item.chunk_id for packed in packed_context.blocks}
            ]
            self.last_trace["packed_omission_reason"] = "complete evidence unit exceeded measured available context budget"
        LOGGER.info("RAG_TIMING evidence_items=%d evidence_chars=%d", len(blocks), sum(len(x["text"]) for x in blocks))
        allowed = [{"chunk_id": item.chunk_id} for item in response.evidence]
        if self.english_only:
            return self._english_only_answer(response, blocks, started, packed_items)
        if self.plain_text_only:
            return self._plain_text_answer(response, blocks)
        standard = next((str(item.metadata.get("standard_name") or item.document) for item in response.evidence), "")
        analytical = is_analytical_query(response.query)
        analytical_items = response.evidence
        if self._reasoning_provider is None:
            self._reasoning_provider = self.reasoning_factory()
        if analytical:
            package, analytical_items = build_analytical_package(response.evidence)
            LOGGER.info("RAG_TIMING analytical_package_chars=%d selected_sources=%d", len(package), len(analytical_items))
            generation_prompt = analytical_prompt(response.query, package)
            generation_system = ANALYTICAL_SYSTEM
        else:
            generation_prompt = reasoning_prompt(response.query, blocks)
            generation_system = REASONING_SYSTEM
        if self.trace_enabled:
            self.last_trace["llama_prompt"] = f"SYSTEM:\n{generation_system}\n\nUSER:\n{generation_prompt}"
            self.last_trace["llama_used"] = True
        draft = self._reasoning_provider.generate_text(generation_prompt, generation_system)
        if self.trace_enabled:
            self.last_trace["raw_llama_output"] = draft
        if self._structure_provider is None:
            self._structure_provider = self.structure_factory()
        formatting_prompt = formatter_prompt(response.query, draft, standard, allowed)
        candidate = self._structure_provider.generate_json(formatting_prompt, FORMATTER_SYSTEM)
        if self.trace_enabled:
            self.last_trace["formatter_prompt"] = f"SYSTEM:\n{FORMATTER_SYSTEM}\n\nUSER:\n{formatting_prompt}"
            self.last_trace["formatter_raw_output"] = dict(candidate)
        LOGGER.info("RAG_TIMING answer_total_seconds=%.3f analytical=%s", time.perf_counter() - started, analytical)
        requested = candidate.pop("citation_chunk_ids", [])
        if not isinstance(requested, list):
            raise ValueError("citation_chunk_ids must be a list")
        by_id = {item.chunk_id: item for item in response.evidence}
        citations = []
        for raw_id in requested:
            chunk_id = str(raw_id)
            item = by_id.get(chunk_id)
            if item is None or any(value["chunk_id"] == chunk_id for value in citations):
                continue
            citations.append({
                "document": item.document,
                "clause": str(item.citation.get("clause") or item.metadata.get("clause_number") or ""),
                "page": str(item.citation.get("printed_page_start") or item.metadata.get("pdf_page_start") or ""),
                "chunk_id": chunk_id,
            })
        answer = GroundedAnswer.model_validate({
            "answer": candidate.get("answer"), "confidence": candidate.get("confidence"),
            "citations": citations,
        })
        final_answer = self._validate_answer(response, answer)
        if self.trace_enabled:
            self.last_trace["llm_latency_seconds"] = time.perf_counter() - started
            self.last_trace["final_output"] = final_answer.model_dump()
            provider_trace = getattr(self._reasoning_provider, "last_generation", None)
            if provider_trace:
                self.last_trace["generation"] = dict(provider_trace)
            self._trace_flush()
        return final_answer

    @staticmethod
    def _context_excerpt(item: EvidenceItem) -> str:
        """Render structured table evidence without losing field/value identity."""
        text = re.sub(r"\s+", " ", item.text).strip()
        try:
            payload = json.loads(item.text)
        except (TypeError, json.JSONDecodeError):
            return text[:250]
        if not isinstance(payload, dict):
            return text[:250]
        parts = []
        for key in ("logical_table_id", "section", "row_id", "entity_type", "raw_cell_values", "field_value_pairs", "units"):
            value = payload.get(key, item.metadata.get(key))
            if value not in (None, "", [], {}):
                parts.append(f"{key}={value}")
        if parts:
            return "; ".join(parts)[:250]
        return text[:250]

    def _english_only_answer(self, response: RetrievalResponse, blocks: list[dict], started: float,
                             packed_items: list[EvidenceItem] | None = None) -> GroundedAnswer:
        if not response.evidence:
            answer = GroundedAnswer(answer=INSUFFICIENT_ANSWER, confidence="insufficient")
            if self.trace_enabled:
                self.last_trace.update({"llama_used": False, "raw_llama_output": "", "final_output": answer.model_dump(),
                                        "llm_latency_seconds": 0.0})
            return answer
        rendered = [f"[S{index}] chunk_id={block['chunk_id']}; {block['citation']}\n{block['text']}" for index, block in enumerate(blocks, 1)]
        system = (
            "Answer the English question naturally using only the supplied evidence. Use the length needed by the question. "
            "Preserve exact numbers, units, field meanings, and entity types. An NPT item ID is not a hole size; "
            "a hole section is not a formation. For problem questions, clearly separate observed/reporting events from recommendations "
            "and possible consequences. Do not infer that an event occurred or assign its cause from a prevention recommendation. "
            "Avoid repeated headings such as Direct Answer and Reported Problems. "
            "Cite supporting source labels such as [S1]. If evidence is insufficient, say so."
        )
        prompt = f"EVIDENCE:\n\n{chr(10).join(rendered)}\n\nQUESTION:\n{response.query}"
        if self._reasoning_provider is None:
            self._reasoning_provider = self.reasoning_factory()
        if self.trace_enabled:
            request_id = getattr(self, "_active_request_id", "unknown")
            tokenizer = getattr(self._reasoning_provider, "tokenize", None)
            prompt_tokens = len(tokenizer(prompt)) if callable(tokenizer) else None
            settings = getattr(self._reasoning_provider, "config", {})
            self.last_trace.update({"request_id": request_id, "process_id": os.getpid(), "worker_exit": False,
                                    "prompt_tokens_pre_generation": prompt_tokens,
                                    "generation_start_time": time.time(),
                                    "effective_generation_settings": {"max_new_tokens": settings.get("max_new_tokens"),
                                                                         "temperature": settings.get("temperature"),
                                                                         "n_ctx": settings.get("n_ctx")}})
            self.last_trace["llama_prompt"] = f"SYSTEM:\n{system}\n\nUSER:\n{prompt}"
            self.last_trace["llama_used"] = True
            self._trace_flush()
            LOGGER.info("RAG_GENERATION_START request_id=%s pid=%s", request_id, os.getpid())
        text = self._reasoning_provider.generate_text(prompt, system).strip()
        if self.trace_enabled:
            self.last_trace["raw_llama_output"] = text
        LOGGER.info("RAG_VALIDATION raw_answer=%r chars=%d", text, len(text))
        cited_indexes = {int(value) for value in re.findall(r"\[S(\d+)\]", text)}
        source_items = packed_items or response.evidence[:len(blocks)]
        selected = [source_items[index - 1] for index in sorted(cited_indexes) if 1 <= index <= len(source_items)]
        selected = selected or source_items
        citations = [{
            "document": item.document,
            "clause": str(item.citation.get("clause") or item.metadata.get("clause_number") or ""),
            "page": str(item.citation.get("printed_page_start") or item.metadata.get("pdf_page_start") or ""),
            "chunk_id": item.chunk_id,
        } for item in selected]
        candidate = GroundedAnswer(answer=text or INSUFFICIENT_ANSWER, citations=citations,
                                   confidence="medium" if text else "insufficient")
        final_answer = self._validate_answer(response, candidate)
        LOGGER.info("RAG_VALIDATION final_confidence=%s final_answer=%r", final_answer.confidence, final_answer.answer)
        if self.trace_enabled:
            self.last_trace["llm_latency_seconds"] = time.perf_counter() - started
            self.last_trace["final_output"] = final_answer.model_dump()
            provider_trace = getattr(self._reasoning_provider, "last_generation", None)
            if provider_trace:
                self.last_trace["generation"] = dict(provider_trace)
            self._trace_flush()
        return final_answer

    def _plain_text_answer(self, response: RetrievalResponse, blocks: list[dict]) -> GroundedAnswer:
        if not response.evidence:
            return GroundedAnswer(answer="اطلاعات کافی برای پاسخ به این سؤال در اسناد بازیابی‌شده وجود ندارد.",
                                  confidence="insufficient")
        rendered = []
        for index, block in enumerate(blocks, 1):
            rendered.append(f"[S{index}] Page: {block['citation']}\nContent:\n{block['text']}")
        system = (
            "You are a document question-answering assistant. Answer only from the supplied context. "
            "Do not use outside knowledge or guess numbers. Preserve numbers and units exactly. "
            "If evidence is insufficient, say so. Answer in the user's language. "
            "You may cite only source IDs that exist, such as [S1] or [S2]."
        )
        prompt = f"CONTEXT:\n\n{chr(10).join(rendered)}\n\nQUESTION:\n{response.query}"
        if self._reasoning_provider is None:
            self._reasoning_provider = self.reasoning_factory()
        text = self._reasoning_provider.generate_text(prompt, system).strip()
        if not text:
            raise RuntimeError("LLM_GENERATION_FAILED: empty answer")
        cited_indexes = [int(value) for value in re.findall(r"\[S(\d+)\]", text)]
        citations = []
        selected = {value for value in cited_indexes if 1 <= value <= len(response.evidence)}
        evidence_for_sources = [response.evidence[value - 1] for value in sorted(selected)] or response.evidence
        for item in evidence_for_sources:
            citations.append({
                "document": item.document,
                "clause": str(item.citation.get("clause") or item.metadata.get("clause_number") or ""),
                "page": str(item.citation.get("pdf_page_start") or item.metadata.get("pdf_page_start") or ""),
                "chunk_id": item.chunk_id,
            })
        return GroundedAnswer(answer=text, citations=citations, confidence="medium")

    @staticmethod
    def _validate_answer(response: RetrievalResponse, answer: GroundedAnswer) -> GroundedAnswer:
        text = answer.answer.strip()
        cited_ids = {citation.chunk_id for citation in answer.citations}
        cited_evidence = [item for item in response.evidence if item.chunk_id in cited_ids]
        evidence_text = "\n".join(item.text for item in cited_evidence)
        truncated = not text.endswith((".", "?", "!"))
        unsupported_numbers = any(
            match.group(0).strip() not in evidence_text
            for match in NUMBER_PATTERN.finditer(semantic_number_text(text))
        )
        requested_level = QUALITY_LEVEL_PATTERN.search(response.query)
        ambiguous_table_claim = bool(
            requested_level
            and NUMBER_PATTERN.search(text)
            and any(str(item.metadata.get("content_type")) == "table_segment" for item in cited_evidence)
        )
        if not cited_evidence or unsupported_numbers or ambiguous_table_claim:
            LOGGER.info(
                "RAG_VALIDATION rejected cited=%s truncated=%s unsupported_numbers=%s ambiguous_table=%s",
                bool(cited_evidence), truncated, unsupported_numbers, ambiguous_table_claim,
            )
            return GroundedAnswer(
                answer=INSUFFICIENT_ANSWER,
                confidence="insufficient",
                citations=answer.citations,
            )
        return answer

    @staticmethod
    def _close(provider: Any) -> None:
        close = getattr(provider, "close", None)
        if callable(close):
            close()

    def close(self) -> None:
        self._close(self._reasoning_provider)
        self._reasoning_provider = None
        self._close(self._structure_provider)
        self._structure_provider = None
        close = getattr(self.base, "close", None)
        if callable(close):
            close()


def build_retrieval_service(api_config: dict[str, Any]) -> RetrievalService:
    hybrid_path = Path(api_config.get("hybrid_config_path") or DEFAULT_HYBRID_CONFIG).resolve()
    hybrid_config = resolve_config_paths(load_hybrid_config(hybrid_path))
    index_path = Path(api_config.get("vector_index_path", data_root() / "indexes" / "chroma"))
    if not (index_path / "chroma.sqlite3").is_file():
        raise RetrievalUnavailable("Configured vector index is not available.")
    retriever = HybridRetriever(hybrid_config, Path(hybrid_config["manifest_path"]))
    chunk_paths = [Path(path) for path in hybrid_config["bm25"]["chunk_paths"]]
    return RetrievalService(
        retriever=retriever,
        text_store=ChunkTextStore(chunk_paths),
        default_top_k=int(api_config["top_k_default"]),
        timeout_seconds=float(api_config["timeout_seconds"]),
    )


def build_llm_enhanced_service(base: RetrievalServiceProtocol, plain_text_only: bool = False,
                               retrieval_factory: Callable[[], RetrievalServiceProtocol] | None = None,
                               release_each_request: bool = False,
                               english_only: bool = True,
                               trace_enabled: bool = False) -> LLMEnhancedRetrievalService:
    from rag_system.answering.run import DEFAULT_CONFIG, build_provider, load_config

    config = load_config(DEFAULT_CONFIG)
    # Llama is the primary answer generator. Qwen remains available for the
    # translation/fallback roles and is not used to replace normal answering.
    llama_config = dict(config["providers"]["reasoning"])
    llama_config["max_new_tokens"] = effective_generation_tokens(llama_config, english_only)
    config["providers"]["reasoning"] = llama_config
    config["providers"]["formatter"] = llama_config
    shared_provider: LLMProvider | None = None
    def provider_factory() -> LLMProvider:
        nonlocal shared_provider
        if shared_provider is None:
            shared_provider = build_provider(config["providers"]["reasoning"])
        return shared_provider
    return LLMEnhancedRetrievalService(
        base,
        provider_factory,
        provider_factory,
        plain_text_only=plain_text_only,
        retrieval_factory=retrieval_factory,
        release_each_request=release_each_request,
        english_only=english_only,
        trace_enabled=trace_enabled,
        generation_timeout_seconds=float(config.get("answering", {}).get("generation_timeout_seconds", 120.0)),
    )

from __future__ import annotations

import unittest
import asyncio
import os
import tempfile
import gc
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_system.api.app import create_app
from rag_system.api.schemas import EvidenceItem, RetrievalResponse
from rag_system.api.service import InvalidQuery, RetrievalUnavailable
from rag_system.api.service import LLMEnhancedRetrievalService, RetrievalService, effective_generation_tokens
from rag_system.api.schemas import GroundedAnswer
from rag_system.api.schemas import RetrievalFilters
from rag_system.retrieval.schemas import RetrievalResult


CONFIG = {"host": "127.0.0.1", "port": 8080, "top_k_default": 5, "timeout_seconds": 1}
LLM_READY = {
    "status": "available",
    "roles": {
        "llm_reasoning": {"mode": "llama_cpp", "selected_model": "local-model.gguf", "available": True},
        "llm_structure": {"mode": "llama_cpp", "selected_model": "local-model.gguf", "available": True},
    },
}

CITATION = {
    "display": "ISO 5817:2014(E), Clause 6, Table 1, printed page 5",
    "clause": "6",
    "table": "1",
    "pdf_page_start": 9,
    "pdf_page_end": 9,
    "printed_page_start": "5",
    "printed_page_end": "5",
}


class FakeRetrievalService:
    def __init__(self):
        self.calls = []

    async def query(self, query, top_k=None, filters=None, standard_id=None):
        self.calls.append((query, top_k, filters))
        return RetrievalResponse(
            query=query.strip(),
            evidence=[EvidenceItem(
                chunk_id="ISO-5817-2014-EN__TABLE-1__CHUNK-001",
                document="ISO-5817-2014-EN",
                text="1.5 401 Lack of fusion",
                score=0.12,
                metadata={"content_type": "table_segment", "score_type": "cosine_distance"},
                citation=CITATION,
            )],
        )

    def close(self):
        return None


class FailingRetrievalService:
    async def query(self, query, top_k=None, filters=None, standard_id=None):
        raise RetrievalUnavailable("retrieval backend unavailable")


class UnexpectedFailureService:
    async def query(self, query, top_k=None, filters=None, standard_id=None):
        raise RuntimeError("sensitive backend detail")


class FakeHybridRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k=None, collections=("text", "tables")):
        self.calls.append((query, top_k, collections))
        return [RetrievalResult(
            chunk_id="chunk-1", content_type="definition", document_id="ISO-5817-2014-EN",
            clause_number="3.1", clause_title="quality level", table_id="",
            pdf_page_start=6, pdf_page_end=6, citation=CITATION,
            distance=0.2, source_hash="a" * 64, collection="weld_standard_text",
        )]

    def close(self):
        return None


class FakeTextStore:
    def get(self, chunk_id):
        return "stored evidence text"


class FakeLLMProvider:
    def __init__(self, value):
        self.value = value
        self.calls = []
        self.closed = False

    def generate_text(self, prompt, system_prompt=""):
        self.calls.append((prompt, system_prompt))
        return str(self.value)

    def generate_json(self, prompt, system_prompt=""):
        self.calls.append((prompt, system_prompt))
        return dict(self.value)

    def close(self):
        self.closed = True


class ApiTests(unittest.TestCase):
    def test_english_only_generation_budget_does_not_hide_configured_output_limit(self):
        self.assertEqual(effective_generation_tokens({"max_new_tokens": 256}, english_only=True), 256)

    def test_evidence_backed_answer_is_not_rejected_only_for_missing_terminal_punctuation(self):
        service = LLMEnhancedRetrievalService(FakeRetrievalService(), lambda: FakeLLMProvider("unused"), lambda: FakeLLMProvider({}), english_only=True)
        response = asyncio.run(FakeRetrievalService().query("lack of fusion"))
        answer = GroundedAnswer(answer="The reported value is 1.5 [S1]", confidence="medium", citations=[{"document": "ISO-5817-2014-EN", "clause": "6", "page": "5", "chunk_id": "ISO-5817-2014-EN__TABLE-1__CHUNK-001"}])
        validated = service._validate_answer(response, answer)
        self.assertEqual(validated.answer, answer.answer)

    def make_client(self, service):
        return TestClient(create_app(service=service, config=CONFIG, llm_readiness_check=lambda: LLM_READY), raise_server_exceptions=False)

    def test_health_endpoint(self):
        with self.make_client(FakeRetrievalService()) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "api_ready": True, "rag_ready": True})

    def test_chat_endpoint_retrieves_before_using_document_sources(self):
        with tempfile.TemporaryDirectory(prefix="drilldoc-chat-test-") as temporary:
            with patch.dict(os.environ, {"DRILLDOC_DATA_DIR": str(Path(temporary) / "data")}):
                app = create_app(service=FakeRetrievalService(), config=CONFIG, llm_readiness_check=lambda: LLM_READY)
                client = TestClient(app, raise_server_exceptions=False)
                with client:
                    response = client.post("/api/v1/chat", json={"query": "What is the recorded value?", "language": "en"})
                app.state.chat_repository = None
                del client
                del app
                gc.collect()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["sources"])
        self.assertEqual(response.json()["sources"][0]["chunk_id"], "ISO-5817-2014-EN__TABLE-1__CHUNK-001")

    def test_health_and_readiness_when_index_is_missing(self):
        with tempfile.TemporaryDirectory(prefix="drilldoc-api-no-index-") as temporary:
            with patch.dict(os.environ, {"DRILLDOC_DATA_DIR": str(Path(temporary) / "data")}):
                app = create_app(llm_readiness_check=lambda: LLM_READY)
                client = TestClient(app, raise_server_exceptions=False)
                with client:
                    health = client.get("/health")
                    ready = client.get("/ready")
                del client
                app.state.chat_repository = None
                del app
                gc.collect()
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json(), {"status": "ok", "api_ready": True, "rag_ready": False})
            self.assertEqual(ready.status_code, 503)
            self.assertEqual(ready.json(), {"detail": "retrieval backend unavailable"})

    def test_service_readiness_requires_retrieval_and_selected_llms(self):
        with self.make_client(FakeRetrievalService()) as client:
            response = client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        self.assertEqual(response.json()["embedding_model"], "BAAI/bge-m3")

        unavailable = {**LLM_READY, "status": "unavailable"}
        app = create_app(
            service=FakeRetrievalService(), config=CONFIG,
            llm_readiness_check=lambda: unavailable,
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/ready")
        self.assertEqual(response.status_code, 503)

    def test_missing_selected_model_is_reported_separately_from_service_health(self):
        readiness = {
            "status": "unavailable",
            "roles": {
                "llm_reasoning": {"mode": "llama_cpp", "selected_model": "missing.gguf", "available": False},
                "llm_structure": {"mode": "llama_cpp", "selected_model": "local-model.gguf", "available": True},
            },
        }
        app = create_app(
            service=FakeRetrievalService(), config=CONFIG,
            llm_readiness_check=lambda: readiness,
        )
        with TestClient(app, raise_server_exceptions=False) as client:
            health = client.get("/health")
            llm = client.get("/readiness")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok", "api_ready": True, "rag_ready": True})
        self.assertEqual(llm.status_code, 200)
        self.assertEqual(llm.json(), readiness)

    def test_valid_retrieval_request(self):
        service = FakeRetrievalService()
        with self.make_client(service) as client:
            response = client.post("/api/v1/retrieval/query", json={"query": "lack of fusion", "top_k": 3})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["query"], "lack of fusion")
        self.assertEqual(len(response.json()["evidence"]), 1)
        self.assertEqual(response.json()["contract_version"], "1.0")
        self.assertEqual(service.calls[0][1], 3)

    def test_invalid_request_schema(self):
        with self.make_client(FakeRetrievalService()) as client:
            response = client.post("/api/v1/retrieval/query", json={"query": "porosity", "top_k": 0})
        self.assertEqual(response.status_code, 422)

    def test_empty_query(self):
        with self.make_client(FakeRetrievalService()) as client:
            response = client.post("/api/v1/retrieval/query", json={"query": "   ", "top_k": 2})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "query must not be empty"})

    def test_response_schema_validation(self):
        with self.make_client(FakeRetrievalService()) as client:
            body = client.post("/api/v1/retrieval/query", json={"query": "porosity"}).json()
        parsed = RetrievalResponse.model_validate(body)
        self.assertEqual(parsed.evidence[0].document, "ISO-5817-2014-EN")
        self.assertEqual(parsed.evidence[0].metadata["score_type"], "cosine_distance")

    def test_citation_preservation(self):
        with self.make_client(FakeRetrievalService()) as client:
            body = client.post("/api/v1/retrieval/query", json={"query": "lack of fusion"}).json()
        self.assertEqual(body["evidence"][0]["citation"], CITATION)

    def test_retrieval_failure_handling(self):
        with self.make_client(FailingRetrievalService()) as client:
            response = client.post("/api/v1/retrieval/query", json={"query": "porosity"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "retrieval backend unavailable"})

    def test_unexpected_failure_is_sanitized(self):
        with self.make_client(UnexpectedFailureService()) as client:
            response = client.post("/api/v1/retrieval/query", json={"query": "porosity"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "unexpected internal error"})

    def test_service_calls_existing_retriever_and_maps_evidence(self):
        retriever = FakeHybridRetriever()
        service = RetrievalService(retriever, FakeTextStore(), default_top_k=5, timeout_seconds=1)
        result = asyncio.run(service.query("quality level", 2, RetrievalFilters(collections=["text"])))
        self.assertEqual(retriever.calls, [("quality level", 2, ("text",))])
        self.assertEqual(result.evidence[0].text, "stored evidence text")
        self.assertEqual(result.evidence[0].score, 0.2)
        self.assertEqual(result.evidence[0].citation, CITATION)

    def test_production_llm_wrapper_uses_reasoning_and_structure_and_preserves_citations(self):
        base = FakeRetrievalService()
        reasoning = FakeLLMProvider("Grounded draft [ISO-5817-2014-EN__TABLE-1__CHUNK-001]")
        structure = FakeLLMProvider({
            "answer": "Grounded answer.", "confidence": "high",
            "citation_chunk_ids": ["ISO-5817-2014-EN__TABLE-1__CHUNK-001", "invented"],
        })
        service = LLMEnhancedRetrievalService(base, lambda: reasoning, lambda: structure)
        result = asyncio.run(service.query("lack of fusion", 3))
        service.close()
        self.assertEqual(len(reasoning.calls), 1)
        self.assertEqual(len(structure.calls), 1)
        self.assertTrue(reasoning.closed)
        self.assertTrue(structure.closed)
        self.assertEqual(result.answer.answer, "Grounded answer.")
        self.assertEqual([item.chunk_id for item in result.answer.citations], ["ISO-5817-2014-EN__TABLE-1__CHUNK-001"])
        self.assertEqual(result.evidence[0].citation, CITATION)

    def test_production_llm_wrapper_fails_request_instead_of_returning_unstructured_output(self):
        service = LLMEnhancedRetrievalService(
            FakeRetrievalService(), lambda: FakeLLMProvider("draft"),
            lambda: FakeLLMProvider({"answer": "x", "confidence": "invalid", "citation_chunk_ids": []}),
        )
        with self.assertRaisesRegex(RetrievalUnavailable, "LLM-grounded answering unavailable"):
            asyncio.run(service.query("porosity"))

    def test_truncated_quality_level_table_answer_is_rejected_but_evidence_is_preserved(self):
        service = LLMEnhancedRetrievalService(
            FakeRetrievalService(), lambda: FakeLLMProvider("draft"),
            lambda: FakeLLMProvider({
                "answer": "For quality level B the limit is 2.5% and 5",
                "confidence": "high",
                "citation_chunk_ids": ["ISO-5817-2014-EN__TABLE-1__CHUNK-001"],
            }),
        )
        result = asyncio.run(service.query("ISO 5817 quality level B porosity"))
        self.assertEqual(result.answer.confidence, "insufficient")
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(
            result.answer.citations[0].chunk_id,
            "ISO-5817-2014-EN__TABLE-1__CHUNK-001",
        )

    def test_phased_residency_rebuilds_retrieval_and_releases_llm(self):
        bases = []
        providers = []

        def make_base():
            base = FakeRetrievalService()
            bases.append(base)
            return base

        def make_provider():
            provider = FakeLLMProvider("Grounded answer [S1].")
            providers.append(provider)
            return provider

        service = LLMEnhancedRetrievalService(
            make_base(), make_provider, make_provider,
            plain_text_only=True, retrieval_factory=make_base,
            release_each_request=True,
        )
        first = asyncio.run(service.query("water depth"))
        second = asyncio.run(service.query("total depth"))

        self.assertEqual(len(bases), 2)
        self.assertEqual(len(providers), 2)
        self.assertTrue(all(provider.closed for provider in providers))
        self.assertEqual(len(first.answer.citations), 1)
        self.assertEqual(len(second.answer.citations), 1)

    def test_forensic_english_only_mode_rejects_persian_before_retrieval(self):
        base = FakeRetrievalService()
        service = LLMEnhancedRetrievalService(
            base, lambda: FakeLLMProvider("draft"), lambda: FakeLLMProvider({}),
            english_only=True,
        )
        with self.assertRaisesRegex(InvalidQuery, "English-only"):
            asyncio.run(service.query("عمق چاه چقدر است؟"))
        self.assertEqual(base.calls, [])

    def test_trace_permission_error_does_not_fail_main_answer(self):
        service = LLMEnhancedRetrievalService(
            FakeRetrievalService(), lambda: FakeLLMProvider("Grounded draft [S1]."),
            lambda: FakeLLMProvider({}), english_only=True, trace_enabled=True,
        )
        with patch("pathlib.Path.open", side_effect=PermissionError("readonly trace")):
            result = asyncio.run(service.query("lack of fusion"))
        self.assertEqual(result.answer.confidence, "medium")
        self.assertIn("Grounded draft", result.answer.answer)
    def test_forensic_trace_records_zero_qwen_and_exact_context(self):
        reasoning = FakeLLMProvider("Grounded draft.")
        formatting = FakeLLMProvider({
            "answer": "Grounded answer.", "confidence": "high",
            "citation_chunk_ids": ["ISO-5817-2014-EN__TABLE-1__CHUNK-001"],
        })
        service = LLMEnhancedRetrievalService(
            FakeRetrievalService(), lambda: reasoning, lambda: formatting,
            english_only=True, trace_enabled=True,
        )
        with patch("pathlib.Path.open", side_effect=PermissionError("trace output unavailable")):
            result = asyncio.run(service.query("lack of fusion"))
        trace = service.last_trace
        self.assertEqual(trace["qwen_call_count"], 0)
        self.assertTrue(trace["llama_used"])
        self.assertFalse(trace["fallback_used"])
        self.assertIn("1.5 401 Lack of fusion", trace["selected_context"])
        self.assertEqual(trace["raw_llama_output"], "Grounded draft.")
        self.assertEqual(len(reasoning.calls), 1)
        self.assertEqual(len(formatting.calls), 0)
        self.assertEqual(result.answer.citations[0].chunk_id, "ISO-5817-2014-EN__TABLE-1__CHUNK-001")
        self.assertEqual(trace["final_output"]["citations"][0]["chunk_id"], "ISO-5817-2014-EN__TABLE-1__CHUNK-001")
        self.assertTrue(service._trace_output_disabled)
        self.assertTrue(service.trace_enabled)

    def test_english_only_mode_uses_one_llama_pass_then_backend_validation(self):
        reasoning = FakeLLMProvider("The value is 1.5 [S1].")
        structure = FakeLLMProvider({"answer": "must not be called"})
        service = LLMEnhancedRetrievalService(
            FakeRetrievalService(), lambda: reasoning, lambda: structure,
            english_only=True, trace_enabled=True,
        )
        result = asyncio.run(service.query("lack of fusion"))
        self.assertEqual(result.answer.answer, "The value is 1.5 [S1].")
        self.assertEqual(len(reasoning.calls), 1)
        self.assertEqual(structure.calls, [])
        self.assertEqual(service.last_trace["qwen_call_count"], 0)


if __name__ == "__main__":
    unittest.main()


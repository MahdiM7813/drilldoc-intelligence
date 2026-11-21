"""Run offline synthetic retrieval and (unless --retrieval-only) real local Llama evaluation."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "backend"))

from rag_system.api.service import ChunkTextStore, RetrievalService, build_llm_enhanced_service
from rag_system.answering.run import build_provider, load_config as load_answer_config
from rag_system.evaluation.dataset import load_jsonl
from rag_system.evaluation.metrics import abstention_marker, answer_match_metrics, citation_metrics, retrieval_metrics
from rag_system.hybrid_retrieval.hybrid_retriever import HybridRetriever
from rag_system.hybrid_retrieval.query_analyzer import QueryAnalyzer


def _external(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == PROJECT or PROJECT in resolved.parents:
        raise ValueError(f"Evaluation artifacts must be outside the project: {resolved}")
    return resolved


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if p == 0.5:
        return round(statistics.median(ordered), 4)
    return round(ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))], 4)


def _aliases(records: list[dict[str, Any]]) -> dict[str, str]:
    result = {}
    for row in records:
        if row.get("content_type") == "clause":
            match = re.search(r"\d+(?:\.\d+)*", str(row.get("clause_number") or ""))
            if match:
                result[f"C{int(match.group()):02d}"] = row["chunk_id"]
        elif row.get("content_type") == "table_segment":
            segment = str(row["chunk_id"]).rsplit("-", 1)[-1]
            for table_num, table_segment in ((1, "001"), (2, "002"), (3, "003")):
                if segment == table_segment:
                    for row_num in range(1, 4):
                        result[f"T{table_num}:R{row_num}"] = row["chunk_id"]
    return result


def _macro(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    return {key: round(statistics.mean(row[key] for row in items), 4) for key in items[0]}


def run(args: argparse.Namespace) -> dict[str, Any]:
    chunks_path, manifest_path, chroma_root = map(_external, (args.chunks, args.manifest, args.chroma_root))
    output_dir = _external(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = _external(output_dir / "runtime-data")
    os.environ.update({"DRILLDOC_DATA_DIR": str(data_root), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONDONTWRITEBYTECODE": "1"})

    records = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = load_jsonl(args.dataset.resolve())
    aliases = _aliases(records)
    for question in questions:
        for evidence_id in question["relevant_evidence_ids"]:
            if evidence_id not in aliases:
                raise ValueError(f"Ground-truth evidence ID cannot be resolved: {question['question_id']} {evidence_id}")
        for document_id in question["relevant_document_ids"]:
            if document_id not in {str(x.get("document_id")) for x in records}:
                raise ValueError(f"Ground-truth document ID not present in chunks: {document_id}")

    retrieval_config = {
        "embedding": {"model_name": args.embedding_model, "device": args.embedding_device, "batch_size": 2,
                      "normalize_embeddings": True, "trust_remote_code": False, "local_files_only": True},
        "vector_store": {"persist_directory": str(chroma_root), "text_collection": args.text_collection,
                         "table_collection": args.table_collection},
        "retrieval": {"top_k": 5, "distance_metric": "cosine", "read_only_copy": False},
    }
    hybrid_config = {
        "rag04": retrieval_config, "manifest_path": str(manifest_path),
        "bm25": {"chunk_paths": [str(chunks_path)], "k1": 1.5, "b": 0.75},
        "hybrid": {"top_k": 5, "candidate_pool": 15},
        "reranking": {"rrf_k": 60, "vector_weight": 1.0, "keyword_weight": 1.2,
                      "phrase_boost": 0.05, "definition_boost": 0.035, "title_boost": 0.06,
                      "table_defect_boost": 0.02},
    }
    init_started = time.perf_counter()
    hybrid = HybridRetriever(hybrid_config, manifest_path)
    retrieval_model_load = time.perf_counter() - init_started
    analyzer = QueryAnalyzer()
    rerank_times: list[float] = []
    original_rerank = hybrid.reranker.rerank

    def timed_rerank(analysis: Any, candidates: list[Any]) -> list[Any]:
        started = time.perf_counter()
        try:
            return original_rerank(analysis, candidates)
        finally:
            rerank_times.append(time.perf_counter() - started)
    hybrid.reranker.rerank = timed_rerank

    retrieval_cases, retrieval_latencies, vector_latencies, bm25_latencies = [], [], [], []
    for question in questions:
        query = question["question"]
        analysis = analyzer.analyze(query)
        started = time.perf_counter(); vector = hybrid.vector_retriever.retrieve(analysis.expanded, 15); vector_latencies.append(time.perf_counter() - started)
        started = time.perf_counter(); keyword = hybrid.bm25.search(analysis.tokens, 15); bm25_latencies.append(time.perf_counter() - started)
        scores: dict[str, float] = defaultdict(float)
        for rank, row in enumerate(vector, 1): scores[row.chunk_id] += 1.0 / (60 + rank)
        for rank, (row, _, _) in enumerate(keyword, 1): scores[row.chunk_id] += 1.2 / (60 + rank)
        rrf = [row_id for row_id, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]
        started = time.perf_counter(); ranked = hybrid.retrieve(query, 5); retrieval_latencies.append(time.perf_counter() - started)
        gold = sorted({aliases[x] for x in question["relevant_evidence_ids"]})
        methods = {"bm25": [row.chunk_id for row, _, _ in keyword],
                   "semantic": [row.chunk_id for row in vector], "hybrid_rrf": rrf,
                   "hybrid_reranked": [row.chunk_id for row in ranked]}
        case = {"question_id": question["question_id"], "category": question["category"],
                "answering_allowed": question["answering_allowed"], "gold_chunk_ids": gold,
                "ranked_chunk_ids": methods,
                "metrics": {name: retrieval_metrics(ids, gold) for name, ids in methods.items()} if gold else None}
        retrieval_cases.append(case)

    answer_cases = []
    model_load_seconds = None
    if not args.retrieval_only:
        config_path = output_dir / "answering_runtime.yaml"
        import yaml
        model_path = args.llama_model.expanduser().resolve()
        config = {"retrieval": {"config_path": "unused", "manifest_path": str(manifest_path), "top_k": 5},
                  "providers": {"reasoning": {"type": "llama_cpp", "model_path": str(model_path),
                    "n_ctx": args.context, "n_batch": 128, "n_threads": args.threads, "n_gpu_layers": args.gpu_layers,
                    "max_new_tokens": args.max_new_tokens, "temperature": 0.0},
                    "formatter": {"type": "llama_cpp", "model_path": str(model_path), "n_ctx": args.context,
                    "n_batch": 128, "n_threads": args.threads, "n_gpu_layers": args.gpu_layers,
                    "max_new_tokens": args.max_new_tokens, "temperature": 0.0}},
                  "answering": {"generation_timeout_seconds": 300, "require_citations": True}}
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        os.environ["DRILLDOC_ANSWERING_CONFIG"] = str(config_path)
        provider_config = load_answer_config(config_path)["providers"]["reasoning"]
        started = time.perf_counter(); provider = build_provider(provider_config); model_load_seconds = time.perf_counter() - started
        base = RetrievalService(hybrid, ChunkTextStore([chunks_path]), default_top_k=5, timeout_seconds=180)
        service = build_llm_enhanced_service(base, english_only=True, trace_enabled=True)
        service.reasoning_factory = lambda: provider
        service.structure_factory = lambda: provider

        async def answer_all() -> None:
            for question in questions:
                started_at = time.perf_counter()
                response = await service.query(question["question"], top_k=5)
                total = time.perf_counter() - started_at
                final = response.answer.model_dump()
                trace = dict(service.last_trace)
                evidence = [{"chunk_id": item.chunk_id, "text": item.text, "citation": item.citation,
                             "metadata": item.metadata} for item in response.evidence]
                gold = sorted({aliases[x] for x in question["relevant_evidence_ids"]})
                cited = [row.get("chunk_id", "") for row in final.get("citations", [])]
                raw_answer = trace.get("raw_llama_output", "")
                observed_abstention = abstention_marker(final.get("answer", "")) or final.get("confidence") == "insufficient"
                answer_cases.append({"question_id": question["question_id"], "category": question["category"],
                    "question": question["question"], "expected_answer": question["expected_answer"],
                    "answering_allowed": question["answering_allowed"], "expected_abstention": question["expected_abstention"],
                    "observed_abstention": observed_abstention,
                    "abstention_correct": (observed_abstention == question["expected_abstention"])
                        if not question["answering_allowed"] else None,
                    "raw_model_answer": raw_answer,
                    "raw_answer_metrics": answer_match_metrics(raw_answer, question["expected_answer"]),
                    "final_answer_metrics": answer_match_metrics(final.get("answer", ""), question["expected_answer"]),
                    "final_answer": final, "retrieved_evidence": evidence,
                    "selected_context_chunk_ids": trace.get("selected_record_ids", []),
                    "gold_chunk_ids": gold, "citation_metrics": citation_metrics(cited, gold, [x["chunk_id"] for x in evidence]),
                    "latency_seconds": total, "generation": dict(provider.last_generation), "trace": trace})
        asyncio.run(answer_all())
        service.close()

    retrievable = [row for row in retrieval_cases if row["metrics"] is not None]
    method_summary = {method: _macro([row["metrics"][method] for row in retrievable])
                      for method in ("bm25", "semantic", "hybrid_rrf", "hybrid_reranked")}
    summary: dict[str, Any] = {
        "python": sys.version.split()[0], "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device, "retrieval_model_init_seconds": round(retrieval_model_load, 4),
        "document_count": len({x.get("document_id") for x in records}), "chunk_count": len(records),
        "question_count": len(questions), "answerable_retrieval_question_count": len(retrievable),
        "retrieval_macro_metrics": method_summary,
        "latency_seconds": {"semantic_median": _percentile(vector_latencies, .5), "semantic_p95": _percentile(vector_latencies, .95),
            "bm25_median": _percentile(bm25_latencies, .5), "bm25_p95": _percentile(bm25_latencies, .95),
            "hybrid_median": _percentile(retrieval_latencies, .5), "hybrid_p95": _percentile(retrieval_latencies, .95),
            "rerank_median": _percentile(rerank_times, .5), "rerank_p95": _percentile(rerank_times, .95)},
        "retrieval_only": args.retrieval_only,
    }
    if answer_cases:
        answer_latencies = [x["latency_seconds"] for x in answer_cases]
        answerable_cases = [x for x in answer_cases if x["answering_allowed"]]
        numeric_cases = [x for x in answerable_cases if x["raw_answer_metrics"]["numeric_value_match"] is not None]
        unit_cases = [x for x in answerable_cases if x["raw_answer_metrics"]["unit_match"] is not None]
        unanswerable_cases = [x for x in answer_cases if not x["answering_allowed"]]
        summary.update({"llm_model": str(args.llama_model.resolve()), "context_tokens": args.context,
            "gpu_layers": args.gpu_layers, "llm_model_load_seconds": round(model_load_seconds or 0, 4),
            "answer_latency_median_seconds": _percentile(answer_latencies, .5),
            "answer_latency_p95_seconds": _percentile(answer_latencies, .95),
            "raw_normalized_exact_match": round(sum(x["raw_answer_metrics"]["normalized_exact_match"] for x in answerable_cases) / len(answerable_cases), 4),
            "raw_numeric_value_accuracy": round(sum(x["raw_answer_metrics"]["numeric_value_match"] is True for x in numeric_cases) / len(numeric_cases), 4) if numeric_cases else None,
            "raw_unit_accuracy": round(sum(x["raw_answer_metrics"]["unit_match"] is True for x in unit_cases) / len(unit_cases), 4) if unit_cases else None,
            "citation_precision_macro": round(statistics.mean(x["citation_metrics"]["citation_precision"] for x in answerable_cases), 4),
            "citation_coverage_macro": round(statistics.mean(x["citation_metrics"]["citation_coverage"] for x in answerable_cases), 4),
            "unanswerable_verbal_abstention_count": sum(x["observed_abstention"] for x in unanswerable_cases),
            "unanswerable_confidence_insufficient_count": sum(x["final_answer"].get("confidence") == "insufficient" for x in unanswerable_cases),
            "manual_faithfulness_review_required": True})
    result = {"summary": summary, "retrieval_per_question": retrieval_cases, "answer_per_question": answer_cases}
    output = output_dir / "benchmark_results.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    hybrid.close()
    return {"output": str(output), "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--chroma-root", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "tests/fixtures/synthetic_rag_eval.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text-collection", default="eval_text")
    parser.add_argument("--table-collection", default="eval_tables")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--llama-model", type=Path)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    if not args.retrieval_only and args.llama_model is None:
        parser.error("--llama-model is required unless --retrieval-only is set")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

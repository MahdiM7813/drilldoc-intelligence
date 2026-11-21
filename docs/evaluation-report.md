# Phase 3.4 — RAG Evaluation & Benchmarking Report

**Status: measured synthetic benchmark completed with significant limitations.** Retrieval and real local Llama execution ran. Results are diagnostic for one small synthetic procedure, not evidence of general RAG quality or production readiness.

> This report's 40/40 test count records the Phase 3.4 run. Finalization later ran the current suite at 44/44; this does not change any benchmark measurements below.

## 1. Existing capabilities inspected

Before adding evaluation code, the project already had PDF extraction and structure detection, chunking and chunk validation, local BGE-M3 embedding, Chroma indexing, BM25, semantic retrieval, hybrid RRF, a deterministic reranker, context selection/packing, the llama.cpp provider, API answer validation, and citation mapping. Existing tests covered API behavior and citation/runtime paths, but there was no labeled evaluation dataset, ranking metric package, answer scoring, or comparable ablation runner. The new code reuses these components; it does not implement another retriever or indexer.

## 2. Corpus and extraction/chunking results

The benchmark used only the checked-in fictional engineering fixture and a generated PDF; no drilling material, previous index, chat data, or external document was used. The PDF was 13 pages. The extractor found all 10 numbered clauses and detected one logical table entity spanning three visually separate table pages. Its output preserved all three table page segments and their synthetic values, but incorrectly grouped the three physical tables under one table ID. Chunking produced 14 chunks: 10 clauses, one table parent, and three table segments. Chroma indexed 13 records because table parents are excluded by configuration.

A rendered table page was visually checked and readable. A reproducibility rerun produced an identical chunk JSONL hash. The extraction and chunking numbers below are single elapsed samples, not repeated latency statistics.

| Stage | Measured result |
|---|---:|
| PDF extraction | 0.249 s; 13 pages; 10 clauses; 1 logical table entity |
| Chunking | 0.009 s; 14 chunks; 3 table segments |
| BGE-M3 batch encoding | 1.672 s for 13 indexed text/segment records; 1,024 dimensions |
| Indexing job | 10.580 s; 13 Chroma records; CPU |

## 3. Retrieval comparison

Thirty labeled questions were included. Ranking metrics are macro-averaged over the 25 answerable questions; the five unanswerable questions are assessed separately. All systems use the same query set, evidence labels, candidate pool, and top-five scoring depth. Relevance is binary at chunk level; table-row labels resolving to the same segment count as one relevant chunk.

| Method | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.54 | 0.96 | 0.96 | 0.248 | 0.80 | 0.8437 |
| BGE-M3 semantic | 0.74 | 1.00 | 1.00 | 0.256 | 0.94 | 0.9493 |
| Hybrid RRF, before reranker signals | 0.60 | 1.00 | 1.00 | 0.256 | 0.86 | 0.8959 |
| Hybrid with existing reranker | 0.60 | 1.00 | 1.00 | 0.256 | 0.86 | 0.8959 |

Semantic retrieval had the strongest MRR and nDCG on this corpus. Hybrid improved recall over BM25, but did not outperform semantic retrieval. Reranking changed no top-five ordering on 0 of 25 answerable questions and produced no metric improvement. No parameter tuning was done.

For the three specific low/nominal/high flow-stage questions, the relevant table segment ranked first for BM25, semantic, hybrid, and reranked hybrid. The corrected S04 hose-rating exceedance query also ranked its relevant clause/table evidence first across all four methods. The maximum-return-temperature table question ranked its relevant segment first for semantic retrieval and second for BM25/hybrid. This small fixture does not establish that the earlier flow-rate ranking weakness is generally resolved.

## 4. Answer generation, citations, and abstention

The actual local `llama.cpp` provider loaded `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`; Qwen was not used. The project service ran all 30 questions, including five unanswerable questions. Nine answerable cases that reached the initial 64-token limit were rerun with a 192-token limit. S04 was independently clarified from ambiguous “margin” wording to “exceedance,” then rerun with Llama and included in a fresh retrieval run. The final per-question results therefore contain 21 cases at 64 tokens and 9 at 192 tokens; the total was 40 real generation calls including initial attempts and reruns.

| Answer check | Result |
|---|---:|
| Raw Llama literal numeric values present | 23/23 numeric answer cases |
| Raw Llama literal units present | 23/23 unit-bearing answer cases |
| Strict normalized exact matches | 0/25 |
| Answerable final responses returned as `insufficient` | 5/25 |
| Unanswerable responses with explicit verbal abstention | 5/5 |
| Unanswerable responses with `insufficient` confidence | 0/5 |
| Macro citation precision | 0.752 |
| Macro citation coverage | 0.980 |
| Citations mapped to a retrieved chunk | 100% |
| Gold chunks retained after context selection/packing | 100% |

The 23/23 numeric/unit result is a literal-presence check, not a semantic or arithmetic correctness score. Exact match is too strict for generated full sentences in this set. Manual answer-faithfulness and unsupported natural-language claim review remains **not performed**; no human score is claimed.

The application converted five answerable raw Llama responses to “Insufficient evidence.” The raw answers contained the expected calculated values (including 0.6 MPa, 90 minutes, and 78 L), but those derived numbers do not appear literally in the source chunks cited as evidence. The current backend numeric grounding validator therefore rejects them. This is an observed limitation in the present answer/validation path, not a successful final answer. The raw S04 answer correctly described the setpoint as exceeding the hose rating by 0.6 MPa; the application still returned `insufficient` because 0.6 is derived rather than quoted.

For all five intentionally unanswerable questions, the model said the fact was missing or insufficient. However, the final confidence label was `medium` for all five; the service did not mark any as `insufficient`. Verbal abstention worked in this set, but confidence calibration did not.

Citation IDs were always constrained to retrieved items, but citation precision was reduced by irrelevant extra citations. Citation validity therefore should not be confused with claim-level citation support. Semantic faithfulness and whether each citation supports each sentence need manual review.

## 5. Ablation findings

1. **BM25 only:** lower recall/MRR than semantic retrieval on these questions.
2. **Semantic only:** best measured MRR (0.94) and nDCG@5 (0.9493).
3. **Hybrid RRF:** recall@5 was 1.00, but MRR/nDCG were below semantic-only.
4. **Hybrid plus reranker:** identical top-five rankings for all 25 answerable questions; no measured gain.
5. **Hybrid, reranker, and evidence selection:** gold-evidence coverage remained 1.00 after selection/packing. No “selection disabled” answer-generation run was performed, so answer-quality improvement from selection is **not measured**.

## 6. Runtime and environment

- Conda: the existing local Conda environment; Python 3.11.16.
- Embedding: `BAAI/bge-m3`, 1,024 dimensions, CPU. Retriever initialization was 1.844 s in the final retrieval-only run; a prior warm/cold-cache sample was 3.697 s.
- LLM: local GGUF Q4_K_M, context 2,048, 8 CPU threads, `n_gpu_layers=0`. Provider construction took 0.78 s; cold filesystem/cache state was not measured.
- PyTorch reported CUDA available and an NVIDIA RTX 3050 Laptop GPU, but this benchmark did not use the GPU or measure GPU utilization.
- Per-query retrieval latency (median / nearest-rank P95): semantic 0.0776 / 0.0818 s; BM25 0.0001 / 0.0001 s; hybrid 0.0780 / 0.0874 s; reranking 0.0002 / 0.0004 s.
- Answer latency includes retrieval and generation. With 64 output tokens (n=21), median was 36.89 s and P95 61.59 s. With 192 tokens (n=9), median was 89.27 s and P95 131.87 s. The mixed-budget results are not an apples-to-apples latency comparison.
- RAM/VRAM telemetry was not collected. No usage amount is inferred.
- Relevant installed versions: ChromaDB 1.5.9, sentence-transformers 6.0.1, llama-cpp-python 0.3.35, PyTorch 2.10.0, Transformers 5.16.1, pypdf 6.19.0, pdfplumber 0.11.10, FastAPI 0.141.1, httpx 0.28.1. No packages or models were installed or changed. A prior retriever initialization sample was 3.697 s; cache state was not controlled.

At the Phase 3.4 run, the full existing and new unit-test suite passed: **40/40**. Investigation of the prior 39/40 run found that a trace-file write failure disabled tracing itself before raw Llama output could be retained in memory. The service now disables only trace-file persistence while preserving in-memory trace capture. The regression test verifies actual raw output, zero Qwen calls, exact selected context, and citation chunk provenance under a simulated trace-file permission failure. The focused regression and full suite passed. The test process emitted a Starlette/httpx deprecation warning. No Git repository was initialized.

## 7. Offline constraint note

One early retrieval-only invocation emitted a Hugging Face warning that unauthenticated Hub requests were being sent, despite the model being present in the local cache. No model download or successful external response was observed, but the warning indicates a possible outbound request in that invocation; zero external contact cannot be certified for it. Final preparation and retrieval reruns set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before process startup and completed without that warning. This is recorded as a constraint incident.

## 8. Artifacts and changed files

Small, synthetic, reusable fixtures are retained in the project. Detailed answer traces, retrieval ranks, extraction/chunking reports, model manifest, generated PDF, and run metadata are stored in a dedicated external, synthetic-only evaluation-artifact directory, outside the project repository.

Files added or changed for this phase:

- `backend/rag_system/api/service.py` (preserve in-memory trace capture when trace-file output fails)
- `tests/test_api.py` (trace persistence failure and provenance regression coverage)
- `backend/rag_system/evaluation/__init__.py`
- `backend/rag_system/evaluation/dataset.py`
- `backend/rag_system/evaluation/metrics.py`
- `scripts/evaluation/build_synthetic_pdf.py`
- `scripts/evaluation/prepare_synthetic_corpus.py`
- `scripts/evaluation/run_benchmark.py`
- `tests/fixtures/synthetic_engineering_corpus.json`
- `tests/fixtures/synthetic_rag_eval.jsonl`
- `tests/test_evaluation_metrics.py`
- `docs/phase-3.4-methodology.md`
- `docs/phase-3.4-evaluation-report.md`

## 9. Limitations and remaining publication blockers

- The evaluation corpus is one small synthetic document; cross-document retrieval/generalization is untested.
- The extractor merges three physical tables into one logical table entity. Table segmentation preserved values, but the metadata grouping defect should be corrected before presenting broad table extraction quality.
- The reranker showed no ranking effect in this benchmark; hybrid retrieval underperformed semantic-only MRR/nDCG.
- The answer validator rejects legitimate derived numeric answers when the calculated number is not written verbatim in evidence. This blocks five answerable final responses.
- Unanswerable text was cautious, but confidence remained `medium` in all five cases.
- Citation precision is 0.752; citation entailment and overall faithfulness require human review.
- Answer inference was scored with mixed token budgets because only truncated cases were rerun at 192 tokens. A clean comparable answer benchmark should rerun all 30 with one fixed budget.
- Memory telemetry and manual faithfulness ratings are absent.
- The earlier Hugging Face warning leaves one outbound-attempt uncertainty; repeat only with offline flags set before process startup.
- A fresh independent environment installation/reproducibility audit and publication review remain separate tasks.

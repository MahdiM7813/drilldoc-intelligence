# Phase 3.3.3 — Local Llama 3.1 Answer Generation Validation

> Historical phase report: its results describe that phase and may predate the current implementation. See the [latest Phase 3.4 evaluation report](evaluation-report.md) for benchmark results and current evaluation limitations.


**Result:** Full synthetic RAG answer generation and citation validation succeeded using actual local Llama inference. This validates the tested local configuration; it does not establish fresh-install reproducibility or production readiness.

## Model and provider

- Model directory: a locally configured directory containing the Llama 3.1 8B Instruct Q4_K_M GGUF model.
- Format: one GGUF file, `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` (4,920,739,232 bytes; approximately 4.58 GiB; Q4_K_M quantization).
- No Safetensors shards or separate tokenizer/config files were present in the inspected directory; GGUF model metadata is used.
- Provider: existing `llama.cpp` integration via `llama-cpp-python` 0.3.35. The provider accepts the GGUF path directly; no conversion was needed.
- Runtime: the existing local Conda environment, Python 3.11.16. No packages or environment settings were installed or changed.
- A temporary answering config selected `llama_cpp` for both reasoning and formatting roles. It set model path to the GGUF, context to 2,048, batch to 128, CPU threads to 8, GPU layers to 0, temperature to 0, and per-call output limit to 192 tokens. Default project configuration was not changed.

## Synthetic pipeline and actual inference

A fictional, two-page engineering hydrostatic test PDF was created solely for this test. It contained multiple technical sections, distinct facts, and a small numerical table. Inputs, extracted text, chunks, model index and temporary configuration were stored outside the repository.

- PDF extraction, section/table structure detection, normalization, chunking, and metadata generation: completed (2 pages, 2 clause chunks, table parent and table segment; 4 chunks total).
- BGE-M3 embedding and Chroma indexing: completed using the existing local model cache in offline mode. Embeddings were 1,024-dimensional; 3 unique records were indexed across the text/table collections.
- Semantic retrieval, BM25/hybrid retrieval, reranking, and evidence selection: completed. The answer query was: “What was the maximum test pressure in the synthetic procedure?” Retrieved context included the synthetic Table 1 chunk and relevant clauses.
- Llama answer generation: actual local model inference completed for reasoning and formatting; no mock response was used. The answer was `2.4 MPa`, with confidence `high`.
- Citation: the answer cited `synthetic-weld-procedure-01__TABLE-1__CHUNK-001`. The citation ID was present in the retrieved evidence, and the cited table chunk contained the maximum limit value `2.4 MPa`. Application citation validation and the explicit evidence/value checks passed.

## Runtime observations

- Model load: 2.91 seconds.
- Two generation calls: 61.28 seconds (570 prompt / 34 completion tokens; about 0.55 tokens/sec) and 54.08 seconds (294 prompt / 45 completion tokens; about 0.83 tokens/sec).
- Total answer operation: 115.45 seconds.
- GPU capability was detected by the environment, but the test explicitly set `n_gpu_layers=0`; this run was CPU-only. No GPU inference or GPU utilization was measured.
- The Windows process-memory instrumentation returned zero for working set/private bytes. These readings are unavailable/unreliable telemetry, not evidence of zero memory use. Memory usage therefore remains unmeasured.
- No model loading or generation errors occurred.

## Regression tests and code changes

The full unittest suite ran in the existing environment with the backend on `PYTHONPATH`: **33 tests passed**. This includes API health/readiness, citation validation, runtime path checks, table chunking, and the new manifest-path regression test. A first discovery invocation without the backend import path failed to import the suite; rerunning with the configured backend path passed all 33 tests.

A genuine path-type defect encountered in the production retrieval path was corrected: manifest paths are now passed as `Path` instances in the API service builder and hybrid CLI runner, matching the retriever's `.resolve()` use. A focused regression test was added for the API builder. No default model configuration or original workspace files were modified.

Files changed in this phase:

- `backend/rag_system/api/service.py`
- `backend/rag_system/hybrid_retrieval/run.py`
- `tests/test_hybrid_manifest_path.py` (new)
- `docs/phase-3.3.3-llama-validation-report.md` (this report)

## Limitations and conclusion

The tested answer-generation path, including actual Llama inference and citation provenance validation against retrieved evidence, is validated for this synthetic case. Resource measurements are partial because process memory telemetry was unavailable and GPU offload was disabled. The run does not prove broad answer quality, fresh independent installation, or production suitability. No external service, original drilling document, original data artifact, or original workspace was used. Temporary synthetic data and index were removed after validation; the user model and existing model cache were retained.


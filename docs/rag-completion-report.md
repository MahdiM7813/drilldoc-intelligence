# Phase 3.3.2 — Local Model Discovery and RAG Completion Report

> Historical phase report: its results describe that phase and may predate the current implementation. See the [latest Phase 3.4 evaluation report](evaluation-report.md) for benchmark results and current evaluation limitations.


**Project:** DrillDoc Intelligence  
**Environment:** the existing local Conda environment (Python 3.11.16)
**Result:** Synthetic retrieval pipeline completed through evidence selection and source-citation provenance. LLM answer generation remains blocked because no usable configured LLM model is present. This is not a fresh-install reproducibility result.

## Model discovery

Inspection was limited to standard Hugging Face/SentenceTransformers/Torch caches, the Conda environment’s conventional model directories, the configured DrillDoc model root, and common local Ollama/model directories. No tokens or credential files were read or printed.

| Model/resource | Finding | Use |
| --- | --- | --- |
| `BAAI/bge-m3` | Complete Hugging Face snapshot `5617a9f61b028005a4858fdac845db406aefb181`; 30 files, 4,587,317,404 bytes, including `modules.json`, tokenizer files, config, and `pytorch_model.bin` | Loaded offline and used by project indexing and retrieval on CPU |
| `sentence-transformers/all-MiniLM-L6-v2` | Complete cached snapshot, about 91.6 MB | Compatible alternative found; not selected because configured BGE-M3 was available |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Snapshot exists but contains only 7 files/about 11.5 MB and no model weight file | Incomplete; not usable as the configured formatter because no model weights are present |
| Configured llama.cpp reasoning model | Config expects a local `reasoning.gguf` under the configured model root; that root/file was absent | Unavailable |
| Ollama models | Local Ollama model directory exists, but no model manifests were found | No registered local Ollama model found; no Ollama service was contacted |

The exact BGE-M3 cache was used with `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, telemetry disabled, CPU execution, and no cache override to another user’s or project’s model directory. No download was performed.

## ONNX Runtime and ChromaDB

- ChromaDB distribution: **1.5.9**.
- `onnxruntime` distribution metadata: **not installed**.
- `onnxruntime-gpu` distribution: **1.30.0**.
- `import onnxruntime` succeeds and reports **1.30.0**, with `CUDAExecutionProvider` and `CPUExecutionProvider` available.
- A `PersistentClient` and both project collections initialized successfully in an external temporary directory.
- `pip check` still reports that ChromaDB requires the `onnxruntime` distribution. This is a package metadata inconsistency in this Conda environment, but it was **not an execution blocker** for this project’s explicit-vector path: the real indexer supplied BGE-M3 vectors to Chroma, persisted them, reopened the index, and passed its vector-dimension/document readback verification. Chroma’s default embedding function was not used.

## Embedding and Chroma indexing

The project’s actual `LocalBGEEmbedder` and indexing command loaded BGE-M3 locally. It produced **1,024-dimensional** embeddings on CPU. Indexing completed successfully from the synthetic chunks:

- 6 records passed through preparation; 3 unique indexable records were written after deduplication.
- Chroma collections: `drilldoc_text` with 2 records and `drilldoc_tables` with 1 record.
- Added 3, updated 0, failed 0; no truncations.
- Persistence restart verification passed; documents and 1,024-dimension vectors read back from the persisted database.

The PDF, generated chunks, manifest, reports, and Chroma database were all under a fresh external temporary directory. No repository-local data/index was created.

## Semantic, hybrid, and reranking results

Real BGE query embeddings and Chroma vector queries ran against only the synthetic report. Three independent questions were checked:

| Query | Semantic top result | Hybrid/reranked top result | Observation |
| --- | --- | --- | --- |
| Maximum inlet pressure | Independent-observations section; table ranked third | Table segment | Hybrid ranking promoted the exact limit table |
| Observed flow rate | Independent-observations section | Measurement-procedure section | Hybrid ranking was less relevant than semantic/BM25 evidence for this query |
| Temperature sampling interval | Measurement-procedure section | Measurement-procedure section | Expected section ranked first |

BM25 independently retrieved the synthetic pressure, flow, and sampling facts. The configured deterministic hybrid reranker executed and emitted vector rank, keyword rank, matched terms, and final scores. Ranking is functional but not uniformly accurate on this very small corpus; the flow-rate hybrid result is a concrete quality limitation. No external reranker model was configured or run.

## Evidence selection and citations

The real `RetrievalService` adapter loaded text for retrieved synthetic chunk IDs, returned evidence with metadata, and the project’s `select_context` and `pack_context` functions produced a packed context. Three retrieved chunks were selected for the maximum-pressure query.

For all three retrieved results, the citation page matched stored page metadata; clause citations matched clause numbers and the table citation retained table `1`. This validates retrieved-source citation provenance. **It does not validate a generated answer citation**, because answer generation was unavailable. The standalone citation validation tests passed as part of the test suite.

## Answer generation

Answer generation stopped at local provider availability. The configured llama.cpp GGUF file was absent; the cached Qwen snapshot was incomplete and did not match a usable configured formatter; no local Ollama model manifest was present. No provider was invoked, no answer was fabricated, and no citation was generated from an answer. Answer generation and end-to-end answer citation validation remain unverified.

## Regression tests

All **32 tests passed** in the unchanged Conda environment after the table-row-label correction:

- API: 22 passed.
- Citation validation: 4 passed.
- Runtime configuration: 4 passed.
- Table chunker: 2 passed.

FastAPI/Starlette emitted an `httpx` TestClient deprecation warning. One test intentionally logs a rejected invalid LLM confidence payload; it passed by confirming rejection. No package was changed.

## Remaining blockers and reproducibility

1. No usable local LLM is available, so generated answers and generated answer citations remain blocked.
2. Hybrid retrieval ranked the flow-rate query below a less relevant section; retrieval quality needs evaluation on a larger synthetic set before publication claims.
3. `pip check` metadata remains inconsistent for ONNX Runtime, although actual project Chroma indexing and persistence validation succeeded using explicit BGE embeddings.
4. The earlier fresh `.venv` installation still could not resolve dependencies offline; no lockfile was created. Using this existing Conda environment does not establish that a clean independent install is reproducible.
5. Licensing, attribution, and publication rights review remains outstanding. No original drilling documents or derived artifacts were used.

## Files changed in this phase

- `README.md` — updated the runtime status to reflect synthetic BGE/Chroma/retrieval results and the LLM limitation.
- `docs/setup.md` — linked the latest Conda validation report.
- `docs/phase-3.3.2-rag-completion-report.md` — this report.

No source code, Conda environment files, packages, or Git metadata were changed in Phase 3.3.2. Temporary synthetic PDFs, indexes, databases, and API state were kept outside the repository and removed after validation. The original source workspace was not accessed.



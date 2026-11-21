# Phase 3.3.1 — Existing Conda Environment Validation

> Historical phase report: its results describe that phase and may predate the current implementation. See the [latest Phase 3.4 evaluation report](evaluation-report.md) for benchmark results and current evaluation limitations.


**Project:** DrillDoc Intelligence  
**Conda environment:** the existing local Conda environment named `rag`
**Python executable:** Python 3.11.16 from that environment
**Python version:** CPython 3.11.16  
**Result:** Runtime validation completed as far as the isolated local model availability allowed. This does not establish fresh-install reproducibility.

## Environment preservation

The environment was inspected and used in place. No package was installed, removed, upgraded, or downgraded; no Conda configuration was changed. Model loading was forced offline and all model-cache variables were pointed to fresh temporary paths. Test databases, traces, API state, and synthetic pipeline outputs were directed to external temporary directories under the user temp directory. No existing model cache, source corpus, database, vector store, or history was used. The original source workspace, Git configuration, and GitHub were not accessed.

## Dependency inventory

All requested imports were available in the `rag` environment:

| Dependency | Installed version | Status |
| --- | ---: | --- |
| FastAPI | 0.141.1 | Available |
| PyYAML | 6.0.3 | Available |
| Pydantic | 2.13.5 | Available |
| Uvicorn | 0.52.4 | Available |
| pypdf | 6.19.0 | Available |
| pdfplumber | 0.11.10 | Available |
| ChromaDB | 1.5.9 | Available; see consistency issue below |
| sentence-transformers | 6.0.1 | Available |
| PyTorch | 2.10.0 | Available |
| Transformers | 5.16.1 | Available |
| httpx | 0.28.1 | Available |

Other relevant packages found: `llama-cpp-python 0.3.35`, `huggingface_hub 1.31.0`, `python-multipart 0.0.32`, `setuptools 84.0.0`, and `onnxruntime-gpu 1.30.0`.

`pip check` reports one unresolved environment inconsistency: **ChromaDB 1.5.9 requires the `onnxruntime` distribution, which is not installed**. `onnxruntime-gpu` is present but does not satisfy that distribution requirement. No installation was attempted. The requested dependency list itself has no missing imports. FastAPI/Starlette also emitted a deprecation warning that its TestClient integration with httpx will be deprecated in favor of `httpx2`; tests nevertheless ran successfully.

## Test results

Full `unittest` discovery was run with model downloads disabled and data redirected outside the repository:

- `test_api.py`: 22 passed, including API health, unavailable-index readiness, retrieval errors, and mocked-provider validation.
- `test_citation_validation.py`: 4 passed.
- `test_runtime_config.py`: 4 passed.
- `test_table_chunker.py`: 2 passed.
- **Total: 32 passed.**

The first Conda run exposed two test-harness issues: temporary SQLite cleanup on Windows, and a provider-close assertion that omitted the service’s explicit `close()` lifecycle call. The tests were adjusted to release the temporary app state and explicitly close the service, then rerun successfully. A further synthetic table check found that decimal values in later cells could be mislabeled as row identifiers; `_row_labels` now only considers the first cell, with two regression tests. These are destination-only changes.

## API and frontend results

The API was started with Uvicorn bound only to `127.0.0.1`; no external address or service was contacted. It remained available until the smoke checks completed. On Windows it returned exit code 1 after the process was explicitly terminated; this was the requested shutdown, not a startup failure.

| Check | Result |
| --- | --- |
| `GET /health` | 200; `api_ready=true`, `rag_ready=false` |
| `GET /ready` | 503; `retrieval backend unavailable` |
| `GET /` | 200; static UI served |
| `GET /app.js`, `/app.css` | Both 200 |
| Conversation create/list/detail/rename/delete | 200/200/200/200/204; list empty after delete |
| `POST /api/v1/retrieval/query` | 503; no retrieval service without an index |
| `POST /api/v1/chat/stream` | 503; no retrieval service without an index |

The test suite also exercised the `/readiness` model-state response with an injected readiness result. The actual API was not asked to contact an Ollama service or inspect an existing model cache. Static UI/API paths and SSE event handling were checked against backend declarations; an actual successful streaming answer was not possible without an index and provider.

## Synthetic PDF and RAG pipeline

A two-page, 3,233-byte synthetic engineering test report was generated outside the repository. It contains four section headings, technical paragraphs, a ruled three-column/four-row numerical table, and independent facts about pressure, temperature, flow, and sampling rate. No real or original document was used.

- **PDF extraction:** executed successfully; text extracted from both pages.
- **Structure detection:** executed; three structure entries and two clause records were produced. One page was flagged for manual review.
- **Table extraction:** one table detected; four structured rows were extracted with synthetic values 2.4 MPa, 180 degree C, and 12.0 metres. The table is marked complete by the extractor but still carries a visual-verification warning.
- **Normalization and chunking:** executed successfully; four chunks were produced (two clause chunks, one table parent, one table segment). The chunk report validated successfully, while marking three chunks for manual review and two headings as uncertain.
- **Metadata/citations:** synthetic document IDs, page ranges, table/section identifiers, and citation metadata were generated. Printed page numbers are absent because the test PDF has none.
- **BM25 retrieval:** executed successfully. Queries for “maximum inlet pressure”, “flow meter 18 litres per minute”, and “sampled every 5 seconds” retrieved the corresponding synthetic table or section chunks.
- **Embedding/indexing:** attempted with `local_files_only=true`, offline flags, and a fresh empty cache. It stopped while loading the configured exact model `BAAI/bge-m3`; that model was not present in the isolated cache. No substitute model was used. Chroma indexing was not reached, so no vector index was created.
- **Semantic/hybrid retrieval and ranking:** not reached because indexing could not complete. BM25 alone is the only retrieval path exercised end to end.
- **Answer generation and generated citations:** not executed. No LLM model was available in the isolated model root; no fake answer was used.

The indexing attempt confirms the model boundary and exact failure. Chroma itself was not runtime-exercised; the `pip check` ONNX Runtime discrepancy is an additional blocker to a clean Chroma validation.

## Files changed in Phase 3.3.1

- `README.md` — separated existing-environment results from fresh-install limitations.
- `docs/setup.md` — documented the Conda validation and linked this report.
- `backend/rag_system/chunking/table_chunker.py` — prevented numeric values in non-leading table cells from being labeled as row IDs.
- `tests/test_api.py` — Windows temporary-state cleanup and explicit service close in the lifecycle test.
- `tests/test_table_chunker.py` — added two row-label regression tests.
- `docs/phase-3.3.1-conda-validation-report.md` — this report.

No Conda environment files or package records were changed.

## Remaining blockers and interpretation

1. Fresh independent installation remains unresolved: the separate isolated `.venv` could not install offline, and no dependency lockfile has been generated. This Conda run does not change that result.
2. The synthetic project pipeline stops before embeddings because `BAAI/bge-m3` is unavailable locally. Embedding/indexing, semantic/hybrid retrieval, successful streaming, and actual LLM answers/citations remain unverified.
3. `pip check` reports missing `onnxruntime` for ChromaDB despite `onnxruntime-gpu` being present; Chroma operation needs a clean environment resolution/validation.
4. Static UI assets and API routes were exercised, but browser interaction and a successful streamed answer were not.
5. Publication still requires licensing, attribution, and rights review; original drilling-material rights remain unknown.

The existing Conda environment validates the tested behavior under Python 3.11.16 only. It does not prove that a fresh project environment can be installed reproducibly. Git was not initialized and nothing was published.


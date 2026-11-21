# Phase 3.3 — Runtime Validation Report

> Historical phase report: its results describe that phase and may predate the current implementation. See the [latest Phase 3.4 evaluation report](evaluation-report.md) for benchmark results and current evaluation limitations.


**Project:** DrillDoc Intelligence  
**Status:** Incomplete; runtime validation is blocked by unavailable dependencies under the no-external-services constraint.  
**Scope:** Destination only. No original corpus or derived data was accessed or copied.

## 1. Environment

- Windows, Python **3.11.9** (`CPython`), matching the project’s selected Python 3.11 baseline.
- Created a fresh environment at `.venv`; no prior project virtual environment was present.
- Installed in the environment: `pip 24.0`, `setuptools 65.5.0`. No application dependencies were installed.
- `pip check` reported no broken requirements among the installed environment packages; this does not validate the project dependency set.
- No package index, remote model, Ollama service, GitHub, or other external service was contacted.

## 2. Dependency audit

Direct runtime imports are represented in `pyproject.toml`: FastAPI/Uvicorn/Pydantic for the API, PyYAML for configuration, `python-multipart` for upload forms, pypdf and pdfplumber for PDF reading/extraction, ChromaDB for persistent vector storage, sentence-transformers/PyTorch/Hugging Face Hub for embeddings and model support. SQLite and Ollama’s HTTP client use the Python standard library.

Provider dependencies are grouped as extras: `llama-cpp-python` under `llama`, and Transformers under `qwen`. The default answering configuration selects a llama.cpp reasoning provider and a Transformers Qwen formatter, so that default needs both extras and local model files. The Ollama provider has no additional Python package, but requires a separately running local service and a locally available model. API tests use FastAPI’s TestClient and therefore require the `httpx` development dependency. Tests use `unittest`; the unused pytest dependency was removed.

The prior `requirements.txt` lower bounds were moved into `pyproject.toml`; `requirements.txt` now points to the project declaration to avoid a second, divergent dependency list. Direct dependency imports were statically audited with no undeclared external import roots found. Exact version resolution, transitive compatibility, and a lockfile remain **unverified**. A lockfile was not fabricated without a successful resolver run.

## 3. Installation result

An offline-only editable install was attempted. Build isolation stopped because the local cache had no `setuptools>=68` distribution. The venv’s bundled setuptools is 65.5.0, below that build requirement. No packages were installed. This was an infrastructure/package-availability failure; it does not establish an application defect or compatibility result.

## 4. Automated tests

A full `unittest` discovery was attempted after adding runtime path coverage:

- `test_citation_validation.py`: **4 passed**.
- `test_runtime_config.py`: **4 passed**. These verify configured data/model roots, data-root-relative paths, and rejection of repository-local data paths.
- `test_api.py`: **not executed**; module import failed because FastAPI is absent from the fresh environment. The API test expectations were updated for health readiness fields and a no-index readiness case, but those assertions remain unverified.
- Overall discovery: **8 passed, 1 test module import error**. No failure was silently skipped.

## 5. Synthetic RAG pipeline

No synthetic PDF was generated and no pipeline stages were claimed as tested. The required PDF, embedding, and vector dependencies are unavailable in the isolated environment, so PDF extraction, structure detection, normalization, chunking, metadata, embedding, Chroma indexing, semantic/BM25/hybrid retrieval, reranking, evidence selection, answer generation, and citation generation were not executed. The LLM provider boundary was not reached. No synthetic or real document, index, model cache, or evaluation artifact was created.

## 6. API and frontend

- **API startup attempted locally:** it stopped during import with `ModuleNotFoundError: yaml`; it did not bind a port.
- The staged app declares `/health`, `/ready`, `/readiness`, chat, conversation, and retrieval routes. Code inspection shows health reports `api_ready` and `rag_ready`; startup captures retrieval initialization errors and `/ready` returns 503 when retrieval is unavailable. This behavior is **inspected, not runtime-tested**.
- Frontend and backend chat/conversation paths and SSE event names were statically compared. The app’s SSE events include `start`, `status`, `answer_delta`, `citation`, and `done`; frontend parsing for response deltas, citations, and completion is present. Streaming behavior was **not runtime-tested**.
- Static frontend smoke test: a temporary loopback-only standard-library server returned HTTP 200 for `/`, `/app.js`, and `/app.css`; expected content was present. Node.js `--check` passed for `app.js`. This verifies static asset serving/syntax only, not integration through FastAPI or browser interaction.

## 7. Implementation status and limitations

- **Implemented:** external runtime data/model path guard; separate API/RAG readiness fields; missing-index startup handling; local static UI and declared API routes.
- **Tested:** citation validation and runtime path guard tests; static asset HTTP responses; JavaScript syntax.
- **Partially tested:** dependency declarations/import roots; API/frontend route and streaming contract by inspection only.
- **Requires external/local setup:** dependency installation; local embedding model; configured LLM provider and model files or a local Ollama service.
- **Not verified:** API process, API tests, no-index HTTP readiness, actual synthetic PDF processing, vector indexing/retrieval, reranking, LLM answer generation, end-to-end citations, or browser interaction.

The project remains an experimental retrieval and question-answering application. It is not represented as production-ready or as a fully autonomous agent.

## 8. Files modified in Phase 3.3

- `pyproject.toml` — carried forward the existing dependency lower bounds and removed unused pytest from the development extra.
- `requirements.txt` — made `pyproject.toml` the single dependency declaration source.
- `README.md` — updated the status from “not run” to the measured Phase 3.3 test/install results.
- `docs/setup.md` — documented Python 3.11, provider extras, and the observed installation limitation.
- `tests/test_api.py` — updated health payload expectations and added a temporary-data-root missing-index readiness test (not executable until API dependencies are installed).
- `tests/test_runtime_config.py` — added four dependency-free path configuration tests.
- `backend/static_app/index.html` — removed leftover energy tagline text.
- `docs/phase-3.3-validation-report.md` — this report.
- `.venv/` — newly created isolated environment; ignored by `.gitignore`.

## 9. Remaining blockers before GitHub publication

1. Resolve dependencies in an authorized environment, record exact Python/package versions, and generate a lockfile or other reproducible resolution artifact.
2. Run the API tests, including the no-index readiness test, and fix any genuine failures.
3. Run the local API and validate health, readiness, chat, conversation routes, and SSE behavior.
4. Create and process a synthetic engineering PDF through extraction, chunking, embedding, Chroma, retrieval, and answering; report the actual stopping point if model/provider setup is unavailable.
5. Complete license, attribution, and publication-rights review. The rights status for the original drilling material remains unknown; no original data was used in this validation.
6. Recheck the clean destination after validation before any later GitHub preparation. Git was not initialized and nothing was published.

The Phase 3.3 definition of done is **not met** because dependency resolution, API execution, and synthetic end-to-end validation remain blocked.


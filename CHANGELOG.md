# Staging change log

## Finalization work (unreleased)

### Changed

- Removed the incomplete Ollama model-management routes, persistence module, and tests; retained the existing llama.cpp provider and made it the documented API answering path.
- Corrected local Llama model readiness and an uninitialized chat document hint that blocked the first API question.
- Updated setup, data/model, architecture, security, README, and historical-report guidance for the local Llama portfolio edition.
- Added an MIT license for original author-owned code after checking for bundled third-party assets and notices; third-party packages and models remain under their own terms. Pinned direct installation requirements and declared ChromaDB's ONNX runtime dependency.
- Verified the existing 44-test suite and a single real Llama-backed API/frontend smoke test using only the synthetic engineering fixture. Benchmark measurements were not changed.

## Phase 3.2 initial private candidate

### Added

- `.gitignore` covering local environments, credentials files, document/data artifacts, model files, indexes, databases, logs, caches, and generated outputs.
- `CHANGELOG.md` recording the staging changes.
- `README.md`, `SECURITY.md`, and `docs/architecture.md`, `docs/data-and-models.md`, `docs/setup.md` for the private development candidate.
- `pyproject.toml` with Python 3.11 baseline and proposed core, optional-provider, and development dependencies; exact version resolution remains pending.
- `backend/rag_system/runtime_config.py` for a configurable external data root and rejection of repository-local runtime paths.

### Modified copied files

- Identity and UI: `backend/static_app/app.js`, `backend/static_app/index.html` (generic text mark, removed fixed document selection, aligned conversation detail route).
- API and chat: `backend/rag_system/api/app.py`, `backend/rag_system/api/schemas.py`, `backend/rag_system/api/service.py`, `backend/rag_system/chat/service.py` (neutral configuration, external storage, degraded startup/readiness behavior, generic document handling, trace default off).
- Model configuration: `backend/rag_system/llm_runtime.py`, `backend/rag_system/model_paths.py`, `backend/rag_system/answering/core.py`, `backend/rag_system/answering/evidence_packing.py`, `backend/rag_system/answering/run.py` (environment-based model/provider paths and external evaluation output).
- Retrieval, indexing, ingestion, and hybrid configuration: `backend/rag_system/hybrid_retrieval/run.py`, `backend/rag_system/indexing/run.py`, `backend/rag_system/ingestion/run.py`, `backend/rag_system/retrieval/run.py`, `backend/rag_system/chunking/run.py` (external data handling; explicit output roots for ingestion, chunking, and indexing; safe document IDs).
- Runtime configs: `backend/rag_system/config/answering.yaml`, `backend/rag_system/config/api.yaml`, `backend/rag_system/config/hybrid_retrieval.yaml`, `backend/rag_system/config/indexing.yaml`, `backend/rag_system/config/retrieval.yaml` (neutral collection and external-root defaults, aligned index manifest location).

### Exclusions and limits

- Only the 86 explicitly allowlisted source candidates were copied. No original documents, extracted text, chunks, databases/indexes, traces, generated answers, logs, model files/caches, virtual environments, dashboard files, branded assets, or audit/planning records were transferred.
- Copyright/third-party notices were preserved. No license was selected pending rights review.
- No tests or application code were executed, no packages were installed, and no Git repository was initialized.

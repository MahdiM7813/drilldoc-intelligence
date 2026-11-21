# Setup

Python 3.11 is the supported baseline. Runtime data, model files, indexes, chat databases, and traces belong outside the repository. `requirements.txt` pins direct package versions for CPython 3.11 on Windows x64. Transitive packages are resolved by pip, so this is not a hash-locked dependency graph. The clean-install procedure was attempted in a temporary Python 3.11 environment but was not verified: the configured default PyPI index probe timed out, and an earlier HTTPS request failed with an authentication error. No proxy or custom pip certificate override was configured, and certificate verification was not bypassed. This does not establish whether the cause is a local trust-chain issue or package-index/network authentication policy. The application serves its static frontend through FastAPI, so Node.js is not required.

## Tests and API without an LLM

Create an environment and install the pinned direct dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

Installation requires access to the configured package index. Set `PYTHONPATH` and run model-free tests/start the API:

```powershell
$env:PYTHONPATH = "backend"
python -m unittest discover -s tests -p "test_*.py" -v
python backend/run.py
```

Open `http://127.0.0.1:9000/`. `/health` reports API and retrieval-backend liveness; `/ready` also requires a compatible index and configured Llama model. The static interface can load without an index, but chat requires ready retrieval and local-model resources. API tests use local test doubles and do not require model inference.

## Synthetic retrieval with local BGE-M3

Ensure `BAAI/bge-m3` and the Python dependencies are already available locally. This command creates a synthetic PDF, extracts and chunks it, embeds it, and creates a fresh Chroma index entirely under an external temporary directory:

```powershell
$work = Join-Path $env:TEMP "drilldoc-synthetic-run"
$env:PYTHONPATH = "backend"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python scripts/evaluation/prepare_synthetic_corpus.py --work-dir $work --embedding-model BAAI/bge-m3 --embedding-device cpu
```

The script refuses to reuse an existing work directory. Its `preparation_summary.json` reports the PDF, chunks, index, and manifest paths. To measure retrieval on the included 30-question synthetic dataset, use `scripts/evaluation/run_benchmark.py --retrieval-only` with the `--chunks`, `--manifest`, `--chroma-root`, and new external `--output-dir` paths from that summary. The benchmark is diagnostic; see [the latest evaluation report](evaluation-report.md) for limitations.

## Local Llama GGUF answer generation

The API uses the configured local Llama 3.1 8B GGUF through llama.cpp for both answer and formatting passes. BGE-M3 must also be available locally for embedding and retrieval. Set machine-local paths before starting the API; do not write them into project configuration or commit them:

```powershell
$env:DRILLDOC_LLM_PROVIDER = "llama_cpp"
$modelDir = Join-Path $env:LOCALAPPDATA "DrillDoc\models"
$env:DRILLDOC_LLM_REASONING_MODEL = Join-Path $modelDir "Llama-3.1-8B-Instruct-Q4_K_M.gguf"
$env:DRILLDOC_DATA_DIR = Join-Path $env:LOCALAPPDATA "DrillDoc\data"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python backend/run.py
```

The file must be a compatible GGUF. `llama-cpp-python` is pinned in `requirements.txt`; local model weights are not included. `DRILLDOC_LLM_REASONING_MODEL` overrides the configured model file; `DRILLDOC_ANSWERING_CONFIG` can select a separate answering YAML file. A successful `/ready` checks that the configured local model file exists; it does not prove that model loading or inference will succeed. The existing Conda environment passed 44/44 automated tests and completed one real synthetic PDF-to-answer API/frontend smoke test using local BGE-M3 and Llama 3.1 8B GGUF resources. Fresh Python 3.11 dependency installation remains unverified because of the package-index TLS/authentication issue described above. The benchmark report's answer-quality and citation limitations still apply.

## External paths and configuration

`DRILLDOC_DATA_DIR` stores runtime data; `DRILLDOC_MODEL_DIR` optionally sets the model directory; and `DRILLDOC_VECTOR_INDEX_DIR` selects the Chroma location. Other API settings include `DRILLDOC_API_CONFIG`, `DRILLDOC_API_HOST`, `DRILLDOC_API_PORT`, `DRILLDOC_CHAT_DB`, `DRILLDOC_STANDARDS_REGISTRY`, `DRILLDOC_STANDARDS_UPLOAD_DIR`, `DRILLDOC_HYBRID_CONFIG`, and `DRILLDOC_COLLECTION_ID`. Keep all such paths outside the repository. See [data and models](data-and-models.md), [architecture](architecture.md), and [the latest evaluation report](evaluation-report.md). Older phase reports are historical snapshots.

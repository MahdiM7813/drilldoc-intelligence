# Phase 3.4 — RAG Evaluation Methodology

## Purpose and existing components

This benchmark reuses the project’s extraction (`rag_system.ingestion`), structure-aware chunking, BGE-M3 indexing and Chroma store, BM25 retrieval, vector retrieval, hybrid RRF and deterministic reranking, evidence selection/packing, Llama provider, and API citation validation. Before this phase, the project had engineering smoke queries and API/citation tests, but no ranking metrics, labeled question set, answer-scoring module, or reproducible benchmark runner. This phase adds small evaluation utilities and scripts rather than a second retrieval implementation.

## Synthetic corpus and ground truth

`tests/fixtures/synthetic_engineering_corpus.json` is the hand-authored source of one fictional hydrostatic qualification procedure. It contains ten numbered clauses, three numeric tables, unit-bearing facts, cross-clause dependencies, calculations, and five deliberately omitted facts. `scripts/evaluation/build_synthetic_pdf.py` writes the fixture as a 13-page PDF using PDF primitives and the existing `pypdf` dependency; no PDF-generation package is required. The generated PDF is transient and must be written outside the repository.

`tests/fixtures/synthetic_rag_eval.jsonl` has 30 independently annotated questions: five each for direct facts, table facts, numerical/unit facts, multi-chunk questions, multi-step questions, and unanswerable questions. Each row has a stable ID, expected answer, document ID, evidence aliases, category, answerability, and expected abstention. Ground truth comes from the fixture text and deterministic arithmetic; it is not produced by an LLM. Dataset integrity and the 5-per-category distribution are tested.

Evidence aliases `C01`–`C10` identify numbered clauses; `T1:R1`–`T3:R3` identify rows on their table page. At retrieval scoring time aliases resolve to the actual chunk IDs. When several table rows share one segment chunk, relevance is counted once at chunk level. A chunk is relevant if it contains an annotated evidence unit sufficient for the question. Retrieval metrics use the 25 answerable questions; the five unanswerable questions have no relevant chunk and are assessed separately for abstention.

There is no parameter tuning or held-out development set in this small first benchmark. Results should be treated as diagnostic, not as a generalization estimate.

During an initial exploratory answer pass, S04's phrase “pressure margin” was found ambiguous because the 2.4 MPa setpoint is above the 1.8 MPa hose rating. The final fixture asks how much the setpoint exceeds the rating and annotates “0.6 MPa over the limit.” The corrected S04 query was rerun with Llama and the full retrieval evaluation was rerun; reported results use that final wording. This is a ground-truth wording correction, not parameter tuning.

## Metrics

- **Recall@k:** relevant retrieved chunk IDs in the first k, divided by all annotated relevant chunk IDs.
- **Precision@5:** relevant unique chunk IDs in the first five divided by five.
- **MRR:** reciprocal rank of the first relevant chunk; zero when no relevant chunk is found.
- **nDCG@5:** binary-relevance discounted cumulative gain divided by ideal gain.
- **Normalized exact match:** equality after Unicode normalization, case folding, punctuation cleanup, and whitespace normalization. This is intentionally strict and is not a semantic score.
- **Numeric/unit match:** every annotated literal numeric value and supported engineering unit appears in the output. Identifier suffixes such as `G-17` are excluded from numeric scoring. This presence check does not validate arithmetic relationships or entailment.
- **Citation precision:** relevant cited chunks divided by valid cited chunks. **Citation coverage:** relevant gold chunks cited divided by all relevant gold chunks. **Citation validity:** citations that refer to an item in the retrieved evidence divided by all citations.
- **Abstention:** explicit uncertainty wording or `insufficient` confidence. Wording and confidence are reported separately because a verbal abstention may still have a non-insufficient confidence label.
- **Faithfulness and unsupported natural-language claims:** require manual review. Automated numeric, unit, and citation-provenance checks do not establish semantic support.

Metrics are macro-averaged over questions. Retrieval systems use the same fixed questions, gold aliases, 15-candidate pool, and top-five scoring depth. BM25 uses the project tokenizer; semantic retrieval uses BGE-M3; hybrid RRF combines vector and keyword ranks with weights 1.0 and 1.2 and `rrf_k=60`; the existing reranker then applies the project’s configured phrase and metadata signals. No tuning was performed on the evaluation questions.

## Ablations

1. BM25 only.
2. BGE-M3 semantic only.
3. Hybrid reciprocal-rank fusion, before reranker signals.
4. Hybrid with the existing reranker.
5. Hybrid with reranker followed by production evidence selection and packing.

For step 5, this phase measures whether annotated gold chunks survive selection and packing. It does not compare LLM answer quality with selection disabled, so no answer-quality improvement is attributed to context selection.

## Answer evaluation protocol

The answer benchmark uses the existing `llama.cpp` provider and local `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` model. The query path uses the actual project service: hybrid retrieval, context selection/packing, one Llama generation pass, backend validation, and citation mapping. Raw Llama text and the final validated application answer are recorded separately. The Qwen provider is not used as a substitute.

Literal exact/numeric/unit scores are automatic. Claims requiring interpretation, arithmetic correctness beyond literal presence, citation entailment, and overall faithfulness are marked for manual review instead of receiving invented judgments. Unanswerable responses are checked for explicit abstention language and confidence separately.

## Performance protocol

The corpus preparation runner records one elapsed sample each for extraction, chunking, and indexing. Embedding batch latency, retrieval latency, reranking latency, and per-query answer latency are recorded. Median/P95 are reported only for repeated per-query measurements; single stage runs are reported as single samples. P95 uses the nearest-rank definition `ceil(0.95 * n)`. Cold model initialization is separate where measured. CPU/GPU settings and token/context configuration are recorded. Memory/VRAM use is reported only if observed; unavailable telemetry is not inferred.

## Reproduction

Use the existing `rag` Conda environment and cached models. Do not install packages. From the project root in PowerShell, select a new external work directory each run because both scripts refuse to overwrite existing outputs:

```powershell
$ProjectRoot = (Get-Location).Path
$env:PYTHONPATH = Join-Path $ProjectRoot 'backend'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$prep = Join-Path $env:TEMP ("drilldoc-phase-3.4-" + [guid]::NewGuid().ToString('N'))
$llamaModel = $env:DRILLDOC_LLM_REASONING_MODEL
if (-not $llamaModel) { throw 'Set DRILLDOC_LLM_REASONING_MODEL to a local GGUF model file.' }
python scripts/evaluation/prepare_synthetic_corpus.py --work-dir $prep --embedding-model BAAI/bge-m3 --embedding-device cpu
python scripts/evaluation/run_benchmark.py `
  --chunks (Join-Path $prep 'processed\chunked\chunks\chunks.jsonl') `
  --manifest (Join-Path $prep 'index\index_manifest.json') `
  --chroma-root (Join-Path $prep 'chroma') `
  --dataset 'tests\fixtures\synthetic_rag_eval.jsonl' `
  --output-dir (Join-Path $prep 'evaluation') `
  --text-collection eval_text --table-collection eval_tables `
  --llama-model $llamaModel `
  --context 2048 --threads 8 --gpu-layers 0 --max-new-tokens 192
```

To test only the retrieval part, add `--retrieval-only` and omit `--llama-model`. The full mode performs real local inference for all fixture questions and writes the detailed JSON to the external output directory. The corpus preparation output includes paths, an extraction/chunk/index summary, and the Chroma manifest. No repository-local data directory is used.

Run tests with:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'backend'
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Known constraints

The fixture is one synthetic document, not a multi-document corpus. It is intentionally small and the table pages expose an extraction grouping issue described in the evaluation report. Literal matching and binary chunk relevance do not measure broad semantic correctness. The benchmark has no human-rated faithfulness labels and no measured memory/VRAM telemetry. Model latency is hardware- and cache-dependent; do not compare it to GPU runs or treat this single-device result as a service-level guarantee.

"""Generate, extract, chunk, embed, and index the public synthetic fixture offline."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "backend"))
from rag_system.chunking.run import run as chunk_run
from rag_system.indexing.run import run as index_run
from rag_system.ingestion.run import run as extraction_run
from build_synthetic_pdf import build as build_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path, help="A new external directory; existing paths are refused")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--embedding-device", default="cpu")
    args = parser.parse_args()
    root = args.work_dir.expanduser().resolve()
    if root == PROJECT or PROJECT in root.parents:
        parser.error("corpus and index outputs must be outside the repository")
    if root.exists():
        parser.error(f"refusing to overwrite existing work directory: {root}")
    root.mkdir(parents=True)
    os.environ.update({"DRILLDOC_DATA_DIR": str(root / "runtime-data"), "HF_HUB_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "PYTHONDONTWRITEBYTECODE": "1"})
    pdf = root / "synthetic-qualification.pdf"
    pdf_result = build_pdf(PROJECT / "tests/fixtures/synthetic_engineering_corpus.json", pdf)
    extract_config = root / "extraction.json"
    extract_config.write_text(json.dumps({"document_type": "synthetic_engineering_test", "authority_status": "synthetic",
        "language": "en", "empty_page_character_threshold": 20, "low_text_character_threshold": 120,
        "table_detection": {"vertical_strategy": "lines", "horizontal_strategy": "lines"}, "verification_terms": []}), encoding="utf-8")
    processed = root / "processed"
    start = time.perf_counter(); extraction = extraction_run(pdf, "synthetic-qualification-EN", processed, extract_config)
    extraction_seconds = time.perf_counter() - start
    chunk_config = PROJECT / "backend/rag_system/config/chunking.yaml"
    start = time.perf_counter()
    chunking = chunk_run(processed / "documents/registry/document_registry.json",
        processed / "outputs/raw_text/pages.jsonl", processed / "outputs/reports/detected_structure.json",
        processed / "outputs/reports/clauses.jsonl", processed / "outputs/tables/tables.json",
        chunk_config, processed / "chunked")
    chunking_seconds = time.perf_counter() - start
    index_config = {
        "embedding": {"model_name": args.embedding_model, "device": args.embedding_device, "batch_size": 2,
            "precision": "float32", "normalize_embeddings": True, "show_progress": False,
            "trust_remote_code": False, "local_files_only": True},
        "vector_store": {"provider": "chromadb", "persist_directory": str(root / "chroma"),
            "distance_metric": "cosine", "text_collection": "eval_text", "table_collection": "eval_tables"},
        "indexing": {"index_table_parents": False, "require_embedding_text": True,
            "fail_on_conflicting_ids": True, "incremental_by_default": True, "prune_by_default": False},
        "verification": {"smoke_test_top_k": 5},
    }
    config_path = root / "indexing.yaml"
    config_path.write_text(yaml.safe_dump(index_config, sort_keys=False), encoding="utf-8")
    start = time.perf_counter()
    indexing = index_run(chunks_path=processed / "chunked/chunks/chunks.jsonl",
        definitions_path=processed / "chunked/chunks/definitions.jsonl",
        tables_path=processed / "chunked/chunks/table_chunks.jsonl",
        registry_path=processed / "documents/registry/document_registry.json",
        config_path=config_path, rebuild=True, prune=True, output_root=root / "index")
    indexing_seconds = time.perf_counter() - start
    result = {"pdf": pdf_result, "extraction_seconds": extraction_seconds,
        "extraction": {"pages": extraction["pdf_page_count"], "clauses": extraction["detected_clause_count"],
                       "detected_table_entities": extraction["detected_table_count"]},
        "chunking_seconds": chunking_seconds, "chunk_count": chunking["total_chunks"],
        "chunk_types": chunking["counts_by_content_type"], "indexing_seconds": indexing_seconds,
        "indexed_count": indexing["indexed_chunk_count"], "embedding_model": args.embedding_model,
        "embedding_dimension": indexing["embedding_dimension"], "device": indexing["device"],
        "chunks_path": str(processed / "chunked/chunks/chunks.jsonl"),
        "manifest_path": str(root / "index/index_manifest.json"), "chroma_root": str(root / "chroma"),
        "text_collection": "eval_text", "table_collection": "eval_tables"}
    (root / "preparation_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

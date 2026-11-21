from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from rag_system.runtime_config import ensure_external_path, resolve_config_paths

from rag_system.indexing.chroma_store import ChromaStore
from rag_system.indexing.embedder import LocalBGEEmbedder
from rag_system.indexing.loader import load_json, prepare_records, sha256_file
from rag_system.indexing.manifest import backup_manifest, build_manifest
from rag_system.indexing.validator import validate_records


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "rag_system"


SMOKE_QUERIES = [
    ("quality levels for weld imperfections", "text"),
    ("definition of projected area", "text"),
    ("lack of fusion", "tables"),
    ("incomplete root penetration", "tables"),
    ("Table 1 quality level B", "tables"),
    ("سطوح کیفی عیوب جوش", "text"),
    ("تعریف سطح تصویرشده", "text"),
    ("عدم ذوب", "tables"),
    ("نفوذ ناقص ریشه", "tables"),
]


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return resolve_config_paths(config)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in values), encoding="utf-8")


def _report_fa(report: dict) -> str:
    return f"""# گزارش Stage RAG-03 - embedding محلی و نمایهٔ پایدار

## خلاصه

- مدل: `{report['embedding_model']}`
- دستگاه: `{report['device']}`
- بُعد embedding: {report['embedding_dimension']}
- رکوردهای بارگذاری‌شده: {report['total_loaded_records']}
- رکوردهای یکتای قابل نمایه‌سازی: {report['unique_indexable_records']}
- رکوردهای موجود در نمایه: {report['indexed_chunk_count']}
- افزوده/به‌روزشده/بدون تغییر: {report['changes']['added']} / {report['changes']['updated']} / {report['changes']['unchanged']}
- حذف‌شده با prune: {report['changes']['removed']}
- skipped: {report['skipped_chunk_count']}
- failed: {report['failed_chunk_count']}
- duplicate: {report['duplicate_chunk_count']}
- conflict: {report['conflict_count']}
- نیازمند بازبینی انسانی: {report['manual_review_chunk_count']}
- ورودی truncate‌شده: {report['truncated_record_count']}

## ذخیره‌سازی

- مسیر Chroma: `{report['persistent_directory']}`
- manifest: `{report['manifest_path']}`
- index ID قطعی: `{report['index_id']}`
- collectionها: `{json.dumps(report['collections'], ensure_ascii=False, sort_keys=True)}`

## یکپارچگی و نسخه‌ها

- hash فایل‌های ورودی: `{json.dumps(report['input_files'], ensure_ascii=False, sort_keys=True)}`
- sentence-transformers: `{report['sentence_transformers_version']}`
- ChromaDB: `{report['chroma_version']}`
- حداکثر ورودی مدل: {report['model_max_sequence_length']} token
- cache مدل: `{report['model_cache_location']}`

## آزمون‌های smoke

{len(report['smoke_queries'])} پرس‌وجوی انگلیسی/فارسی فقط برای بررسی فنی اجرا شد. نتایج شامل ID، فاصله، citation و نوع محتوا است؛ هیچ ادعایی دربارهٔ دقت retrieval یا پاسخ طبیعی ارائه نمی‌شود.

## محدودیت‌ها

کیفیت نمایه به chunkهای Stage RAG-02 وابسته است. تعریف‌های استخراج‌شده و جدول‌های چندصفحه‌ای همچنان نیازمند بازبینی هستند. جست‌وجوی این مرحله صرفاً smoke test برداری است و Retriever نهایی، query rewriting، hybrid search، reranking، پرسش‌وپاسخ، agent، rule engine یا تصمیم پذیرش/رد پیاده‌سازی نشده است.
"""


def _cache_location() -> str:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return str(Path(HF_HUB_CACHE).resolve())
    except Exception:
        return "unknown"


def run(*, chunks_path: Path, definitions_path: Path, tables_path: Path, registry_path: Path, config_path: Path, rebuild: bool = False, prune: bool = False, embedder: Any | None = None, output_root: Path) -> dict:
    output_root = ensure_external_path(output_root)
    config = load_config(config_path)
    registry = load_json(registry_path)
    input_paths = [chunks_path, definitions_path, tables_path, registry_path]
    input_files = [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in input_paths]
    records, load_report = prepare_records([chunks_path, definitions_path, tables_path], registry, config)
    validate_records(records, registry)
    load_report["prepared_records"] = records
    reports_dir = output_root / "reports"
    vector_root = Path(config["vector_store"]["persist_directory"])
    manifest_path = output_root / "index_manifest.json"
    failed_path = reports_dir / "failed_records.jsonl"
    write_json(reports_dir / "duplicate_records.json", {"duplicates": load_report["duplicate_records"], "conflicts": load_report["conflicts"], "skipped": load_report["skipped"]})
    write_jsonl(failed_path, [])

    model = embedder or LocalBGEEmbedder(config["embedding"])
    truncated = model.analyze_lengths(records) if hasattr(model, "analyze_lengths") else []
    store = ChromaStore(vector_root, config["vector_store"])
    previous = load_json(manifest_path) if manifest_path.is_file() else None
    if rebuild:
        print("WARNING: --rebuild explicitly requested; configured collections will be recreated.", file=sys.stderr)
        backup_manifest(manifest_path)
        store.rebuild_collections()
    try:
        sync_stats = store.sync(records, model, prune=prune)
    except Exception:
        write_jsonl(failed_path, getattr(locals().get("sync_stats", {}), "failed_records", []) if False else [{"chunk_id": record.chunk_id, "status": record.index_status} for record in records if record.index_status == "failed"])
        raise
    if sync_stats["failed_records"]:
        write_jsonl(failed_path, sync_stats["failed_records"])
    counts = store.counts()
    expected = Counter(record.collection for record in records)
    if not prune:
        expected = Counter(counts)  # Existing out-of-scope records are retained by default.
    sample_ids = {name: next((record.chunk_id for record in records if record.collection == name), "") for name in store.collection_names}
    verification = store.verify(dict(expected), model.dimension, sample_ids)

    query_vectors = model.encode_queries([query for query, _ in SMOKE_QUERIES])
    smoke_results = []
    for (query, target), vector in zip(SMOKE_QUERIES, query_vectors):
        collection = config["vector_store"]["text_collection"] if target == "text" else config["vector_store"]["table_collection"]
        smoke_results.append(store.smoke_query(collection, query, vector, int(config["verification"]["smoke_test_top_k"])))
    write_json(reports_dir / "smoke_query_results.json", smoke_results)

    manifest = build_manifest(config=config, input_files=input_files, embedder=model, collection_counts=counts, load_report=load_report, sync_stats=sync_stats, truncated=truncated, previous=previous, rebuild=rebuild)
    write_json(manifest_path, manifest)
    report = dict(manifest)
    report.update({
        "total_loaded_records": load_report["total_loaded_records"],
        "unique_indexable_records": len(records),
        "persistent_directory": str(vector_root), "manifest_path": str(manifest_path),
        "model_cache_location": _cache_location(), "smoke_queries": smoke_results,
        "verification": verification, "truncated_records": truncated,
        "skipped_records": load_report["skipped"],
    })
    write_json(reports_dir / "indexing_report.json", report)
    (reports_dir / "INDEXING_REPORT_FA.md").write_text(_report_fa(report), encoding="utf-8")
    store.close()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or incrementally update a fully local persistent Chroma index.")
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--definitions", required=True, type=Path)
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--prune", action="store_true")
    args = parser.parse_args(argv)
    report = run(chunks_path=args.chunks, definitions_path=args.definitions, tables_path=args.tables, registry_path=args.registry, config_path=args.config, rebuild=args.rebuild, prune=args.prune, output_root=args.output_root)
    print(json.dumps({k: report[k] for k in ("index_id", "embedding_model", "device", "embedding_dimension", "total_loaded_records", "unique_indexable_records", "collections", "duplicate_chunk_count", "conflict_count", "truncated_record_count", "changes")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

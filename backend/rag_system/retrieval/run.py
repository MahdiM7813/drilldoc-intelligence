from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from rag_system.runtime_config import data_root, resolve_config_paths

from .retriever import LocalRetriever


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "retrieval.yaml"
DEFAULT_MANIFEST = data_root() / "indexes" / "index_manifest.json"
SMOKE_QUERIES = [
    "lack of fusion",
    "porosity quality level B",
    "definition of projected area",
    "عدم ذوب",
    "سطح کیفی B",
    "تعریف سطح تصویرشده",
]


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return resolve_config_paths(config)


def execute_query(query: str, config_path: Path = DEFAULT_CONFIG, manifest_path: Path = DEFAULT_MANIFEST, top_k: int | None = None) -> dict:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    retriever = LocalRetriever(load_config(config_path), manifest_path)
    try:
        results = retriever.retrieve(query, top_k=top_k)
        return {"query": query, "top_k": top_k or retriever.config["retrieval"]["top_k"], "results": [x.to_dict() for x in results]}
    finally:
        retriever.close()


def run_smoke_evaluation(config_path: Path = DEFAULT_CONFIG, manifest_path: Path = DEFAULT_MANIFEST, output_path: Path | None = None) -> list[dict]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    retriever = LocalRetriever(load_config(config_path), manifest_path)
    try:
        output = [{"query": query, "results": [x.to_dict() for x in retriever.retrieve(query)]} for query in SMOKE_QUERIES]
    finally:
        retriever.close()
    destination = output_path or data_root() / "evaluations" / "retrieval_smoke_results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieve evidence chunks only; no answer generation.")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--smoke-eval", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.query) == bool(args.smoke_eval):
        parser.error("provide exactly one of --query or --smoke-eval")
    value = run_smoke_evaluation(args.config, args.manifest) if args.smoke_eval else execute_query(args.query, args.config, args.manifest, args.top_k)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

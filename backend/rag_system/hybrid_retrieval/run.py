from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from rag_system.runtime_config import resolve_config_paths
from rag_system.retrieval.run import load_config as load_rag04_config
from .hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "hybrid_retrieval.yaml"
def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")); base = path.parent
    raw["rag04"] = load_rag04_config((base/raw["rag04_config"]).resolve())
    return resolve_config_paths(raw)
def main() -> None:
    parser = argparse.ArgumentParser(description="Stage RAG-06 hybrid retrieval")
    parser.add_argument("--query", required=True); parser.add_argument("--top-k", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); args = parser.parse_args()
    config = load_config(args.config); retriever = HybridRetriever(config, Path(config["manifest_path"]))
    try: print(json.dumps([x.to_dict() for x in retriever.retrieve(args.query, args.top_k)], ensure_ascii=False, indent=2))
    finally: retriever.close()
if __name__ == "__main__": main()

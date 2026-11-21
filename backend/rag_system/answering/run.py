from __future__ import annotations
from rag_system.runtime_config import data_root, resolve_config_paths

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from rag_system.retrieval.run import load_config as load_retrieval_config
from rag_system.retrieval.retriever import LocalRetriever
from rag_system.model_paths import resolve_model_path

from .answer_generator import AnswerGenerator


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "answering.yaml"
SMOKE_QUERIES = [
    "What is lack of fusion?", "What is porosity?", "What is quality level B?",
    "عدم ذوب چیست؟", "تخلخل چیست؟", "سطح کیفی B چیست؟",
]


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    configured = os.environ.get("DRILLDOC_ANSWERING_CONFIG")
    config_path = Path(configured).expanduser().resolve() if configured else path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Answering configuration must be a mapping.")
    config = resolve_config_paths(config)
    for role, env_name in (("reasoning", "DRILLDOC_LLM_REASONING_MODEL"), ("formatter", "DRILLDOC_LLM_STRUCTURE_MODEL")):
        model_path = os.environ.get(env_name, "").strip()
        if model_path and role in config.get("providers", {}):
            config["providers"][role]["model_path"] = str(Path(model_path).expanduser().resolve())
    providers = config.get("providers", {})
    if providers.get("reasoning", {}).get("type") == "llama_cpp":
        reasoning_path = providers["reasoning"].get("model_path", "")
        if not os.environ.get("DRILLDOC_LLM_REASONING_MODEL") and os.environ.get("DRILLDOC_MODEL_DIR"):
            reasoning_path = resolve_model_path("llm_reasoning", reasoning_path)
            providers["reasoning"]["model_path"] = reasoning_path
        if not os.environ.get("DRILLDOC_LLM_STRUCTURE_MODEL"):
            providers.get("formatter", {})["model_path"] = reasoning_path
    provider = os.environ.get("DRILLDOC_LLM_PROVIDER", "").strip().lower()
    if provider:
        if provider not in {"llama_cpp", "transformers_qwen"}:
            raise ValueError("DRILLDOC_LLM_PROVIDER must select llama_cpp or transformers_qwen.")
        providers = config.setdefault("providers", {})
        for role in ("reasoning", "formatter"):
            if role in providers:
                providers[role]["type"] = provider
    return config


def build_provider(config: dict[str, Any]) -> LLMProvider:
    provider_type = config.get("type", "llama_cpp")
    if provider_type == "llama_cpp":
        from .llama_provider import LlamaCppProvider
        return LlamaCppProvider(config)
    if provider_type == "transformers_qwen":
        from .qwen_provider import QwenTransformersProvider
        return QwenTransformersProvider(config)
    raise ValueError(f"Unsupported local LLM provider: {provider_type}")


def build_generator(config: dict[str, Any]) -> AnswerGenerator:
    retrieval_config = load_retrieval_config(Path(config["retrieval"]["config_path"]))
    retriever = LocalRetriever(retrieval_config, Path(config["retrieval"]["manifest_path"]))
    return AnswerGenerator(
        retriever,
        lambda: build_provider(config["providers"]["reasoning"]),
        lambda: build_provider(config["providers"]["formatter"]),
        config,
    )


def execute(question: str, config_path: Path = DEFAULT_CONFIG, top_k: int | None = None) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config = load_config(config_path)
    generator = build_generator(config)
    return generator.answer(question, top_k or int(config["retrieval"]["top_k"])).model_dump()


def run_smoke(config_path: Path = DEFAULT_CONFIG, output_path: Path | None = None) -> list[dict[str, Any]]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    config = load_config(config_path)
    answers = build_generator(config).answer_many(SMOKE_QUERIES, int(config["retrieval"]["top_k"]))
    values = [answer.model_dump() for answer in answers]
    destination = output_path or data_root() / "evaluations" / "answering_smoke_results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an evidence-grounded local answer with citations.")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke-eval", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.query) == bool(args.smoke_eval):
        parser.error("provide exactly one of --query or --smoke-eval")
    value = run_smoke(args.config) if args.smoke_eval else execute(args.query, args.config, args.top_k)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


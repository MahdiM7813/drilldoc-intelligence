from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from rag_system.runtime_config import model_root

ANSWERING_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "answering.yaml"


def _model_path(configured: str, env_name: str) -> str:
    override = os.environ.get(env_name, "").strip()
    if override:
        return str(Path(override).expanduser().resolve())
    path = Path(configured).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    if path.suffix:
        return str((model_root() / path.name).resolve())
    return str((model_root() / path).resolve())


def inspect_llm_readiness() -> dict[str, Any]:
    """Report readiness for the configured local Llama model used by the API."""
    try:
        config_value = os.environ.get("DRILLDOC_ANSWERING_CONFIG")
        config_path = Path(config_value).expanduser().resolve() if config_value else ANSWERING_CONFIG_PATH
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        provider = config.get("providers", {}).get("reasoning", {})
        mode = os.environ.get("DRILLDOC_LLM_PROVIDER", str(provider.get("type", "llama_cpp"))).strip().lower()
        configured_path = str(provider.get("model_path") or "")
        model_path = _model_path(configured_path, "DRILLDOC_LLM_REASONING_MODEL") if configured_path else ""
        available = mode == "llama_cpp" and bool(model_path) and Path(model_path).is_file()
        role = {"mode": mode, "selected_model": model_path or None, "available": available}
    except (OSError, TypeError, AttributeError, yaml.YAMLError, ValueError):
        role = {"mode": "llama_cpp", "selected_model": None, "available": False}
    return {
        "status": "available" if role["available"] else "unavailable",
        "roles": {"llm_reasoning": role, "llm_structure": dict(role)},
    }

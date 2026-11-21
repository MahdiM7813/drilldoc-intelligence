"""Resolve optional local LLM files from user configuration."""
from __future__ import annotations

import os
from pathlib import Path

from rag_system.runtime_config import model_root

ENV_BY_KEY = {
    "llm_reasoning": "DRILLDOC_LLM_REASONING_MODEL",
    "llm_structure": "DRILLDOC_LLM_STRUCTURE_MODEL",
}
MODEL_ROOT = "DRILLDOC_MODEL_DIR"

def resolve_model_root() -> str:
    return str(model_root())

def resolve_model_path(key: str, fallback: str) -> str:
    env_name = ENV_BY_KEY.get(key)
    configured = os.environ.get(env_name, "").strip() if env_name else ""
    if configured:
        return str(Path(configured).expanduser().resolve())
    candidate = Path(fallback).expanduser()
    root = model_root()
    if candidate.is_absolute():
        return str((root / candidate.name).resolve())
    if key == "llm_reasoning" and candidate.suffix:
        return str((root / candidate.name).resolve())
    return fallback

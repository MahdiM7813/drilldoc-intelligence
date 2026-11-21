"""Runtime paths for user data and local model files; never rooted in the source tree."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def ensure_external_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    if candidate == repository or repository in candidate.parents:
        raise ValueError("Data, model, and output paths must be outside the repository.")
    return candidate


def data_root() -> Path:
    configured = os.environ.get("DRILLDOC_DATA_DIR")
    if configured:
        return ensure_external_path(configured)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return ensure_external_path(Path(os.environ["LOCALAPPDATA"]) / "DrillDoc Intelligence")
    if os.environ.get("XDG_DATA_HOME"):
        return ensure_external_path(Path(os.environ["XDG_DATA_HOME"]).expanduser() / "drilldoc-intelligence")
    return ensure_external_path(Path.home() / ".local" / "share" / "drilldoc-intelligence")


def model_root() -> Path:
    configured = os.environ.get("DRILLDOC_MODEL_DIR")
    return ensure_external_path(configured) if configured else ensure_external_path(data_root() / "models")


def resolve_data_path(value: str | Path | None, *, default: str | Path | None = None) -> Path:
    chosen = value if value not in (None, "") else default
    if chosen in (None, ""):
        raise ValueError("A data path must be configured.")
    candidate = Path(os.path.expandvars(os.path.expanduser(str(chosen))))
    if not candidate.is_absolute():
        candidate = data_root() / candidate
    return ensure_external_path(candidate)


_DATA_PATH_KEYS = {"path", "paths", "chunk_paths", "manifest_path", "registry_path", "definitions_path", "tables_path", "chunks_path", "persist_directory", "output_path", "output_dir", "output_root", "input_path", "input_dir", "upload_root", "standards_registry_path", "standards_upload_root", "chat_database_path"}
_CONFIG_PATH_KEYS = {"config_path", "rag04_config", "hybrid_config_path"}

def resolve_config_paths(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: resolve_config_paths(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_config_paths(item, key) for item in value]
    if isinstance(value, str) and key not in _CONFIG_PATH_KEYS and key is not None:
        lowered = key.lower()
        if key in _DATA_PATH_KEYS or lowered.endswith(("_path", "_paths", "_directory", "_dir", "_root")):
            return str(resolve_data_path(value))
    return value

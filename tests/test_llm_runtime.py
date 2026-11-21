from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from rag_system.llm_runtime import inspect_llm_readiness
from rag_system.answering.run import load_config


class LlamaReadinessTests(unittest.TestCase):
    def test_local_reasoning_model_controls_api_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "synthetic.gguf"
            model.write_bytes(b"test placeholder; readiness checks file presence only")
            config = root / "answering.yaml"
            config.write_text(yaml.safe_dump({"providers": {
                "reasoning": {"type": "llama_cpp", "model_path": "models/reasoning.gguf"},
                "formatter": {"type": "transformers_qwen", "model_path": "unused-qwen"},
            }}), encoding="utf-8")
            with patch.dict(os.environ, {
                "DRILLDOC_ANSWERING_CONFIG": str(config),
                "DRILLDOC_LLM_REASONING_MODEL": str(model),
            }, clear=False):
                readiness = inspect_llm_readiness()
        self.assertEqual(readiness["status"], "available")
        self.assertTrue(readiness["roles"]["llm_reasoning"]["available"])
        self.assertTrue(readiness["roles"]["llm_structure"]["available"])
        self.assertEqual(readiness["roles"]["llm_reasoning"]["selected_model"], str(model))

    def test_missing_local_model_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "answering.yaml"
            config.write_text(yaml.safe_dump({"providers": {
                "reasoning": {"type": "llama_cpp", "model_path": "missing.gguf"},
            }}), encoding="utf-8")
            with patch.dict(os.environ, {"DRILLDOC_ANSWERING_CONFIG": str(config), "DRILLDOC_DATA_DIR": str(root / "data")}, clear=False):
                readiness = inspect_llm_readiness()
        self.assertEqual(readiness["status"], "unavailable")
        self.assertFalse(readiness["roles"]["llm_reasoning"]["available"])

    def test_model_directory_configures_api_answer_provider(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "models"
            model_dir.mkdir()
            model = model_dir / "reasoning.gguf"
            model.write_bytes(b"test placeholder")
            config = root / "answering.yaml"
            config.write_text(yaml.safe_dump({
                "retrieval": {},
                "providers": {
                    "reasoning": {"type": "llama_cpp", "model_path": "models/reasoning.gguf"},
                    "formatter": {"type": "llama_cpp", "model_path": "models/reasoning.gguf"},
                },
            }), encoding="utf-8")
            with patch.dict(os.environ, {
                "DRILLDOC_ANSWERING_CONFIG": str(config),
                "DRILLDOC_DATA_DIR": str(root / "data"),
                "DRILLDOC_MODEL_DIR": str(model_dir),
            }, clear=False):
                resolved = load_config(config)
                readiness = inspect_llm_readiness()
        self.assertEqual(resolved["providers"]["reasoning"]["model_path"], str(model))
        self.assertEqual(resolved["providers"]["formatter"]["model_path"], str(model))
        self.assertEqual(readiness["status"], "available")

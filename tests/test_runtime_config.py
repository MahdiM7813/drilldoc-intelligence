from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_system.runtime_config import data_root, ensure_external_path, model_root, resolve_config_paths, resolve_data_path


REPOSITORY = Path(__file__).resolve().parents[1]


class RuntimeConfigTests(unittest.TestCase):
    def test_configured_data_and_model_roots_are_resolved_outside_repository(self):
        with tempfile.TemporaryDirectory(prefix="drilldoc-runtime-") as temporary:
            base = Path(temporary)
            with patch.dict(os.environ, {"DRILLDOC_DATA_DIR": str(base / "data"), "DRILLDOC_MODEL_DIR": str(base / "models")}):
                self.assertEqual(data_root(), (base / "data").resolve())
                self.assertEqual(model_root(), (base / "models").resolve())

    def test_relative_data_paths_are_anchored_to_configured_data_root(self):
        with tempfile.TemporaryDirectory(prefix="drilldoc-runtime-") as temporary:
            base = Path(temporary)
            with patch.dict(os.environ, {"DRILLDOC_DATA_DIR": str(base)}):
                self.assertEqual(resolve_data_path("chunks/input.json"), (base / "chunks/input.json").resolve())
                config = resolve_config_paths({"persist_directory": "indexes/chroma"})
                self.assertEqual(Path(config["persist_directory"]), (base / "indexes/chroma").resolve())

    def test_repository_local_data_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            ensure_external_path(REPOSITORY / "outputs")

    def test_repository_local_data_root_override_is_rejected(self):
        with patch.dict(os.environ, {"DRILLDOC_DATA_DIR": str(REPOSITORY / "data")}):
            with self.assertRaisesRegex(ValueError, "outside the repository"):
                data_root()


if __name__ == "__main__":
    unittest.main()

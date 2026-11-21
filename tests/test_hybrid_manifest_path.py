from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from rag_system.api.service import build_retrieval_service


class HybridManifestPathTests(unittest.TestCase):
    def test_api_builder_passes_manifest_as_path(self):
        with tempfile.TemporaryDirectory(prefix="drilldoc-manifest-path-") as temporary:
            root = Path(temporary)
            data = root / "data"
            index = data / "indexes" / "chroma"
            index.mkdir(parents=True)
            (index / "chroma.sqlite3").write_bytes(b"synthetic placeholder")
            chunks = data / "chunks" / "chunks.jsonl"
            chunks.parent.mkdir(parents=True)
            chunks.write_text('{"chunk_id":"synthetic-1"}\n', encoding="utf-8")
            config_path = root / "hybrid.yaml"
            retrieval_config = Path(__file__).resolve().parents[1] / "backend" / "rag_system" / "config" / "retrieval.yaml"
            config_path.write_text(yaml.safe_dump({
                "rag04_config": str(retrieval_config),
                "manifest_path": "indexes/index_manifest.json",
                "bm25": {"chunk_paths": ["chunks/chunks.jsonl"]},
                "hybrid": {"top_k": 3},
                "reranking": {},
            }), encoding="utf-8")
            fake_retriever = unittest.mock.MagicMock()
            expected_manifest = data / "indexes" / "index_manifest.json"
            with patch.dict(os.environ, {"DRILLDOC_DATA_DIR": str(data)}), \
                 patch("rag_system.api.service.HybridRetriever", return_value=fake_retriever) as constructor:
                service = build_retrieval_service({
                    "hybrid_config_path": str(config_path),
                    "vector_index_path": str(index),
                    "top_k_default": 3,
                    "timeout_seconds": 5,
                })
                self.assertIs(service.retriever, fake_retriever)
                self.assertIsInstance(constructor.call_args.args[1], Path)
                self.assertEqual(constructor.call_args.args[1], expected_manifest.resolve())
                service.close()


if __name__ == "__main__":
    unittest.main()

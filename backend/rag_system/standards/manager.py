from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .registry import JsonStandardRegistry
from .schemas import IndexingResult, StandardEntry, StandardStatus
from .storage import StandardFileStorage
from .versioning import make_standard_id


class IndexingPipeline(Protocol):
    def process(self, pdf_path: Path, standard: StandardEntry) -> IndexingResult: ...


class SafePipelineNotConfigured:
    def process(self, pdf_path: Path, standard: StandardEntry) -> IndexingResult:
        raise RuntimeError("safe per-standard indexing pipeline is not configured")


class StandardLibraryManager:
    def __init__(self, registry: JsonStandardRegistry, storage: StandardFileStorage, pipeline: IndexingPipeline):
        self.registry, self.storage, self.pipeline = registry, storage, pipeline

    def list_standards(self) -> list[StandardEntry]:
        return self.registry.list()

    def upload(self, filename: str, content: bytes, name: str, version: str, revision: str | None = None,
               metadata: dict[str, Any] | None = None) -> StandardEntry:
        standard_id = make_standard_id(name, version, revision)
        self.storage.validate(filename, content)
        entry = StandardEntry(standard_id=standard_id, name=name.strip(), version=version.strip(), revision=revision,
            status=StandardStatus.uploaded, source_file="", collection_id=None,
            created_at=datetime.now(timezone.utc), metadata=dict(metadata or {}))
        self.registry.add(entry)
        try:
            stored, digest = self.storage.save(standard_id, filename, content)
            entry = entry.model_copy(update={"source_file": str(stored), "status": StandardStatus.processing,
                                             "metadata": {**entry.metadata, "file_hash_sha256": digest}})
            self.registry.replace(entry)
            result = self.pipeline.process(stored, entry)
            entry = entry.model_copy(update={"status": StandardStatus.indexed, "collection_id": result.collection_id,
                                             "metadata": {**entry.metadata, **result.metadata}})
            return self.registry.replace(entry)
        except Exception as exc:
            failed = entry.model_copy(update={"status": StandardStatus.failed,
                                              "metadata": {**entry.metadata, "processing_error": str(exc)}})
            self.registry.replace(failed)
            raise

    def set_status(self, standard_id: str, status: StandardStatus) -> StandardEntry:
        return self.registry.change_status(standard_id, status)


def build_standard_manager(config: dict[str, Any], pipeline: IndexingPipeline | None = None) -> StandardLibraryManager:
    registry = JsonStandardRegistry(Path(config["standards_registry_path"]))
    storage = StandardFileStorage(Path(config["standards_upload_root"]), int(config["max_upload_bytes"]))
    return StandardLibraryManager(registry, storage, pipeline or SafePipelineNotConfigured())

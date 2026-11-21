from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .schemas import StandardEntry, StandardStatus


class StandardNotFound(KeyError):
    pass


class DuplicateStandard(ValueError):
    pass


class InvalidStatusTransition(ValueError):
    pass


class JsonStandardRegistry:
    """Small atomic JSON registry; it never deletes standards or indexed references."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])
        self._read()

    def list(self) -> list[StandardEntry]:
        with self._lock:
            return sorted(self._read(), key=lambda item: (item.name.casefold(), item.version, item.standard_id))

    def get(self, standard_id: str) -> StandardEntry:
        for entry in self.list():
            if entry.standard_id == standard_id:
                return entry
        raise StandardNotFound(standard_id)

    def add(self, entry: StandardEntry) -> StandardEntry:
        with self._lock:
            entries = self._read()
            if any(item.standard_id == entry.standard_id for item in entries):
                raise DuplicateStandard(f"standard version already exists: {entry.standard_id}")
            if any(item.name.casefold() == entry.name.casefold() and item.version == entry.version and item.revision == entry.revision for item in entries):
                raise DuplicateStandard(f"standard version already exists: {entry.name} {entry.version}")
            entries.append(entry)
            self._write(entries)
            return entry

    def replace(self, entry: StandardEntry) -> StandardEntry:
        with self._lock:
            entries = self._read()
            for index, current in enumerate(entries):
                if current.standard_id == entry.standard_id:
                    entries[index] = entry
                    self._write(entries)
                    return entry
            raise StandardNotFound(entry.standard_id)

    def change_status(self, standard_id: str, status: StandardStatus) -> StandardEntry:
        if status not in {StandardStatus.active, StandardStatus.inactive, StandardStatus.archived}:
            raise InvalidStatusTransition("public status change must be active, inactive, or archived")
        with self._lock:
            entries = self._read()
            target = next((item for item in entries if item.standard_id == standard_id), None)
            if target is None:
                raise StandardNotFound(standard_id)
            if status == StandardStatus.active and (not target.collection_id or target.status not in {StandardStatus.indexed, StandardStatus.inactive, StandardStatus.archived, StandardStatus.active}):
                raise InvalidStatusTransition("only an indexed standard can be activated")
            if status in {StandardStatus.inactive, StandardStatus.archived} and target.status not in {StandardStatus.indexed, StandardStatus.active, StandardStatus.inactive, StandardStatus.archived}:
                raise InvalidStatusTransition("standard has not been indexed")
            updated = []
            for entry in entries:
                if status == StandardStatus.active and entry.standard_id != standard_id and entry.name.casefold() == target.name.casefold() and entry.status == StandardStatus.active:
                    entry = entry.model_copy(update={"status": StandardStatus.inactive})
                if entry.standard_id == standard_id:
                    entry = entry.model_copy(update={"status": status})
                    target = entry
                updated.append(entry)
            self._write(updated)
            return target

    def active(self) -> list[StandardEntry]:
        return [entry for entry in self.list() if entry.status == StandardStatus.active]

    def _read(self) -> list[StandardEntry]:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("standards registry must contain a JSON array")
        return [StandardEntry.model_validate(item) for item in value]

    def _write(self, entries: list[StandardEntry]) -> None:
        payload = json.dumps([entry.model_dump(mode="json") for entry in entries], ensure_ascii=False, indent=2) + "\n"
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, self.path)

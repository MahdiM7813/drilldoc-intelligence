from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StandardStatus(StrEnum):
    uploaded = "uploaded"
    processing = "processing"
    indexed = "indexed"
    active = "active"
    inactive = "inactive"
    archived = "archived"
    failed = "failed"


class StandardEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    standard_id: str = Field(pattern=r"^[A-Z0-9][A-Z0-9._-]*$")
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    revision: str | None = None
    status: StandardStatus
    source_file: str
    collection_id: str | None = None
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class StandardStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: StandardStatus

    @field_validator("status")
    @classmethod
    def public_status_only(cls, value: StandardStatus) -> StandardStatus:
        if value not in {StandardStatus.active, StandardStatus.inactive, StandardStatus.archived}:
            raise ValueError("status endpoint only accepts active, inactive, or archived")
        return value


class IndexingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

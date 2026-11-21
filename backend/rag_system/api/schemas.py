from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collections: list[Literal["text", "tables"]] | None = None


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_version: Literal["1.0"] = "1.0"
    query: str
    top_k: int | None = Field(default=None, ge=1, le=100)
    filters: RetrievalFilters | None = None
    standard_id: str | None = Field(default=None, pattern=r"^[A-Z0-9][A-Z0-9._-]*$")


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    document: str
    text: str
    score: float
    metadata: dict[str, Any]
    citation: dict[str, Any]


class GroundedCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: str
    clause: str = ""
    page: str = ""
    chunk_id: str


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    citations: list[GroundedCitation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low", "insufficient"]


class RetrievalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contract_version: Literal["1.0"] = "1.0"
    query: str
    evidence: list[EvidenceItem]
    answer: GroundedAnswer | None = None


class LLMRoleHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["local_path", "llama_cpp", "transformers_qwen", "unknown"]
    selected_model: str | None = None
    available: bool


class LLMReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["available", "unavailable"]
    roles: dict[str, LLMRoleHealth]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok"] = "ok"
    api_ready: bool = True
    rag_ready: bool = False


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detail: str

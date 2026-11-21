from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnswerCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: str
    clause: str = ""
    page: str = ""
    chunk_id: str


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    answer: str = Field(min_length=1)
    standard: str
    citations: list[AnswerCitation]
    confidence: Literal["high", "medium", "low", "insufficient"]

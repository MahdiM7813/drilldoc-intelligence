from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    conversation_id: str | None = None
    query: str = Field(min_length=1)
    document_ids: list[str] | None = None
    language: str = Field(default="auto", pattern="^(auto|en|fa)$")

class Source(BaseModel):
    document_id: str
    filename: str
    page: int
    section: str | None = None
    chunk_id: str
    score: float = 0.0
    excerpt: str
    modality: str = "TEXT"
    record_id: str | None = None

class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    sources: list[Source]
    language: str = "en"
    answer_mode: str = "EXTRACTIVE_TEXT"
    insufficient_evidence: bool = False
    clarification_required: bool = False
    llama_used: bool = False

class Conversation(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

class Message(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime
    sources: list[Source] = []

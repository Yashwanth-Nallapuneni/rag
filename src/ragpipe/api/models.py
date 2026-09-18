"""Request/response models for the API layer.

`Answer`, `Citation`, `RetrievedChunk` and `Chunk` are already pydantic models
in `ragpipe.schemas` and are the pipeline's stable contract, so responses
reuse them directly rather than duplicating fields into parallel DTOs. Only
genuinely API-specific shapes (the request body, health/stats, error
envelope) get their own models here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..schemas import Answer, Chunk

MAX_QUESTION_CHARS = 2000


class QueryRequest(BaseModel):
    """Body for `POST /query`."""

    question: str
    k: int | None = Field(default=None, ge=1, le=50)
    mode: str | None = None
    filters: dict[str, Any] | None = None

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty or whitespace")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"question exceeds {MAX_QUESTION_CHARS} characters")
        return v


class QueryResponse(BaseModel):
    """The pipeline's `Answer`, plus a request id for tracing a demo run
    back to its log line."""

    request_id: str
    answer: Answer


class ChunkResponse(BaseModel):
    """Full chunk with provenance -- the citation click-through target.
    Deliberately the raw `Chunk`, not a snippet: a reader needs the exact
    passage the answer cited, not a truncated preview."""

    request_id: str
    chunk: Chunk


class ProviderHealth(BaseModel):
    name: str
    status: dict[str, Any]


class HealthResponse(BaseModel):
    request_id: str
    ready: bool
    answerer_loaded: bool
    error: str | None = None
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    store_count: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class StatsResponse(BaseModel):
    request_id: str
    store: dict[str, Any]
    documents: int | None = None
    chunks: int | None = None
    config_fingerprint: str


class IndexResponse(BaseModel):
    request_id: str
    service: str
    endpoints: dict[str, str]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str

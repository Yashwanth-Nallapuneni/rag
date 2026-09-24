"""Core data contracts for the RAG pipeline.

Every module in this package speaks in terms of these types. They are the
stable interface between ingestion, indexing, retrieval, generation and
evaluation, so changes here ripple everywhere -- treat them as an API.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    WEB = "web"
    TEXT = "text"


class SourceDocument(BaseModel):
    """A document as it entered the system, before chunking."""

    doc_id: str
    title: str
    source_type: SourceType
    source_path: str | None = None
    source_uri: str | None = None
    text: str = ""
    page_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    @staticmethod
    def make_doc_id(source: str) -> str:
        return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


class Chunk(BaseModel):
    """A retrievable unit of text plus everything needed to cite it."""

    chunk_id: str
    doc_id: str
    doc_title: str
    chunk_index: int
    text: str
    token_count: int = 0

    # Provenance: at least one of these must let a reader find the passage.
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    char_start: int | None = None
    char_end: int | None = None

    source_type: SourceType = SourceType.TEXT
    source_path: str | None = None
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_chunk_id(doc_id: str, chunk_index: int) -> str:
        return f"{doc_id}::{chunk_index:05d}"

    @property
    def section_label(self) -> str:
        return " > ".join(self.section_path) if self.section_path else ""

    def locator(self) -> str:
        """Human-readable pointer used in citations."""
        parts: list[str] = [self.doc_title]
        if self.section_path:
            parts.append(self.section_label)
        if self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                parts.append(f"pp. {self.page_start}-{self.page_end}")
            else:
                parts.append(f"p. {self.page_start}")
        return " | ".join(parts)

    def to_store_metadata(self) -> dict[str, Any]:
        """Flatten to primitives -- Chroma rejects nested/None values."""
        md: dict[str, Any] = {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "source_type": self.source_type.value,
            "section_path": "\x1f".join(self.section_path),
        }
        for key in (
            "page_start",
            "page_end",
            "char_start",
            "char_end",
            "source_path",
            "source_uri",
        ):
            value = getattr(self, key)
            if value is not None:
                md[key] = value
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                md[f"x_{key}"] = value
        return md

    @classmethod
    def from_store(cls, chunk_id: str, text: str, md: dict[str, Any]) -> Chunk:
        raw_sections = md.get("section_path") or ""
        return cls(
            chunk_id=chunk_id,
            doc_id=md.get("doc_id", ""),
            doc_title=md.get("doc_title", ""),
            chunk_index=int(md.get("chunk_index", 0)),
            text=text,
            token_count=int(md.get("token_count", 0)),
            page_start=md.get("page_start"),
            page_end=md.get("page_end"),
            section_path=[s for s in raw_sections.split("\x1f") if s],
            char_start=md.get("char_start"),
            char_end=md.get("char_end"),
            source_type=SourceType(md.get("source_type", "text")),
            source_path=md.get("source_path"),
            source_uri=md.get("source_uri"),
            metadata={
                k[2:]: v for k, v in md.items() if k.startswith("x_")
            },
        )


RetrieverName = Literal["dense", "sparse", "hybrid", "rerank"]


class RetrievedChunk(BaseModel):
    """A chunk with the scores that got it here. Scores are kept separate so
    the eval harness can attribute wins to a specific retrieval stage."""

    chunk: Chunk
    score: float = 0.0
    rank: int = 0
    retriever: RetrieverName = "dense"
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text


class Citation(BaseModel):
    """One [n] marker in an answer, resolved back to its source passage."""

    marker: int
    chunk_id: str
    doc_id: str
    doc_title: str
    locator: str
    page_start: int | None = None
    section_path: list[str] = Field(default_factory=list)
    source_uri: str | None = None
    quote: str | None = None


class ClaimVerdict(BaseModel):
    """Result of checking one sentence of the answer against its cited chunk."""

    claim: str
    supported: bool
    cited_chunk_ids: list[str] = Field(default_factory=list)
    support_score: float = 0.0
    reason: str = ""


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    REFUSED_NO_CONTEXT = "refused_no_context"
    REFUSED_LOW_SUPPORT = "refused_low_support"
    REFUSED_BY_MODEL = "refused_by_model"
    ERROR = "error"


class Answer(BaseModel):
    """The full, auditable result of a query."""

    question: str
    text: str
    status: AnswerStatus = AnswerStatus.ANSWERED
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[RetrievedChunk] = Field(default_factory=list)
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    refusal_reason: str | None = None
    prompt_version: str | None = None
    model: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    config_fingerprint: str | None = None

    @property
    def refused(self) -> bool:
        return self.status != AnswerStatus.ANSWERED

    @property
    def context_texts(self) -> list[str]:
        return [c.text for c in self.contexts]


class QAPair(BaseModel):
    """One row of the golden evaluation dataset."""

    id: str
    question: str
    ground_truth: str
    doc_id: str | None = None
    expected_chunk_ids: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    category: str = "factual"
    unanswerable: bool = False
    notes: str = ""

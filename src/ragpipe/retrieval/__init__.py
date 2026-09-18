"""Retriever selection.

`get_retriever` is the single seam the generation layer talks to, so adding
hybrid retrieval and reranking changes what is constructed here without
touching the answering pipeline at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..schemas import RetrievedChunk

if TYPE_CHECKING:
    from ..config import Settings
    from ..index.base import VectorStore


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(
        self, query: str, k: int | None = None, where: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]: ...


def get_retriever(settings: "Settings", store: "VectorStore") -> Retriever:
    mode = settings.retrieval.mode
    if mode == "dense":
        from .dense import DenseRetriever

        return DenseRetriever(settings, store)
    if mode in ("sparse", "hybrid"):
        # Phase 4 registers BM25 and hybrid fusion here. Until then, fall
        # back to dense so a hybrid config is still runnable.
        try:
            from .hybrid import HybridRetriever

            return HybridRetriever(settings, store)
        except ImportError:
            from .dense import DenseRetriever

            return DenseRetriever(settings, store)
    raise ValueError(f"unknown retrieval mode: {mode}")

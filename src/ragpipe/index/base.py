"""Vector store protocol.

Retrieval depends on this interface, not on ChromaDB. The store's job is
narrow on purpose: it holds vectors plus enough metadata to rebuild a `Chunk`,
and returns similarity-ordered hits. Everything else -- fusion, reranking,
citation checks -- happens above it, so swapping Chroma for Weaviate touches
exactly one file.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..schemas import Chunk


class VectorStoreError(RuntimeError):
    pass


@runtime_checkable
class VectorStore(Protocol):
    name: str

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Add or replace chunks. Returns the number written."""
        ...

    def query(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Nearest neighbours as (chunk, similarity) with similarity in
        [0, 1], higher is better -- NOT a raw distance. Normalising here
        keeps score fusion from silently mixing distances and similarities."""
        ...

    def get(self, chunk_ids: list[str]) -> list[Chunk]:
        """Fetch chunks by id. Used for citation click-through."""
        ...

    def count(self) -> int: ...

    def document_ids(self) -> list[str]:
        """Distinct source documents in the collection. Needed so /stats can
        report corpus coverage without re-reading the chunk file."""
        ...

    def reset(self) -> None:
        """Drop everything. Used when the config fingerprint changes."""
        ...

    def stats(self) -> dict[str, Any]: ...

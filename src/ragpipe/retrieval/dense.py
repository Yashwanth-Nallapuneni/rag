"""Dense (embedding) retrieval.

Thin by design: embed the query, ask the store for neighbours, wrap them as
`RetrievedChunk` with the dense score and rank recorded separately from the
final score. Keeping per-stage scores distinct is what later lets the eval
harness attribute a win to fusion or to reranking rather than to "retrieval"
as an undifferentiated blob.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..index.base import VectorStore
from ..providers import get_embedder
from ..schemas import RetrievedChunk


class DenseRetriever:
    name = "dense"

    def __init__(self, settings: Settings, store: VectorStore):
        self.settings = settings
        self.store = store
        self.embedder = get_embedder(settings)

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        k = k or self.settings.retrieval.candidate_k
        if not query.strip() or self.store.count() == 0:
            return []
        vector = self.embedder.embed_query(query)
        hits = self.store.query(vector, k=k, where=where)
        return [
            RetrievedChunk(
                chunk=chunk,
                score=score,
                dense_score=score,
                dense_rank=rank,
                rank=rank,
                retriever="dense",
            )
            for rank, (chunk, score) in enumerate(hits, start=1)
        ]

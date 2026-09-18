"""Hybrid retrieval: dense + BM25, fused, then cross-encoder reranked.

The pipeline is deliberately wide-then-narrow. Each retriever proposes
`candidate_k` chunks (default 30), fusion merges them, and the reranker cuts
that shortlist to the `top_k` that reach the model. Retrieving only `top_k`
per retriever and fusing would defeat the reranker: it can only reorder what
it is given, so a chunk that dense retrieval ranked 12th can never be promoted
to first if it was never in the shortlist.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..index.base import VectorStore
from ..logging_utils import get_logger
from ..schemas import RetrievedChunk
from .fusion import fuse

log = get_logger(__name__)


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        *,
        dense=None,
        sparse=None,
        reranker=None,
    ):
        self.settings = settings
        self.cfg = settings.retrieval
        self.mode = self.cfg.mode

        if dense is None and self.mode in ("dense", "hybrid"):
            from .dense import DenseRetriever

            dense = DenseRetriever(settings, store)
        if sparse is None and self.mode in ("sparse", "hybrid"):
            # Built from the STORE, not from chunks.jsonl. If the sparse index
            # were built independently the two could drift, and a BM25 hit for
            # a chunk absent from the store would produce a citation whose
            # click-through 404s. An empty store therefore yields no sparse
            # retriever at all, so the pipeline refuses instead of answering
            # from a half-built index.
            from .bm25 import BM25Retriever

            indexed = store.iter_chunks()
            if indexed:
                sparse = BM25Retriever.build_or_load(settings, indexed)
            else:
                log.warning("vector store is empty; sparse retrieval disabled")
        self.dense = dense
        self.sparse = sparse

        self.rerank_stage = reranker
        if self.rerank_stage is None and settings.rerank.enabled:
            from .rerank import RerankStage

            self.rerank_stage = RerankStage(settings)

    def _first_pass(
        self, query: str, candidate_k: int, where: dict[str, Any] | None
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        dense_hits: list[RetrievedChunk] = []
        sparse_hits: list[RetrievedChunk] = []
        if self.dense is not None:
            dense_hits = self.dense.retrieve(query, k=candidate_k, where=where)
        if self.sparse is not None:
            sparse_hits = self.sparse.retrieve(query, k=candidate_k, where=where)
        return dense_hits, sparse_hits

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        final_k = k or self.cfg.top_k
        candidate_k = max(self.cfg.candidate_k, final_k)

        dense_hits, sparse_hits = self._first_pass(query, candidate_k, where)
        if not dense_hits and not sparse_hits:
            return []

        fused = fuse(
            dense_hits,
            sparse_hits,
            method=self.cfg.fusion,
            rrf_k=self.cfg.rrf_k,
            dense_weight=self.cfg.dense_weight,
            sparse_weight=self.cfg.sparse_weight,
        )

        # min_score applies to the fusion score only. Rerank scores are raw
        # logits and routinely negative, so reusing this threshold after
        # reranking would discard everything; use rerank.score_threshold for
        # that instead.
        if self.cfg.min_score > 0:
            fused = [rc for rc in fused if rc.score >= self.cfg.min_score] or fused[:1]

        if self.rerank_stage is None:
            return fused[:final_k]

        reranked = self.rerank_stage.rerank(query, fused[:candidate_k], top_n=final_k)
        log.debug(
            "hybrid: dense=%d sparse=%d fused=%d reranked=%d",
            len(dense_hits),
            len(sparse_hits),
            len(fused),
            len(reranked),
        )
        return reranked

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "fusion": self.cfg.fusion,
            "candidate_k": self.cfg.candidate_k,
            "top_k": self.cfg.top_k,
            "dense": getattr(self.dense, "name", None),
            "sparse": getattr(self.sparse, "name", None),
            "reranker": self.rerank_stage.model if self.rerank_stage else None,
        }

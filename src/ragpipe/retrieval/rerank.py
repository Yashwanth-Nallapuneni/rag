"""Cross-encoder reranking.

First-pass retrieval scores the query and a chunk *independently* -- two
vectors compared after the fact, or term statistics -- so it can only ever
approximate relevance. A cross-encoder reads the query and the chunk together
in one forward pass and can judge whether the passage actually answers the
question. That is why it reliably improves precision at small k, and why it is
worth the latency on a 20-30 candidate shortlist even though it would be far
too slow to run over the whole corpus.

The scores it returns are raw logits: unbounded, frequently negative, and not
comparable across models. They are recorded in `rerank_score` and used for
ordering, never treated as probabilities or blended into a weighted sum.
"""

from __future__ import annotations

from ..config import Settings
from ..logging_utils import get_logger
from ..providers import get_reranker
from ..schemas import RetrievedChunk

log = get_logger(__name__)


class RerankStage:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.rerank
        self.reranker = get_reranker(settings)

    @property
    def model(self) -> str:
        return self.reranker.model

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int | None = None
    ) -> list[RetrievedChunk]:
        limit = top_n or self.cfg.top_n
        if not candidates or not query.strip():
            return candidates[:limit]

        documents = [rc.chunk.text for rc in candidates]
        results = self.reranker.rerank(query, documents, top_n=None)

        ordered: list[RetrievedChunk] = []
        for new_rank, result in enumerate(results, start=1):
            if not 0 <= result.index < len(candidates):
                # A provider returning an out-of-range index would silently
                # attach one chunk's citation to another chunk's text.
                log.error(
                    "reranker returned out-of-range index %d for %d candidates",
                    result.index,
                    len(candidates),
                )
                continue
            rc = candidates[result.index]
            rc.rerank_score = float(result.score)
            # Ordering score becomes the rerank score; the pre-rerank fusion
            # score stays in `fusion_score` so the stages remain separable.
            rc.score = float(result.score)
            rc.rank = new_rank
            rc.retriever = "rerank"
            ordered.append(rc)

        if self.cfg.score_threshold is not None:
            kept = [rc for rc in ordered if (rc.rerank_score or 0.0) >= self.cfg.score_threshold]
            # Never return nothing on a threshold alone: an empty result makes
            # the pipeline refuse, which is a much stronger claim than "these
            # passages scored low".
            ordered = kept or ordered[:1]

        return ordered[:limit]

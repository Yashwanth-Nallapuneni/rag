"""Combining dense and sparse result lists.

Two strategies, with genuinely different failure modes:

**Reciprocal rank fusion** ignores scores entirely and combines ranks:
`score = sum_r 1/(k + rank_r(d))`. Because it never compares a cosine
similarity against a BM25 score, it is immune to the calibration problem that
makes weighted sums fragile -- BM25 scores are unbounded and corpus-dependent,
so a "0.5/0.5" weighted blend of a raw BM25 score and a cosine similarity is
not actually a half-and-half blend of anything. RRF is the default here for
that reason.

**Weighted sum** can outperform RRF when you genuinely want to lean on one
retriever, and it preserves score magnitude (RRF throws away the difference
between a near-perfect match and a mediocre one at the same rank). It requires
both inputs normalised to [0,1], which is why the sparse retriever normalises
and why that normalisation is documented where it happens.

Either way the per-retriever scores and ranks are carried onto the merged
result, so the evaluation harness can still attribute an outcome to a stage.
"""

from __future__ import annotations

from ..schemas import RetrievedChunk


def _merge_provenance(
    target: RetrievedChunk, source: RetrievedChunk
) -> RetrievedChunk:
    """Fold one retriever's view of a chunk into the merged record."""
    if source.dense_score is not None:
        target.dense_score = source.dense_score
        target.dense_rank = source.dense_rank
    if source.sparse_score is not None:
        target.sparse_score = source.sparse_score
        target.sparse_rank = source.sparse_rank
    return target


def _collect(
    result_lists: list[list[RetrievedChunk]],
) -> dict[str, RetrievedChunk]:
    """One merged record per chunk id, carrying every retriever's scores."""
    merged: dict[str, RetrievedChunk] = {}
    for results in result_lists:
        for rc in results:
            existing = merged.get(rc.chunk_id)
            if existing is None:
                merged[rc.chunk_id] = _merge_provenance(
                    rc.model_copy(deep=True), rc
                )
            else:
                _merge_provenance(existing, rc)
    return merged


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Rank-based fusion. `k` damps the influence of top ranks; 60 is the
    value from the original RRF paper and a sane default."""
    if weights is not None and len(weights) != len(result_lists):
        raise ValueError("weights must match the number of result lists")

    merged = _collect(result_lists)
    scores: dict[str, float] = {cid: 0.0 for cid in merged}

    for i, results in enumerate(result_lists):
        weight = 1.0 if weights is None else weights[i]
        for rank, rc in enumerate(results, start=1):
            scores[rc.chunk_id] += weight / (k + rank)

    return _finalize(merged, scores)


def weighted_sum(
    result_lists: list[list[RetrievedChunk]],
    weights: list[float],
) -> list[RetrievedChunk]:
    """Score-based fusion over normalised scores.

    A chunk missing from one retriever's candidate list contributes 0 for that
    retriever. That is an approximation, not a truth: absence from the top-N
    is evidence of low relevance, but not a measured zero. It is the reason
    weighted fusion is more sensitive to `candidate_k` than RRF is.
    """
    if len(weights) != len(result_lists):
        raise ValueError("weights must match the number of result lists")

    merged = _collect(result_lists)
    scores: dict[str, float] = {cid: 0.0 for cid in merged}
    for weight, results in zip(weights, result_lists, strict=True):
        for rc in results:
            scores[rc.chunk_id] += weight * rc.score

    total = sum(weights) or 1.0
    scores = {cid: value / total for cid, value in scores.items()}
    return _finalize(merged, scores)


def _finalize(
    merged: dict[str, RetrievedChunk],
    scores: dict[str, float],
) -> list[RetrievedChunk]:
    # Tie-break on chunk_id so fusion is deterministic; CI can only gate on
    # quality if the same query yields the same ordering every run.
    ordered = sorted(merged.values(), key=lambda rc: (-scores[rc.chunk_id], rc.chunk_id))
    for rank, rc in enumerate(ordered, start=1):
        rc.fusion_score = scores[rc.chunk_id]
        rc.score = scores[rc.chunk_id]
        rc.rank = rank
        rc.retriever = "hybrid"
    return ordered


def fuse(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    *,
    method: str = "rrf",
    rrf_k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[RetrievedChunk]:
    """Fuse exactly the two lists this system produces.

    A single non-empty list is returned as-is rather than fused: RRF over one
    list is a no-op that would still rewrite every score into 1/(k+rank),
    throwing away the retriever's own calibration for nothing.
    """
    lists = [lst for lst in (dense, sparse) if lst]
    if not lists:
        return []
    if len(lists) == 1:
        return lists[0]

    if method == "weighted_sum":
        return weighted_sum([dense, sparse], [dense_weight, sparse_weight])
    if method == "rrf":
        return reciprocal_rank_fusion(
            [dense, sparse], k=rrf_k, weights=[dense_weight, sparse_weight]
        )
    raise ValueError(f"unknown fusion method: {method}")

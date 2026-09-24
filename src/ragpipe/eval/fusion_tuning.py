"""Tune hybrid fusion weights against the human-verified golden set.

Retrieval-only: dense embeddings (local model) + BM25 + a local cross-encoder
reranker. No LLM calls, so this costs $0 to run regardless of how large the
grid is.

Anti-overfitting discipline (see docs/STATE.md §6 and the project owner's
standing rule against unmeasured claims):

- A deterministic stratified split (by `category`, fixed seed) separates a
  TUNE set (~60%) from a held-out TEST set (~40%). Every grid config is
  scored on TUNE only; the winner is re-scored on TEST exactly once.
- The reported "does this beat the 0.5/0.5 baseline" verdict is decided on
  TEST, with a paired bootstrap 95% CI. If that CI includes 0, the answer is
  "keep 0.5/0.5" -- the data does not support a change.
- Cross-encoder reranking is genuinely slow, so first-pass dense/BM25 hits
  are retrieved once per (query, candidate_k) and cross-encoder scores are
  cached once per (query, chunk_id) pair; every fusion config in the grid
  re-fuses and reranks from those cached pieces instead of re-querying.

This module has no CLI of its own -- see scripts/tune_fusion.py.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ..retrieval.fusion import fuse
from ..retrieval.rerank import RerankStage
from ..schemas import QAPair, RetrievedChunk

# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------


def stratified_split(
    pairs: list[QAPair], tune_ratio: float = 0.6, seed: int = 1337
) -> tuple[list[QAPair], list[QAPair]]:
    """Deterministic split, stratified by `category` (unanswerable pairs get
    their own pseudo-category so they're spread across both sides too).

    Same seed + same input pairs -> same split, always. Sorting each
    category's ids before shuffling means the split does not depend on the
    order pairs happen to appear in the file.
    """
    by_cat: dict[str, list[QAPair]] = defaultdict(list)
    for qa in pairs:
        key = "unanswerable" if qa.unanswerable else qa.category
        by_cat[key].append(qa)

    tune: list[QAPair] = []
    test: list[QAPair] = []
    for key in sorted(by_cat):
        group = sorted(by_cat[key], key=lambda qa: qa.id)
        rng = random.Random(f"{seed}:{key}")
        rng.shuffle(group)
        n_tune = round(len(group) * tune_ratio)
        tune.extend(group[:n_tune])
        test.extend(group[n_tune:])
    return tune, test


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def recall_at_k(ranked_chunk_ids: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    return 1.0 if set(ranked_chunk_ids[:k]) & expected else 0.0


def mrr_at_k(ranked_chunk_ids: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    for i, cid in enumerate(ranked_chunk_ids[:k], start=1):
        if cid in expected:
            return 1.0 / i
    return 0.0


def doc_hit_at_k(ranked_doc_ids: list[str], expected_doc_id: str | None, k: int) -> float:
    if not expected_doc_id:
        return 0.0
    return 1.0 if expected_doc_id in ranked_doc_ids[:k] else 0.0


@dataclass
class QueryMetrics:
    qa_id: str
    category: str
    recall_1: float
    recall_5: float
    mrr_10: float
    doc_hit_5: float


def score_ranking(qa: QAPair, ranked: list[RetrievedChunk]) -> QueryMetrics:
    chunk_ids = [rc.chunk_id for rc in ranked]
    doc_ids = [rc.chunk.doc_id for rc in ranked]
    expected = set(qa.expected_chunk_ids)
    return QueryMetrics(
        qa_id=qa.id,
        category=qa.category,
        recall_1=recall_at_k(chunk_ids, expected, 1),
        recall_5=recall_at_k(chunk_ids, expected, 5),
        mrr_10=mrr_at_k(chunk_ids, expected, 10),
        doc_hit_5=doc_hit_at_k(doc_ids, qa.doc_id, 5),
    )


def aggregate(metrics: list[QueryMetrics]) -> dict[str, float]:
    if not metrics:
        return {"n": 0, "recall@1": 0.0, "recall@5": 0.0, "mrr@10": 0.0, "doc_hit@5": 0.0}
    n = len(metrics)
    return {
        "n": n,
        "recall@1": sum(m.recall_1 for m in metrics) / n,
        "recall@5": sum(m.recall_5 for m in metrics) / n,
        "mrr@10": sum(m.mrr_10 for m in metrics) / n,
        "doc_hit@5": sum(m.doc_hit_5 for m in metrics) / n,
    }


def per_category(metrics: list[QueryMetrics]) -> dict[str, dict[str, float]]:
    by_cat: dict[str, list[QueryMetrics]] = defaultdict(list)
    for m in metrics:
        by_cat[m.category].append(m)
    return {cat: aggregate(ms) for cat, ms in sorted(by_cat.items())}


# --------------------------------------------------------------------------
# bootstrap CI
# --------------------------------------------------------------------------


def bootstrap_ci(
    paired_diffs: list[float],
    n_resamples: int = 1000,
    seed: int = 1337,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap 95% CI over the per-query paired difference
    (chosen_config_metric - baseline_metric). Deterministic given the seed."""
    n = len(paired_diffs)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample = [paired_diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    hi_idx = min(hi_idx, n_resamples - 1)
    mean = sum(paired_diffs) / n
    return {"mean": mean, "lo": means[lo_idx], "hi": means[hi_idx], "n": n}


def decide(ci: dict[str, float]) -> bool:
    """True -> recommend the change: the improvement's CI lies entirely above
    0. A CI entirely BELOW 0 means the candidate is significantly worse --
    that must keep the baseline, not recommend the change."""
    return ci["lo"] > 0.0


# --------------------------------------------------------------------------
# first-pass + rerank caching
# --------------------------------------------------------------------------


@dataclass
class QueryCandidates:
    """Cached first-pass results for one query, reused across every fusion
    config in the grid so the grid sweep does not re-hit the store/BM25."""

    qa: QAPair
    dense: list[RetrievedChunk]
    sparse: list[RetrievedChunk]


def collect_candidates(
    pairs: list[QAPair],
    dense_retriever: Any,
    sparse_retriever: Any,
    candidate_k: int,
) -> list[QueryCandidates]:
    out = []
    for qa in pairs:
        dense_hits = dense_retriever.retrieve(qa.question, k=candidate_k) if dense_retriever else []
        sparse_hits = sparse_retriever.retrieve(qa.question, k=candidate_k) if sparse_retriever else []
        out.append(QueryCandidates(qa=qa, dense=dense_hits, sparse=sparse_hits))
    return out


class RerankCache:
    """Caches cross-encoder scores per (query, chunk_id) pair.

    A grid config only changes which chunks make it into the fused shortlist
    and their order -- not the score the cross-encoder assigns a given
    (query, chunk) pair. So the expensive rerank call for a given query is
    only ever needed once per **candidate set that appears**; in practice
    fused shortlists for the same query overlap heavily across configs, so
    this caches per (qa_id, tuple of candidate chunk_ids).
    """

    def __init__(self, rerank_stage: RerankStage | None):
        self.rerank_stage = rerank_stage
        self._cache: dict[tuple[str, tuple[str, ...]], list[RetrievedChunk]] = {}
        self.calls = 0

    def rerank(self, qa: QAPair, fused: list[RetrievedChunk], candidate_k: int, top_n: int) -> list[RetrievedChunk]:
        shortlist = fused[:candidate_k]
        if self.rerank_stage is None:
            return shortlist[:top_n]
        key = (qa.id, tuple(rc.chunk_id for rc in shortlist))
        cached = self._cache.get(key)
        if cached is None:
            self.calls += 1
            cached = self.rerank_stage.rerank(qa.question, shortlist, top_n=None)
            self._cache[key] = cached
        return cached[:top_n]


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------


@dataclass
class FusionConfig:
    label: str
    method: str  # "rrf" | "weighted_sum"
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "method": self.method,
            "rrf_k": self.rrf_k,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
        }


def build_grid() -> list[FusionConfig]:
    configs: list[FusionConfig] = []
    for rrf_k in (20, 60, 100):
        for dw, sw in ((0.5, 0.5), (0.7, 0.3), (0.3, 0.7), (0.9, 0.1), (0.1, 0.9)):
            configs.append(
                FusionConfig(
                    label=f"rrf_k{rrf_k}_d{dw}_s{sw}",
                    method="rrf",
                    rrf_k=rrf_k,
                    dense_weight=dw,
                    sparse_weight=sw,
                )
            )
    for sw in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        configs.append(
            FusionConfig(
                label=f"weighted_s{sw:.1f}",
                method="weighted_sum",
                dense_weight=round(1.0 - sw, 1),
                sparse_weight=sw,
            )
        )
    return configs


BASELINE = FusionConfig(label="baseline_rrf_0.5_0.5", method="rrf", rrf_k=60, dense_weight=0.5, sparse_weight=0.5)


def run_config(
    candidates: list[QueryCandidates],
    config: FusionConfig,
    *,
    candidate_k: int,
    top_k: int,
    rerank_cache: RerankCache | None,
) -> list[QueryMetrics]:
    """Fuse cached first-pass results under one config, optionally rerank
    (via the shared cache), and score the resulting ranking."""
    out = []
    for qc in candidates:
        fused = fuse(
            qc.dense,
            qc.sparse,
            method=config.method,
            rrf_k=config.rrf_k,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
        )
        if rerank_cache is not None:
            ranked = rerank_cache.rerank(qc.qa, fused, candidate_k, top_k)
        else:
            ranked = fused[:top_k]
        out.append(score_ranking(qc.qa, ranked))
    return out


def sweep(
    candidates: list[QueryCandidates],
    grid: list[FusionConfig],
    *,
    candidate_k: int,
    top_k: int,
    rerank_cache: RerankCache | None,
    primary_metric: str = "recall@5",
) -> list[dict[str, Any]]:
    """Run every grid config on `candidates` (a TUNE-only call, by
    convention of the caller) and return rows sorted best-first on
    `primary_metric`."""
    rows = []
    for config in grid:
        metrics = run_config(
            candidates, config, candidate_k=candidate_k, top_k=top_k, rerank_cache=rerank_cache
        )
        agg = aggregate(metrics)
        rows.append({"config": config.as_dict(), **agg})
    rows.sort(key=lambda r: r[primary_metric], reverse=True)
    return rows


def pick_best(rows: list[dict[str, Any]], primary_metric: str = "recall@5") -> dict[str, Any]:
    return max(rows, key=lambda r: r[primary_metric])

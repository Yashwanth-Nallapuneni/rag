from __future__ import annotations

import pytest

from ragpipe.retrieval.fusion import fuse, reciprocal_rank_fusion, weighted_sum
from ragpipe.schemas import Chunk, RetrievedChunk


def _rc(i: int, score: float, kind: str, rank: int) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"c{i}", doc_id="d", doc_title="T", chunk_index=i, text=f"text {i}"
    )
    extra = (
        {"dense_score": score, "dense_rank": rank}
        if kind == "dense"
        else {"sparse_score": score * 10, "sparse_rank": rank}
    )
    return RetrievedChunk(
        chunk=chunk, score=score, rank=rank, retriever=kind, **extra
    )


DENSE = [_rc(1, 0.9, "dense", 1), _rc(2, 0.8, "dense", 2), _rc(3, 0.7, "dense", 3)]
SPARSE = [_rc(3, 0.95, "sparse", 1), _rc(4, 0.6, "sparse", 2), _rc(1, 0.5, "sparse", 3)]


# --- fusion ---------------------------------------------------------------


def test_fusion_unions_both_result_lists():
    out = fuse(DENSE, SPARSE)
    assert {rc.chunk_id for rc in out} == {"c1", "c2", "c3", "c4"}


def test_fusion_preserves_per_retriever_provenance():
    """Without this the eval harness cannot attribute an outcome to a stage."""
    out = {rc.chunk_id: rc for rc in fuse(DENSE, SPARSE)}
    both = out["c1"]
    assert both.dense_score == 0.9 and both.dense_rank == 1
    assert both.sparse_score == 5.0 and both.sparse_rank == 3
    dense_only = out["c2"]
    assert dense_only.dense_score is not None and dense_only.sparse_score is None


def test_fusion_sets_rank_and_fusion_score():
    out = fuse(DENSE, SPARSE)
    assert [rc.rank for rc in out] == list(range(1, len(out) + 1))
    assert all(rc.fusion_score is not None for rc in out)
    assert all(rc.retriever == "hybrid" for rc in out)


def test_fusion_is_sorted_descending():
    out = fuse(DENSE, SPARSE)
    assert [rc.score for rc in out] == sorted((rc.score for rc in out), reverse=True)


def test_fusion_is_deterministic():
    """CI can only gate on quality if ordering is stable across runs."""
    a = [rc.chunk_id for rc in fuse(DENSE, SPARSE)]
    b = [rc.chunk_id for rc in fuse(DENSE, SPARSE)]
    assert a == b


def test_rrf_ignores_score_magnitude():
    """The whole point of RRF: an uncalibrated BM25 score cannot dominate."""
    inflated = [
        _rc(3, 999.0, "sparse", 1),
        _rc(4, 998.0, "sparse", 2),
        _rc(1, 997.0, "sparse", 3),
    ]
    normal = [rc.chunk_id for rc in reciprocal_rank_fusion([DENSE, SPARSE])]
    huge = [rc.chunk_id for rc in reciprocal_rank_fusion([DENSE, inflated])]
    assert normal == huge


def test_weighted_sum_responds_to_weights():
    dense_heavy = weighted_sum([DENSE, SPARSE], [0.95, 0.05])
    sparse_heavy = weighted_sum([DENSE, SPARSE], [0.05, 0.95])
    assert dense_heavy[0].chunk_id == "c1"
    assert sparse_heavy[0].chunk_id == "c3"


def test_weighted_sum_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        weighted_sum([DENSE, SPARSE], [1.0])


def test_single_list_passes_through_unfused():
    """RRF over one list would overwrite its calibration for nothing."""
    out = fuse(DENSE, [])
    assert [rc.score for rc in out] == [0.9, 0.8, 0.7]


def test_empty_inputs():
    assert fuse([], []) == []


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        fuse(DENSE, SPARSE, method="magic")

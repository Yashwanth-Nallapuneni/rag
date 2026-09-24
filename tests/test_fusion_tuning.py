from __future__ import annotations

from ragpipe.eval.fusion_tuning import (
    QueryMetrics,
    aggregate,
    bootstrap_ci,
    decide,
    doc_hit_at_k,
    mrr_at_k,
    per_category,
    recall_at_k,
    stratified_split,
)
from ragpipe.schemas import QAPair


def _qa(id_, category, unanswerable=False, doc_id="d1", expected=("d1::00000",)):
    return QAPair(
        id=id_,
        question=f"q-{id_}",
        ground_truth="gt",
        doc_id=doc_id,
        expected_chunk_ids=list(expected) if not unanswerable else [],
        category=category,
        unanswerable=unanswerable,
    )


# --------------------------------------------------------------------------
# split determinism / stratification
# --------------------------------------------------------------------------


def test_split_is_deterministic():
    pairs = [_qa(f"a{i}", "factual") for i in range(10)] + [_qa(f"b{i}", "numeric") for i in range(10)]
    tune1, test1 = stratified_split(pairs, tune_ratio=0.6, seed=42)
    tune2, test2 = stratified_split(pairs, tune_ratio=0.6, seed=42)
    assert [qa.id for qa in tune1] == [qa.id for qa in tune2]
    assert [qa.id for qa in test1] == [qa.id for qa in test2]


def test_split_different_seed_can_differ():
    pairs = [_qa(f"a{i}", "factual") for i in range(20)]
    tune1, _ = stratified_split(pairs, seed=1)
    tune2, _ = stratified_split(pairs, seed=2)
    assert [qa.id for qa in tune1] != [qa.id for qa in tune2]


def test_split_is_stratified_by_category():
    pairs = [_qa(f"a{i}", "factual") for i in range(10)] + [_qa(f"b{i}", "numeric") for i in range(10)]
    tune, test = stratified_split(pairs, tune_ratio=0.6, seed=7)
    tune_cats = {qa.category for qa in tune}
    test_cats = {qa.category for qa in test}
    assert tune_cats == {"factual", "numeric"}
    assert test_cats == {"factual", "numeric"}
    # each category split ~60/40
    for cat in ("factual", "numeric"):
        n_tune_cat = sum(1 for qa in tune if qa.category == cat)
        n_test_cat = sum(1 for qa in test if qa.category == cat)
        assert n_tune_cat == 6
        assert n_test_cat == 4


def test_split_no_overlap_and_covers_all():
    pairs = [_qa(f"a{i}", "factual") for i in range(7)] + [_qa(f"b{i}", "numeric") for i in range(5)]
    tune, test = stratified_split(pairs, seed=3)
    tune_ids = {qa.id for qa in tune}
    test_ids = {qa.id for qa in test}
    assert not (tune_ids & test_ids)
    assert tune_ids | test_ids == {qa.id for qa in pairs}


def test_unanswerable_pairs_get_own_stratum():
    pairs = [_qa(f"a{i}", "factual") for i in range(6)] + [
        _qa(f"u{i}", "factual", unanswerable=True) for i in range(6)
    ]
    tune, test = stratified_split(pairs, seed=5)
    n_tune_unans = sum(1 for qa in tune if qa.unanswerable)
    n_test_unans = sum(1 for qa in test if qa.unanswerable)
    assert n_tune_unans == 4
    assert n_test_unans == 2


# --------------------------------------------------------------------------
# metrics math
# --------------------------------------------------------------------------


def test_recall_at_k_hit_and_miss():
    ranked = ["c3", "c1", "c2"]
    assert recall_at_k(ranked, {"c1"}, k=1) == 0.0
    assert recall_at_k(ranked, {"c1"}, k=2) == 1.0
    assert recall_at_k(ranked, {"c9"}, k=3) == 0.0
    assert recall_at_k(ranked, set(), k=3) == 0.0


def test_mrr_at_k():
    ranked = ["c3", "c1", "c2"]
    assert mrr_at_k(ranked, {"c1"}, k=10) == 0.5
    assert mrr_at_k(ranked, {"c3"}, k=10) == 1.0
    assert mrr_at_k(ranked, {"cX"}, k=10) == 0.0
    assert mrr_at_k(ranked, {"c2"}, k=1) == 0.0  # outside k


def test_doc_hit_at_k():
    doc_ids = ["dA", "dB", "dC"]
    assert doc_hit_at_k(doc_ids, "dB", k=5) == 1.0
    assert doc_hit_at_k(doc_ids, "dZ", k=5) == 0.0
    assert doc_hit_at_k(doc_ids, None, k=5) == 0.0


def test_aggregate_averages_and_handles_empty():
    metrics = [
        QueryMetrics("q1", "factual", recall_1=1.0, recall_5=1.0, mrr_10=1.0, doc_hit_5=1.0),
        QueryMetrics("q2", "factual", recall_1=0.0, recall_5=1.0, mrr_10=0.5, doc_hit_5=0.0),
    ]
    agg = aggregate(metrics)
    assert agg["n"] == 2
    assert agg["recall@1"] == 0.5
    assert agg["recall@5"] == 1.0
    assert agg["mrr@10"] == 0.75
    assert aggregate([])["n"] == 0


def test_per_category_groups_correctly():
    metrics = [
        QueryMetrics("q1", "factual", 1.0, 1.0, 1.0, 1.0),
        QueryMetrics("q2", "numeric", 0.0, 0.0, 0.0, 0.0),
    ]
    grouped = per_category(metrics)
    assert set(grouped) == {"factual", "numeric"}
    assert grouped["factual"]["recall@1"] == 1.0
    assert grouped["numeric"]["recall@1"] == 0.0


# --------------------------------------------------------------------------
# bootstrap CI determinism + decision rule
# --------------------------------------------------------------------------


def test_bootstrap_ci_is_deterministic():
    diffs = [0.1, -0.05, 0.2, 0.0, 0.15, -0.1, 0.3]
    ci1 = bootstrap_ci(diffs, n_resamples=500, seed=99)
    ci2 = bootstrap_ci(diffs, n_resamples=500, seed=99)
    assert ci1 == ci2


def test_bootstrap_ci_empty_input():
    ci = bootstrap_ci([], seed=1)
    assert ci["n"] == 0
    assert ci["mean"] == 0.0


def test_decide_true_when_all_positive():
    diffs = [0.2, 0.3, 0.25, 0.18, 0.22] * 5
    ci = bootstrap_ci(diffs, n_resamples=500, seed=1)
    assert decide(ci) is True


def test_decide_false_when_straddles_zero():
    diffs = [0.5, -0.5, 0.1, -0.1, 0.0, 0.05, -0.05]
    ci = bootstrap_ci(diffs, n_resamples=500, seed=1)
    assert decide(ci) is False


def test_decide_false_on_no_data():
    ci = bootstrap_ci([], seed=1)
    assert decide(ci) is False


def test_decide_rejects_a_significantly_worse_candidate():
    # CI entirely below zero = the candidate is significantly WORSE.
    assert decide({"mean": -0.1, "lo": -0.2, "hi": -0.05, "n": 60}) is False

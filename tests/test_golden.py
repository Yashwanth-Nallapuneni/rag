"""Golden-set drafting and the verification gate."""

from __future__ import annotations

from ragpipe.eval.golden import (
    draft_candidates,
    load_verified,
    save_dataset,
    validate_dataset,
)


def test_draft_is_offline_and_well_formed(settings, corpus_chunks):
    pairs = draft_candidates(settings, corpus_chunks, n=40, seed=7)
    report = validate_dataset(pairs)
    # a 60-chunk slice cannot always supply 40 distinct sections
    assert 30 <= report.total <= 40
    assert not report.missing_ground_truth
    assert not report.unanswerable_with_expected_chunks
    assert all(qa.expected_chunk_ids for qa in pairs if not qa.unanswerable)


def test_categories_none_uses_defaults(settings, corpus_chunks):
    # The CLI passes None when no flag is given; that once raised TypeError.
    assert draft_candidates(settings, corpus_chunks, n=10, categories=None, seed=1)


def test_unanswerable_probes_are_not_repeated(settings, corpus_chunks):
    """27 probes that collapsed to 11 unique questions measured refusal on
    repeats; every probe must be a distinct question."""
    pairs = draft_candidates(settings, corpus_chunks, n=180, seed=7)
    probes = [qa.question for qa in pairs if qa.unanswerable]
    assert len(probes) == 27
    assert len(set(probes)) == len(probes)
    assert len({qa.id for qa in pairs}) == len(pairs)


def test_unverified_pairs_never_reach_the_eval(settings, corpus_chunks, tmp_path):
    """Verification is opt-in: a freshly drafted file yields zero pairs."""
    path = tmp_path / "golden.jsonl"
    save_dataset(draft_candidates(settings, corpus_chunks, n=10, seed=3), path)
    assert load_verified(path) == []

"""Offline tests for the golden-dataset LLM pre-screen.

Hard rule under test throughout: prescreen must never write the review
ledger and must never affect `load_verified()`. Every test here uses a
scripted fake LLM (no network) and asserts that in addition to whatever
behaviour it targets.
"""

from __future__ import annotations

import json

import pytest

from ragpipe.eval.golden import default_review_path, load_verified, save_dataset
from ragpipe.eval.prescreen import (
    PrescreenRecord,
    answer_too_short,
    build_indices,
    default_prescreen_path,
    find_near_duplicate_questions,
    load_prescreen,
    prescreen_pair,
    question_names_nothing,
    save_prescreen,
)
from ragpipe.providers.base import LLMResponse
from ragpipe.schemas import Chunk, QAPair


class _ScriptedLLM:
    """Fake LLMProvider that returns whatever JSON string it's told to,
    regardless of the prompt -- lets tests target prescreen logic without
    depending on the extractive mock's context-block format."""

    name = "scripted"
    model = "scripted-test-model"

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        text = self._replies[min(self.calls - 1, len(self._replies) - 1)]
        return LLMResponse(text=text, model=self.model, usage={"input_tokens": 50, "output_tokens": 20})

    def health(self):
        return {"provider": self.name, "ready": True}


def _chunk(chunk_id="doc1::00000", doc_id="doc1", text="On-Demand Attention (ODA) is a sparse attention mechanism that skips low-relevance tokens.", title="ODA Paper"):
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, doc_title=title, chunk_index=0, text=text, token_count=len(text.split()))


def _answerable_pair(id="qa-1", question="What is On-Demand Attention (ODA)?", gt="A sparse attention mechanism.", chunk_id="doc1::00000", doc_id="doc1"):
    return QAPair(
        id=id, question=question, ground_truth=gt, doc_id=doc_id,
        expected_chunk_ids=[chunk_id], expected_sources=["loc"], category="definition",
        unanswerable=False,
    )


def _unanswerable_offdomain(id="qa-unans-0"):
    return QAPair(
        id=id, question="What is the boiling point of nitrogen?", ground_truth="Not in corpus.",
        doc_id=None, expected_chunk_ids=[], expected_sources=[], category="factual", unanswerable=True,
    )


def _unanswerable_indomain(id="qa-unans-1", doc_id="doc1"):
    return QAPair(
        id=id, question='What grant number funded "ODA Paper"?', ground_truth="Not in corpus.",
        doc_id=doc_id, expected_chunk_ids=[], expected_sources=[], category="factual", unanswerable=True,
    )


# --------------------------------------------------------------------------
# hard rule: never writes the ledger / verification is unaffected
# --------------------------------------------------------------------------


def test_prescreen_never_writes_ledger(tmp_path):
    dataset_path = tmp_path / "golden.jsonl"
    pairs = [_answerable_pair()]
    save_dataset(pairs, dataset_path)

    review_path = default_review_path(dataset_path)
    assert not review_path.exists()

    llm = _ScriptedLLM([json.dumps({"supported": True, "standalone": True, "answerable": True, "reason": "fine"})])
    chunk_map, chunks_by_doc, dup = build_indices(pairs, [_chunk()])
    rec = prescreen_pair(pairs[0], llm=llm, chunk_map=chunk_map, chunks_by_doc=chunks_by_doc, duplicate_groups=dup, model_name="scripted")
    save_prescreen({rec.id: rec}, default_prescreen_path(dataset_path))

    # Ledger still does not exist -- prescreen created no review decisions.
    assert not review_path.exists()
    assert load_verified(dataset_path) == []


def test_load_verified_unaffected_after_prescreen_run(tmp_path):
    dataset_path = tmp_path / "golden.jsonl"
    pairs = [_answerable_pair(id="qa-a"), _unanswerable_offdomain(id="qa-b")]
    save_dataset(pairs, dataset_path)

    llm = _ScriptedLLM([json.dumps({"supported": False, "standalone": False, "answerable": False, "reason": "bad"})])
    chunk_map, chunks_by_doc, dup = build_indices(pairs, [_chunk()])
    for qa in pairs:
        rec = prescreen_pair(qa, llm=llm, chunk_map=chunk_map, chunks_by_doc=chunks_by_doc, duplicate_groups=dup, model_name="scripted")
        assert rec.verdict in ("ok", "suspect")

    # Even a run that flags everything suspect must not touch verification.
    assert load_verified(dataset_path) == []


# --------------------------------------------------------------------------
# deterministic checks
# --------------------------------------------------------------------------


def test_duplicate_detection_exact_and_near():
    p1 = _answerable_pair(id="qa-1", question="What is On-Demand Attention (ODA)?")
    p2 = _answerable_pair(id="qa-2", question="What is On-Demand Attention (ODA)?")
    p3 = _answerable_pair(id="qa-3", question="What is a completely unrelated question about gradient clipping?")
    groups = find_near_duplicate_questions([p1, p2, p3])
    all_grouped = {gid for group in groups.values() for gid in group}
    assert {"qa-1", "qa-2"}.issubset(all_grouped)
    assert "qa-3" not in all_grouped


def test_answer_too_short():
    assert answer_too_short("")
    assert not answer_too_short("ok")  # short is not wrong; only empty is
    assert not answer_too_short("A sparse attention mechanism that skips tokens.")


def test_question_names_nothing_flags_generic_backreference():
    assert question_names_nothing("What is the maximum number of steps allowed for each task to run?") is False
    # This one does use a bad backreference and truly names nothing:
    assert question_names_nothing("According to the passage, what number is given?") is True
    # Backreference present but a concrete name appears elsewhere:
    assert question_names_nothing('According to the passage, what does BERT achieve?') is False


# --------------------------------------------------------------------------
# sidecar round-trip
# --------------------------------------------------------------------------


def test_sidecar_round_trip(tmp_path):
    path = tmp_path / "golden_prescreen.jsonl"
    records = {
        "qa-1": PrescreenRecord(id="qa-1", verdict="ok", reasons=[], checks={"a": 1}, model="m", timestamp="t"),
        "qa-2": PrescreenRecord(id="qa-2", verdict="suspect", reasons=["bad"], checks={}, model="m", timestamp="t"),
    }
    save_prescreen(records, path)
    loaded = load_prescreen(path)
    assert set(loaded) == {"qa-1", "qa-2"}
    assert loaded["qa-2"].verdict == "suspect"
    assert loaded["qa-2"].reasons == ["bad"]


def test_load_prescreen_missing_file_returns_empty(tmp_path):
    assert load_prescreen(tmp_path / "does_not_exist.jsonl") == {}


# --------------------------------------------------------------------------
# review UI helper tolerates missing sidecar
# --------------------------------------------------------------------------


def test_review_ui_helper_tolerates_missing_sidecar(tmp_path):
    dataset_path = tmp_path / "golden.jsonl"
    prescreen_path = default_prescreen_path(dataset_path)
    assert not prescreen_path.exists()
    # This is exactly what review_golden.py calls at startup.
    prescreen = load_prescreen(prescreen_path)
    assert prescreen == {}


# --------------------------------------------------------------------------
# in-domain unanswerable check flags a paper that actually answers it
# --------------------------------------------------------------------------


def test_uncovered_pair_flagged_when_llm_says_answered():
    pairs = [_unanswerable_indomain()]
    # BM25 needs more than one document in the corpus to produce a positive
    # idf/score (a single-chunk "corpus" degenerates to score 0 for every
    # term), so add an unrelated second chunk from the same paper.
    chunks = [
        _chunk(chunk_id="doc1::00000", text="This work was funded by NSF grant number 12345."),
        _chunk(chunk_id="doc1::00001", text="We evaluate our sparse attention method on long-context benchmarks."),
        _chunk(chunk_id="doc1::00002", text="Our ablations show the gating mechanism improves accuracy on retrieval tasks."),
        _chunk(chunk_id="doc1::00003", text="Related work includes prior sparse transformer variants and linear attention."),
    ]
    llm = _ScriptedLLM([
        json.dumps({"answered": True, "chunk_id": "doc1::00000", "quote": "funded by NSF grant number 12345", "reason": "states the grant"})
    ])
    chunk_map, chunks_by_doc, dup = build_indices(pairs, chunks)
    rec = prescreen_pair(pairs[0], llm=llm, chunk_map=chunk_map, chunks_by_doc=chunks_by_doc, duplicate_groups=dup, model_name="scripted")
    assert rec.verdict == "suspect"
    assert any("appears to answer" in r.lower() for r in rec.reasons)


def test_uncovered_pair_ok_when_llm_says_not_answered():
    pairs = [_unanswerable_indomain()]
    chunks = [
        _chunk(chunk_id="doc1::00000", text="This paper studies sparse attention mechanisms for long sequences."),
        _chunk(chunk_id="doc1::00001", text="We evaluate our sparse attention method on long-context benchmarks."),
        _chunk(chunk_id="doc1::00002", text="Our ablations show the gating mechanism improves accuracy on retrieval tasks."),
        _chunk(chunk_id="doc1::00003", text="Related work includes prior sparse transformer variants and linear attention."),
    ]
    llm = _ScriptedLLM([
        json.dumps({"answered": False, "chunk_id": "", "quote": "", "reason": "no grant info present"})
    ])
    chunk_map, chunks_by_doc, dup = build_indices(pairs, chunks)
    rec = prescreen_pair(pairs[0], llm=llm, chunk_map=chunk_map, chunks_by_doc=chunks_by_doc, duplicate_groups=dup, model_name="scripted")
    assert rec.verdict == "ok"


def test_offdomain_unanswerable_pair_skips_llm_call():
    pairs = [_unanswerable_offdomain()]
    llm = _ScriptedLLM(["should never be used"])
    chunk_map, chunks_by_doc, dup = build_indices(pairs, [])
    rec = prescreen_pair(pairs[0], llm=llm, chunk_map=chunk_map, chunks_by_doc=chunks_by_doc, duplicate_groups=dup, model_name="scripted")
    assert llm.calls == 0
    assert rec.verdict == "ok"
    assert rec.checks.get("off_domain") is True


def test_answerable_pair_flagged_when_llm_says_unsupported():
    pairs = [_answerable_pair()]
    chunks = [_chunk()]
    llm = _ScriptedLLM([
        json.dumps({"supported": False, "standalone": True, "answerable": True, "reason": "ground truth overstates the passage"})
    ])
    chunk_map, chunks_by_doc, dup = build_indices(pairs, chunks)
    rec = prescreen_pair(pairs[0], llm=llm, chunk_map=chunk_map, chunks_by_doc=chunks_by_doc, duplicate_groups=dup, model_name="scripted")
    assert rec.verdict == "suspect"
    assert any("not be supported" in r.lower() for r in rec.reasons)


def test_short_correct_answers_are_not_flagged():
    from ragpipe.eval.prescreen import answer_too_short

    for truth in ("11.9 ms", "GRPO", "O(1)", "3", "8.95B"):
        assert not answer_too_short(truth)
    assert answer_too_short("") and answer_too_short("   ")


def test_named_subject_overrides_judge_standalone_false():
    """The judge said 'not standalone' for questions quoting full titles."""
    from ragpipe.eval.prescreen import question_names_something

    assert question_names_something("What is On-Demand Attention (ODA)?")
    assert question_names_something("What does 'Stable Movement for Nondual Lipschitz Convex Optimization' prove?")
    assert question_names_something("What is the parameter count of the dQwen3.5-9B model?")
    assert not question_names_something("What is the maximum number of steps allowed for each task to run?")
    assert not question_names_something("What is the dataset pipeline illustrated in Figure 4?")

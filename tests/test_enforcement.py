from __future__ import annotations

import pytest

from ragpipe.generation.answerer import Answerer
from ragpipe.generation.context import BLOCK_HEADER_RE, render_context
from ragpipe.generation.verify import LexicalVerifier, get_verifier
from ragpipe.schemas import AnswerStatus, Chunk, RetrievedChunk


# --- the header-regex regression -----------------------------------------


def _rc(i: int, title: str, section: list[str], text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"doc::{i:05d}",
            doc_id="doc",
            doc_title=title,
            chunk_index=i,
            text=text,
            page_start=i,
            section_path=section,
        ),
        score=1.0 / i,
        rank=i,
    )


def test_parentheses_in_a_heading_do_not_break_block_parsing():
    """Regression, and a bad one: the header regex required a balanced
    "(...)" locator, so a heading containing parentheses -- "Results > ...
    (RE3 & RE5)", straight from the corpus -- failed to match. The block was
    then never recognised, its text merged into the PREVIOUS passage, and
    every sentence in it was attributed to the wrong chunk. Citations were
    silently wrong, which is worse than no citation at all."""
    chunks = [
        _rc(1, "Paper A", ["Abstract"], "First passage body."),
        _rc(
            2,
            "Paper A",
            ["Results > versioning and auditing ensure traceability (RE3 & RE5)"],
            "Second passage body about FedAvg aggregation.",
        ),
        _rc(3, "Paper B (extended)", ["Methods (part 2)"], "Third passage body."),
    ]
    rendered = render_context(chunks, max_tokens=4000)
    assert [int(m.group(1)) for m in BLOCK_HEADER_RE.finditer(rendered.text)] == [1, 2, 3]
    assert len(rendered.marker_to_chunk) == 3
    assert rendered.marker_to_chunk[2].chunk_id == "doc::00002"


def test_mock_llm_sees_the_same_blocks_the_renderer_wrote(settings):
    """The renderer and the offline model must agree on the format, or the
    model cites markers that mean something else."""
    from ragpipe.providers.llm.mock import _BLOCK_RE

    chunks = [
        _rc(1, "Paper A", ["Intro (background)"], "Alpha body text here."),
        _rc(2, "Paper A", ["Results (RE3 & RE5)"], "Beta body text here."),
    ]
    rendered = render_context(chunks, max_tokens=4000)
    assert [int(m.group(1)) for m in _BLOCK_RE.finditer(rendered.text)] == [1, 2]


def test_marker_resolves_to_the_chunk_that_contains_the_sentence(settings):
    """End-to-end version of the same bug: ask the offline model to answer
    from context whose headings contain parentheses, and check every cited
    marker points at a chunk that actually contains the cited sentence."""
    from ragpipe.generation.citations import cited_chunks, split_claims
    from ragpipe.providers import LLMRequest, get_llm

    chunks = [
        _rc(1, "P", ["Intro (background)"], "Widgets are small mechanical components."),
        _rc(2, "P", ["Method (RE3 & RE5)"], "FedAvg is used for federated aggregation."),
        _rc(3, "P", ["Results (final)"], "Latency decreased by eleven percent."),
    ]
    rendered = render_context(chunks, max_tokens=4000)
    _, user = __import__(
        "ragpipe.prompts", fromlist=["load_prompt"]
    ).load_prompt("answer", "v2", str(settings.prompts.path)).render(
        context=rendered.text, question="What is used for federated aggregation?"
    )
    text = get_llm(settings).complete(
        LLMRequest(system="s", user=user, task="answer")
    ).text
    claims = split_claims(text)
    assert claims
    for claim in claims:
        for rc in cited_chunks(claim, rendered):
            key = claim.text.split()[0].lower()
            assert key in rc.chunk.text.lower(), (
                f"claim {claim.text!r} cites {rc.chunk_id}, which does not contain it"
            )


# --- enforcement in the pipeline -----------------------------------------


@pytest.fixture
def enforcing(offline_store):
    settings, store = offline_store
    settings.citation.enforce = True
    settings.citation.verifier = "lexical"
    settings.citation.min_relevance_score = None  # mock reranker has no logits
    return Answerer(settings, store, verifier=get_verifier(settings)), settings


def test_answer_records_the_support_ratio(enforcing, corpus_chunks):
    answerer, _ = enforcing
    a = answerer.answer(" ".join(corpus_chunks[0].text.split()[:14]))
    if not a.refused:
        assert a.usage["claims_checked"] >= 1
        assert a.usage["claims_supported"] <= a.usage["claims_checked"]


def test_refuses_when_support_is_below_threshold(offline_store, monkeypatch):
    """The trust property: an answer whose claims are not traceable to the
    retrieved passages must be refused, not shipped."""
    settings, store = offline_store
    settings.citation.enforce = True
    settings.citation.min_supported_ratio = 0.8
    settings.citation.min_relevance_score = None

    class AlwaysUnsupported:
        def verify(self, answer_text, rendered, question):
            from ragpipe.schemas import ClaimVerdict

            return [
                ClaimVerdict(claim="c", supported=False, support_score=0.0, reason="no")
            ], 0.0

    answerer = Answerer(settings, store, verifier=AlwaysUnsupported())
    a = answerer.answer("model evaluation benchmark results")
    assert a.status == AnswerStatus.REFUSED_LOW_SUPPORT
    assert "supported" in a.refusal_reason
    assert a.claim_verdicts, "the failing verdicts must be surfaced, not hidden"
    assert not a.citations


def test_enforcement_can_be_disabled(offline_store):
    settings, store = offline_store
    settings.citation.enforce = False
    settings.citation.min_relevance_score = None
    answerer = Answerer(settings, store, verifier=get_verifier(settings))
    assert answerer.verifier is None


def test_relevance_gate_refuses_before_generation(offline_store, monkeypatch):
    """Grounding and relevance are different properties: a faithful quotation
    of an irrelevant passage passes every citation check and still fails the
    user. The gate must fire before the model is called at all."""
    settings, store = offline_store
    settings.citation.enforce = True
    settings.citation.min_relevance_score = 0.0

    answerer = Answerer(settings, store, verifier=None)
    calls = {"n": 0}
    real = answerer.llm.complete

    def counting(request):
        calls["n"] += 1
        return real(request)

    monkeypatch.setattr(answerer.llm, "complete", counting)

    # Force every candidate to look irrelevant to the cross-encoder.
    original = answerer.retriever.retrieve

    def low_scoring(query, k=None, where=None):
        hits = original(query, k=k, where=where)
        for h in hits:
            h.rerank_score = -50.0
        return hits

    monkeypatch.setattr(answerer.retriever, "retrieve", low_scoring)

    a = answerer.answer("evaluation of models")
    assert a.status == AnswerStatus.REFUSED_NO_CONTEXT
    assert "relevance" in a.refusal_reason
    assert calls["n"] == 0, "the model was called for a question we already refused"


def test_relevance_gate_allows_a_relevant_question(offline_store, monkeypatch):
    settings, store = offline_store
    settings.citation.enforce = True
    settings.citation.min_relevance_score = -7.0
    answerer = Answerer(settings, store, verifier=None)

    original = answerer.retriever.retrieve

    def high_scoring(query, k=None, where=None):
        hits = original(query, k=k, where=where)
        for h in hits:
            h.rerank_score = 5.0
        return hits

    monkeypatch.setattr(answerer.retriever, "retrieve", high_scoring)
    a = answerer.answer("evaluation of models on the benchmark")
    assert a.status != AnswerStatus.REFUSED_NO_CONTEXT


def test_gate_is_skipped_when_no_rerank_scores_exist(offline_store):
    """Without a reranker there are no logits to threshold; the gate must not
    refuse everything by treating a missing score as zero."""
    settings, store = offline_store
    settings.citation.min_relevance_score = -7.0
    settings.rerank.enabled = False
    answerer = Answerer(settings, store, verifier=None)
    a = answerer.answer("evaluation of models on the benchmark")
    assert a.status != AnswerStatus.REFUSED_NO_CONTEXT

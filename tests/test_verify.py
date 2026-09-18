from __future__ import annotations

import pytest

from ragpipe.generation.context import render_context
from ragpipe.generation.verify import (
    HybridVerifier,
    LexicalVerifier,
    LLMVerifier,
    _numbers,
    get_verifier,
    judge_lexically,
)
from ragpipe.schemas import Chunk, RetrievedChunk

PASSAGE = (
    "The Transformer uses multi-head self-attention instead of recurrence. "
    "This allows significantly more parallelization during training. "
    "BLEU improves by 3.1 points on the English-German task."
)


def _rendered(text: str = PASSAGE):
    rc = RetrievedChunk(
        chunk=Chunk(
            chunk_id="doc::00001",
            doc_id="doc",
            doc_title="Attention Is All You Need",
            chunk_index=1,
            text=text,
            page_start=3,
        ),
        score=0.9,
        rank=1,
    )
    return render_context([rc], max_tokens=2000)


# --- the three lexical checks --------------------------------------------


def test_verbatim_claim_is_supported():
    j = judge_lexically(
        "The Transformer uses multi-head self-attention instead of recurrence.", PASSAGE
    )
    assert j.coverage == 1.0 and not j.certain_fail


def test_paraphrase_is_supported():
    j = judge_lexically("Training is more parallelizable.", PASSAGE)
    assert j.coverage >= 0.45 and not j.certain_fail


def test_unrelated_claim_has_no_coverage():
    j = judge_lexically("Paris is the capital of France.", PASSAGE)
    assert j.coverage == 0.0


def test_wrong_number_is_caught_despite_perfect_wording():
    """The highest-value cheap check: word overlap is 100% and the claim is
    still false."""
    j = judge_lexically("BLEU improves by 2.5 points.", PASSAGE)
    assert j.coverage > 0.8
    assert not j.numbers_ok
    assert "2.5" in j.missing_numbers
    assert j.certain_fail


def test_negation_flip_is_caught():
    """Coverage alone cannot see this: the sentences share almost every word."""
    j = judge_lexically(
        "The Transformer does not use self-attention instead of recurrence.", PASSAGE
    )
    assert not j.negation_ok
    assert j.certain_fail


def test_enumeration_markers_are_not_treated_as_figures():
    """Regression: "(6) The FedAvg technique is used" was rejected for citing
    a figure 6 that the passage did not contain. The 6 is list structure."""
    assert _numbers("(6) The FedAvg technique is used for aggregation.") == set()
    assert _numbers("3. Results follow.") == set()


def test_cross_references_are_not_treated_as_figures():
    assert _numbers("See Figure 3 and Table 2 in Section 4.1.") == set()


def test_real_quantities_are_still_extracted():
    assert _numbers("Accuracy reaches 91% with 2.5 BLEU gain") == {"91", "2.5"}


def test_percent_and_decimal_normalisation():
    assert _numbers("40%") == _numbers("40") == {"40"}
    assert _numbers("2.50") == _numbers("2.5")


# --- LexicalVerifier ------------------------------------------------------


@pytest.fixture
def lexical(settings):
    settings.citation.verifier = "lexical"
    return LexicalVerifier(settings)


def test_supported_answer_scores_one(lexical):
    verdicts, ratio = lexical.verify(
        "The Transformer uses multi-head self-attention instead of recurrence. [S1]",
        _rendered(),
        "q",
    )
    assert ratio == 1.0 and verdicts[0].supported


def test_uncited_sentence_is_unsupported_when_required(settings):
    settings.citation.require_citation_per_sentence = True
    verdicts, ratio = LexicalVerifier(settings).verify(
        "The Transformer uses self-attention.", _rendered(), "q"
    )
    assert ratio == 0.0
    assert "no citation" in verdicts[0].reason


def test_uncited_sentence_is_scored_when_not_required(settings):
    """Lenient mode: a grounded but uncited sentence is a citation-quality
    problem, not a hallucination."""
    settings.citation.require_citation_per_sentence = False
    verdicts, ratio = LexicalVerifier(settings).verify(
        "The Transformer uses multi-head self-attention instead of recurrence.",
        _rendered(),
        "q",
    )
    assert ratio == 1.0 and verdicts[0].supported


def test_ratio_is_fraction_of_supported_claims(lexical):
    answer = (
        "The Transformer uses multi-head self-attention instead of recurrence. [S1] "
        "Paris is the capital of France. [S1]"
    )
    verdicts, ratio = lexical.verify(answer, _rendered(), "q")
    assert len(verdicts) == 2
    assert ratio == 0.5


def test_verdict_records_which_chunk_it_checked(lexical):
    verdicts, _ = lexical.verify("Training is more parallelizable. [S1]", _rendered(), "q")
    assert verdicts[0].cited_chunk_ids == ["doc::00001"]


def test_empty_answer(lexical):
    assert lexical.verify("", _rendered(), "q") == ([], 0.0)


# --- LLM and hybrid -------------------------------------------------------


def test_llm_verifier_parses_judge_json(settings):
    settings.citation.verifier = "llm"
    verdicts, ratio = LLMVerifier(settings).verify(
        "The Transformer uses multi-head self-attention instead of recurrence. [S1]",
        _rendered(),
        "q",
    )
    assert len(verdicts) == 1
    assert 0.0 <= ratio <= 1.0


@pytest.mark.parametrize(
    "payload", ["not json at all", "{broken json", '{"verdicts": "wrong type"}', ""]
)
def test_unparseable_judge_output_does_not_pass_claims(payload):
    """A judge whose output cannot be read must not silently approve
    everything -- that turns enforcement off exactly when it matters."""
    assert LLMVerifier._parse(payload, 2) == {}


def test_judge_verdicts_outside_range_are_dropped():
    parsed = LLMVerifier._parse(
        '{"verdicts":[{"claim_id":99,"supported":true},{"claim_id":1,"supported":true}]}', 2
    )
    assert set(parsed) == {1}


def test_hybrid_does_not_escalate_hard_numeric_failures(settings):
    """A model must not be able to talk itself into accepting 2.5 against a
    passage that says 3.1."""
    settings.citation.verifier = "hybrid"
    verdicts, ratio = HybridVerifier(settings).verify(
        "BLEU improves by 2.5 points. [S1]", _rendered(), "q"
    )
    assert not verdicts[0].supported
    assert ratio == 0.0


def test_hybrid_keeps_lexical_result_when_judge_fails(settings, monkeypatch):
    settings.citation.verifier = "hybrid"
    verifier = HybridVerifier(settings)

    class Boom:
        def judge(self, *a, **k):
            raise RuntimeError("provider down")

    monkeypatch.setattr(HybridVerifier, "llm", property(lambda self: Boom()))
    verdicts, ratio = verifier.verify(
        "The Transformer uses multi-head self-attention instead of recurrence. [S1]",
        _rendered(),
        "q",
    )
    assert verdicts and ratio == 1.0


def test_factory_honours_config(settings):
    for kind, cls in (("lexical", LexicalVerifier), ("hybrid", HybridVerifier)):
        settings.citation.verifier = kind
        assert isinstance(get_verifier(settings), cls)


def test_factory_returns_none_when_enforcement_is_off(settings):
    settings.citation.enforce = False
    assert get_verifier(settings) is None

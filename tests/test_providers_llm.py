from __future__ import annotations

import json

import pytest

from ragpipe.config import load_settings
from ragpipe.providers import (
    LLMRequest,
    MissingCredentialsError,
    ProviderError,
    get_llm,
)

CONTEXT_PROMPT = """Context:
[1] (Attention Is All You Need | 3 Model Architecture | p. 3)
The Transformer uses multi-head self-attention instead of recurrence. This allows significantly more parallelization during training.

[2] (BERT | p. 2)
BERT is pretrained with a masked language modeling objective on unlabeled text.

Question: What does the Transformer use instead of recurrence?"""


@pytest.fixture
def llm(settings):
    return get_llm(settings)


def test_mock_answers_from_context_with_citation(llm):
    out = llm.complete(LLMRequest(system="s", user=CONTEXT_PROMPT, task="answer")).text
    assert "self-attention" in out
    assert "[1]" in out


def test_mock_does_not_cite_irrelevant_passages(llm):
    """A padded answer that cites unrelated chunks would inflate faithfulness."""
    out = llm.complete(LLMRequest(system="s", user=CONTEXT_PROMPT, task="answer")).text
    assert "[2]" not in out
    assert "BERT" not in out


def test_mock_refuses_when_context_is_irrelevant(llm):
    prompt = CONTEXT_PROMPT.replace(
        "What does the Transformer use instead of recurrence?",
        "What is the capital city of Mongolia?",
    )
    out = llm.complete(LLMRequest(system="s", user=prompt, task="answer")).text
    assert out.strip() == "INSUFFICIENT_CONTEXT"


def test_mock_refuses_with_no_context(llm):
    out = llm.complete(LLMRequest(system="s", user="Question: anything?", task="answer")).text
    assert out.strip() == "INSUFFICIENT_CONTEXT"


def test_mock_is_deterministic(llm):
    req = LLMRequest(system="s", user=CONTEXT_PROMPT, task="answer")
    assert llm.complete(req).text == llm.complete(req).text


def test_mock_claim_check_separates_supported_from_unsupported(llm):
    prompt = """Context:
[1] (X | p. 1)
The Transformer uses multi-head self-attention instead of recurrence.

CLAIM 1: The Transformer uses multi-head self-attention instead of recurrence.
CLAIM 2: The Transformer was trained on 400 TPUs for six months."""
    verdicts = json.loads(
        llm.complete(LLMRequest(system="s", user=prompt, task="claim_check")).text
    )["verdicts"]
    assert len(verdicts) == 2
    assert verdicts[0]["supported"] is True
    # Regression guard: the claim text must not leak into the context the
    # checker reads, or every claim scores as supported.
    assert verdicts[1]["supported"] is False


def test_response_carries_usage_and_latency(llm):
    r = llm.complete(LLMRequest(system="s", user=CONTEXT_PROMPT, task="answer"))
    assert r.usage["input_tokens"] > 0 and r.usage["output_tokens"] > 0
    assert r.latency_ms >= 0
    assert r.model


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_hosted_providers_fail_cleanly_without_keys(provider):
    """A missing key must be an actionable error, never a raw SDK traceback."""
    s = load_settings(overrides={"llm": {"provider": provider}})
    with pytest.raises((MissingCredentialsError, ProviderError)) as exc:
        get_llm(s)
    assert provider.upper() in str(exc.value).upper()


def test_unknown_provider_rejected():
    s = load_settings()
    s.llm.provider = "nope"  # bypass validation to test the registry guard
    with pytest.raises(ProviderError):
        get_llm(s)

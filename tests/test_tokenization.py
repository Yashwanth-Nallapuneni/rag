from __future__ import annotations

from ragpipe.tokenization import count_tokens, truncate_to_tokens


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_scales_with_length():
    short = count_tokens("The Transformer uses self-attention.")
    long = count_tokens("The Transformer uses self-attention. " * 10)
    assert 0 < short < long


def test_truncate_respects_budget():
    text = "The Transformer architecture relies on self-attention. " * 50
    assert count_tokens(truncate_to_tokens(text, 40)) <= 40
    assert truncate_to_tokens("short text", 1000) == "short text"

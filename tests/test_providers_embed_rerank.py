from __future__ import annotations

import math

import pytest

from ragpipe.config import load_settings
from ragpipe.providers import (
    MissingCredentialsError,
    ProviderError,
    get_embedder,
    get_reranker,
)


def _cos(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


@pytest.fixture
def embedder(settings):
    return get_embedder(settings)


@pytest.fixture
def reranker(settings):
    return get_reranker(settings)


def test_mock_embedder_is_deterministic(embedder):
    assert embedder.embed_query("self-attention") == embedder.embed_query("self-attention")


def test_mock_embedder_respects_dimension(embedder):
    vec = embedder.embed_query("x")
    assert len(vec) == 384
    assert all(isinstance(v, float) for v in vec), "Chroma rejects numpy scalars"


def test_mock_embedder_is_semantically_ordered(embedder):
    """A mock that scores everything alike makes retrieval tests meaningless."""
    q = embedder.embed_query("the cat sat on the mat")
    related, unrelated = embedder.embed_documents(
        ["a cat sat on a mat", "quantum chromodynamics lattice gauge theory"]
    )
    assert _cos(q, related) > _cos(q, unrelated) + 0.1


def test_embed_documents_matches_input_length(embedder):
    assert len(embedder.embed_documents(["a", "b", "c"])) == 3
    assert embedder.embed_documents([]) == []


def test_reranker_returns_original_indices(reranker):
    """index must point into the ORIGINAL list or citations get scrambled."""
    docs = ["quantum gauge theory", "a cat sat on a mat", "dogs bark loudly"]
    out = reranker.rerank("cat on mat", docs)
    assert out[0].index == 1
    assert {r.index for r in out} <= set(range(len(docs)))
    assert len({r.index for r in out}) == len(out), "duplicate indices"


def test_reranker_sorted_descending(reranker):
    out = reranker.rerank("cat", ["cat", "dog", "cat sat"])
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)


def test_reranker_top_n(reranker):
    out = reranker.rerank("cat", ["cat", "dog", "cat sat", "bird"], top_n=2)
    assert len(out) == 2


def test_reranker_handles_empty_documents(reranker):
    assert reranker.rerank("anything", []) == []


def test_embedder_instances_are_cached(settings):
    assert get_embedder(settings) is get_embedder(settings)


def test_openai_embedder_fails_cleanly_without_key():
    s = load_settings(overrides={"embeddings": {"provider": "openai", "dimension": 1536}})
    with pytest.raises((MissingCredentialsError, ProviderError)):
        get_embedder(s)


@pytest.mark.slow
def test_local_embedder_real_weights():
    """Downloads ~130MB on first run. Marked slow; excluded from the CI gate."""
    s = load_settings()
    e = get_embedder(s)
    q = e.embed_query("What is self-attention?")
    docs = e.embed_documents(
        ["Self-attention relates positions of a single sequence.", "Tax law in Belgium."]
    )
    assert len(q) == s.embeddings.dimension
    assert _cos(q, docs[0]) > _cos(q, docs[1])

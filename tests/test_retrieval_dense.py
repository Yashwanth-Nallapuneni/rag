from __future__ import annotations

from ragpipe.retrieval import get_retriever
from ragpipe.retrieval.dense import DenseRetriever


def test_retrieves_ranked_chunks(offline_store, corpus_chunks):
    settings, store = offline_store
    hits = DenseRetriever(settings, store).retrieve(corpus_chunks[0].text[:200], k=5)
    assert hits
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_dense_scores_are_recorded_separately(offline_store, corpus_chunks):
    """Per-stage scores are what let the eval harness attribute a win to
    fusion or reranking instead of to 'retrieval' as a blob."""
    settings, store = offline_store
    hits = DenseRetriever(settings, store).retrieve(corpus_chunks[0].text[:200], k=3)
    for h in hits:
        assert h.dense_score is not None
        assert h.dense_rank is not None
        assert h.retriever == "dense"
        assert h.sparse_score is None and h.rerank_score is None


def test_empty_query_returns_nothing(offline_store):
    settings, store = offline_store
    assert DenseRetriever(settings, store).retrieve("   ") == []


def test_respects_k(offline_store, corpus_chunks):
    settings, store = offline_store
    assert len(DenseRetriever(settings, store).retrieve(corpus_chunks[1].text[:100], k=3)) <= 3


def test_factory_honours_configured_mode(offline_store):
    settings, store = offline_store
    settings.retrieval.mode = "dense"
    assert get_retriever(settings, store).name == "dense"


def test_hybrid_mode_is_still_runnable_before_phase_4(offline_store, corpus_chunks):
    """A hybrid config must degrade to dense rather than crash."""
    settings, store = offline_store
    settings.retrieval.mode = "hybrid"
    retriever = get_retriever(settings, store)
    assert retriever.retrieve(corpus_chunks[0].text[:120], k=3)

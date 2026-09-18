from __future__ import annotations

import pytest

from ragpipe.providers.base import RerankResult
from ragpipe.retrieval.hybrid import HybridRetriever
from ragpipe.retrieval.rerank import RerankStage
from ragpipe.schemas import Chunk, RetrievedChunk


def _rc(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"c{i}", doc_id="d", doc_title="T", chunk_index=i, text=f"body {i}"
        ),
        score=1.0 / i,
        rank=i,
        fusion_score=1.0 / i,
    )


class _ReversingReranker:
    """Deterministically inverts the incoming order, so a stage that ignores
    the reranker or mis-maps its indices is immediately visible."""

    name = "fake"
    model = "reversing"

    def rerank(self, query, documents, top_n=None):
        n = len(documents)
        return [RerankResult(index=i, score=float(i)) for i in reversed(range(n))]

    def health(self):
        return {"ready": True}


@pytest.fixture
def stage(settings, monkeypatch):
    import ragpipe.retrieval.rerank as mod

    monkeypatch.setattr(mod, "get_reranker", lambda s: _ReversingReranker())
    settings.rerank.enabled = True
    settings.rerank.score_threshold = None
    return RerankStage(settings)


def test_rerank_reorders(stage):
    candidates = [_rc(i) for i in range(1, 6)]
    out = stage.rerank("q", candidates, top_n=5)
    assert [rc.chunk_id for rc in out] == ["c5", "c4", "c3", "c2", "c1"]


def test_rerank_maps_indices_to_the_right_chunks(stage):
    """An index mis-map silently attaches one chunk's citation to another
    chunk's text -- the worst possible failure here."""
    out = stage.rerank("q", [_rc(i) for i in range(1, 4)], top_n=3)
    for rc in out:
        assert rc.chunk.text == f"body {rc.chunk.chunk_index}"


def test_rerank_records_scores_and_preserves_fusion_score(stage):
    out = stage.rerank("q", [_rc(i) for i in range(1, 4)], top_n=3)
    for rc in out:
        assert rc.rerank_score is not None
        assert rc.fusion_score is not None, "pre-rerank score must stay separable"
        assert rc.retriever == "rerank"
    assert [rc.rank for rc in out] == [1, 2, 3]


def test_rerank_truncates_to_top_n(stage):
    assert len(stage.rerank("q", [_rc(i) for i in range(1, 11)], top_n=3)) == 3


def test_rerank_handles_empty_and_blank(stage):
    assert stage.rerank("q", []) == []
    assert len(stage.rerank("  ", [_rc(1), _rc(2)], top_n=1)) == 1


def test_threshold_never_empties_the_result(settings, monkeypatch):
    """Returning nothing makes the pipeline refuse, which is a much stronger
    claim than 'these passages scored low'."""
    import ragpipe.retrieval.rerank as mod

    monkeypatch.setattr(mod, "get_reranker", lambda s: _ReversingReranker())
    settings.rerank.enabled = True
    settings.rerank.score_threshold = 10_000.0
    out = RerankStage(settings).rerank("q", [_rc(i) for i in range(1, 4)], top_n=3)
    assert len(out) == 1


def test_out_of_range_index_is_skipped_not_crashed(settings, monkeypatch):
    class Bad:
        name = "bad"
        model = "bad"

        def rerank(self, query, documents, top_n=None):
            return [RerankResult(index=99, score=1.0), RerankResult(index=0, score=0.5)]

        def health(self):
            return {"ready": True}

    import ragpipe.retrieval.rerank as mod

    monkeypatch.setattr(mod, "get_reranker", lambda s: Bad())
    settings.rerank.enabled = True
    settings.rerank.score_threshold = None
    out = RerankStage(settings).rerank("q", [_rc(1), _rc(2)], top_n=5)
    assert [rc.chunk_id for rc in out] == ["c1"]


# --- hybrid end to end ----------------------------------------------------


@pytest.fixture
def hybrid(offline_store, corpus_chunks, monkeypatch):
    from ragpipe.retrieval.bm25 import BM25Retriever
    from ragpipe.retrieval.dense import DenseRetriever

    settings, store = offline_store
    settings.retrieval.mode = "hybrid"
    settings.rerank.enabled = False
    return HybridRetriever(
        settings,
        store,
        dense=DenseRetriever(settings, store),
        sparse=BM25Retriever(settings, corpus_chunks),
        reranker=None,
    )


def test_hybrid_returns_top_k(hybrid, corpus_chunks):
    hits = hybrid.retrieve(corpus_chunks[0].text[:150], k=5)
    assert 0 < len(hits) <= 5


def test_hybrid_results_carry_both_signals(hybrid, corpus_chunks):
    hits = hybrid.retrieve(corpus_chunks[2].text[:150], k=10)
    assert any(h.dense_score is not None for h in hits)
    assert any(h.sparse_score is not None for h in hits)
    assert all(h.fusion_score is not None for h in hits)


def test_hybrid_retrieves_a_wide_shortlist_before_narrowing(hybrid, corpus_chunks):
    """The reranker can only reorder what it is given, so the first pass must
    be wider than top_k."""
    dense_hits, sparse_hits = hybrid._first_pass(
        corpus_chunks[1].text[:150], hybrid.cfg.candidate_k, None
    )
    assert len(dense_hits) > 5 or len(sparse_hits) > 5


def test_hybrid_blank_query(hybrid):
    assert hybrid.retrieve("   ") == []


def test_hybrid_describes_itself(hybrid):
    described = hybrid.describe()
    assert described["mode"] == "hybrid"
    assert described["dense"] == "dense" and described["sparse"] == "sparse"


def test_hybrid_survives_one_dead_retriever(offline_store, corpus_chunks):
    """Sparse-only and dense-only configs must both still work."""
    from ragpipe.retrieval.bm25 import BM25Retriever

    settings, store = offline_store
    settings.retrieval.mode = "sparse"
    settings.rerank.enabled = False
    sparse_only = HybridRetriever(
        settings, store, dense=None, sparse=BM25Retriever(settings, corpus_chunks),
        reranker=None,
    )
    assert sparse_only.retrieve("model evaluation dataset", k=3)


def test_sparse_index_is_built_from_the_store_not_the_chunk_file(
    offline_store, corpus_chunks
):
    """Regression: BM25 used to read chunks.jsonl independently, so it could
    serve chunks the vector store did not have. Those produce citations whose
    click-through 404s, and an empty store still answered."""
    settings, store = offline_store
    settings.retrieval.mode = "hybrid"
    settings.rerank.enabled = False
    retriever = HybridRetriever(settings, store)  # builds its own sparse index

    assert retriever.sparse is not None
    assert retriever.sparse.stats()["documents"] == store.count()

    indexed_ids = {c.chunk_id for c in corpus_chunks}
    hits = retriever.retrieve("evaluation of model performance on the benchmark", k=10)
    assert hits
    for hit in hits:
        assert hit.chunk_id in indexed_ids
        assert store.get([hit.chunk_id]), "a hit the store cannot return is not citable"


def test_empty_store_disables_sparse_rather_than_half_answering(tmp_path):
    from ragpipe.config import load_settings
    from ragpipe.index.chroma_store import ChromaStore

    settings = load_settings(
        overrides={
            "llm": {"provider": "mock"},
            "embeddings": {"provider": "mock", "dimension": 384},
            "rerank": {"provider": "mock", "enabled": False},
            "retrieval": {"mode": "hybrid"},
            "vector_store": {"path": str(tmp_path / "empty"), "collection": "empty"},
        }
    )
    store = ChromaStore(settings.vector_store, 384)
    retriever = HybridRetriever(settings, store)
    assert retriever.sparse is None
    assert retriever.retrieve("anything at all") == []

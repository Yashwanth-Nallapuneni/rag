from __future__ import annotations

import pytest

from ragpipe.retrieval.bm25 import BM25Retriever
from ragpipe.retrieval.tokenize import tokenize


# --- tokenizer ------------------------------------------------------------


@pytest.mark.parametrize(
    "text,must_contain",
    [
        ("Cross-encoder reranking", "cross-encoder"),
        ("bge-small-en-v1.5 embeddings", "bge-small-en-v1.5"),
        ("filed under cs.CL today", "cs.cl"),
        ("the GPT-4o model", "gpt-4o"),
        ("set model_name here", "model_name"),
    ],
)
def test_identifiers_survive_tokenization(text, must_contain):
    """Shattering bge-small-en-v1.5 into 'bge small en v1 5' destroys exactly
    the exact-term matching BM25 is here to provide."""
    assert must_contain in tokenize(text)


def test_compounds_also_emit_subtokens():
    tokens = tokenize("cross-encoder")
    assert "cross-encoder" in tokens
    assert "cross" in tokens and "encoder" in tokens


def test_stopwords_removed():
    assert tokenize("the of and a an") == []


def test_no_stemming():
    """Stemming would trade away the surface-form precision BM25 provides."""
    assert "retrieval" in tokenize("retrieval")
    assert "retrieve" not in tokenize("retrieval")


# --- retriever ------------------------------------------------------------


@pytest.fixture(scope="module")
def bm25(request):
    from ragpipe.config import load_settings
    from ragpipe.ingest.pipeline import read_chunks
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "processed" / "chunks.jsonl"
    if not path.exists():
        pytest.skip("run `make ingest` first")
    settings = load_settings()
    return BM25Retriever(settings, read_chunks(path))


def test_retrieves_ranked_results(bm25):
    hits = bm25.retrieve("self-attention transformer architecture", k=5)
    assert hits
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_scores_are_normalised_but_raw_is_kept(bm25):
    """Fusion mixes these with cosine similarities, so an unbounded BM25
    score would dominate any weighted sum."""
    hits = bm25.retrieve("evaluation benchmark dataset", k=10)
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert all(h.sparse_score is not None for h in hits)
    assert any(h.sparse_score > 1.0 for h in hits), "raw BM25 value was lost"


def test_sparse_provenance_is_tagged(bm25):
    for h in bm25.retrieve("neural network training", k=3):
        assert h.retriever == "sparse"
        assert h.sparse_rank is not None
        assert h.dense_score is None


def test_exact_term_actually_appears_in_results(bm25):
    """The point of BM25 here: a literal term lookup returns passages that
    literally contain the term."""
    hits = bm25.retrieve("eCPM", k=5)
    if hits:
        assert any("ecpm" in h.chunk.text.lower() for h in hits)


def test_empty_and_stopword_queries_return_nothing(bm25):
    assert bm25.retrieve("", k=5) == []
    assert bm25.retrieve("   ", k=5) == []
    assert bm25.retrieve("the of and", k=5) == []


def test_unknown_term_returns_nothing_not_junk(bm25):
    """Zero-score filler would be fused as if it were evidence."""
    assert bm25.retrieve("zzzzqqqxyzzy", k=5) == []


def test_respects_k(bm25):
    assert len(bm25.retrieve("model", k=3)) <= 3


def test_stats(bm25):
    stats = bm25.stats()
    for key in ("documents", "vocabulary_size", "avg_doc_length", "k1", "b"):
        assert key in stats
    assert stats["documents"] > 100


def test_stale_index_is_rebuilt_not_served(tmp_path):
    """A stale sparse index beside a fresh dense one is a very confusing
    class of bug."""
    from ragpipe.config import load_settings
    from ragpipe.ingest.pipeline import read_chunks
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = root / "data" / "processed" / "chunks.jsonl"
    if not src.exists():
        pytest.skip("run `make ingest` first")
    chunks = read_chunks(src)[:80]
    index_path = tmp_path / "bm25.pkl"
    settings = load_settings(
        overrides={"retrieval": {"bm25": {"index_path": str(index_path)}}}
    )

    first = BM25Retriever.build_or_load(settings, chunks)
    assert index_path.exists()
    baseline = first.retrieve("model evaluation", k=5)

    reloaded = BM25Retriever.build_or_load(settings, chunks)
    assert [h.chunk_id for h in reloaded.retrieve("model evaluation", k=5)] == [
        h.chunk_id for h in baseline
    ]

    rebuilt = BM25Retriever.build_or_load(settings, chunks[:40])
    assert rebuilt.stats()["documents"] == 40

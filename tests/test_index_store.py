from __future__ import annotations

import pytest

from ragpipe.index.base import VectorStoreError
from ragpipe.providers import get_embedder


def test_upsert_and_count(offline_store, corpus_chunks):
    _, store = offline_store
    assert store.count() == len(corpus_chunks)


def test_upsert_is_idempotent(offline_store, corpus_chunks):
    settings, store = offline_store
    embedder = get_embedder(settings)
    before = store.count()
    store.upsert(corpus_chunks, embedder.embed_documents([c.text for c in corpus_chunks]))
    assert store.count() == before, "re-indexing must replace, not duplicate"


def test_dimension_mismatch_is_rejected(offline_store, corpus_chunks):
    """Silently accepting a wrong-width vector corrupts the collection."""
    _, store = offline_store
    with pytest.raises(VectorStoreError):
        store.upsert(corpus_chunks[:1], [[0.1, 0.2, 0.3]])


def test_length_mismatch_is_rejected(offline_store, corpus_chunks):
    _, store = offline_store
    with pytest.raises(VectorStoreError):
        store.upsert(corpus_chunks[:2], [[0.0] * 384])


def test_query_returns_similarity_not_distance(offline_store, corpus_chunks):
    """Downstream fusion mixes these scores; a distance leaking through
    would silently invert the ranking."""
    settings, store = offline_store
    embedder = get_embedder(settings)
    hits = store.query(embedder.embed_query(corpus_chunks[0].text[:200]), k=5)
    assert hits
    scores = [s for _, s in hits]
    assert all(0.0 <= s <= 1.0 for s in scores), scores
    assert scores == sorted(scores, reverse=True)


def test_query_finds_the_exact_chunk_it_was_given(offline_store, corpus_chunks):
    settings, store = offline_store
    embedder = get_embedder(settings)
    target = corpus_chunks[3]
    hits = store.query(embedder.embed_query(target.text), k=3)
    assert hits[0][0].chunk_id == target.chunk_id


def test_get_preserves_request_order_and_skips_unknown(offline_store, corpus_chunks):
    _, store = offline_store
    ids = [corpus_chunks[5].chunk_id, "nope", corpus_chunks[1].chunk_id]
    got = store.get(ids)
    assert [c.chunk_id for c in got] == [corpus_chunks[5].chunk_id, corpus_chunks[1].chunk_id]


def test_provenance_survives_the_store_roundtrip(offline_store, corpus_chunks):
    """If page/section are lost here, every citation degrades to a filename."""
    _, store = offline_store
    original = corpus_chunks[2]
    restored = store.get([original.chunk_id])[0]
    assert restored.text == original.text
    assert restored.page_start == original.page_start
    assert restored.section_path == original.section_path
    assert restored.doc_title == original.doc_title
    assert restored.locator() == original.locator()


def test_document_ids(offline_store, corpus_chunks):
    _, store = offline_store
    assert set(store.document_ids()) == {c.doc_id for c in corpus_chunks}


def test_reset_empties_the_collection(offline_store, corpus_chunks):
    settings, store = offline_store
    store.reset()
    assert store.count() == 0
    embedder = get_embedder(settings)
    assert store.query(embedder.embed_query("anything"), k=3) == []


def test_stats_reports_the_essentials(offline_store):
    _, store = offline_store
    stats = store.stats()
    for key in ("provider", "collection", "count", "dimension", "distance"):
        assert key in stats

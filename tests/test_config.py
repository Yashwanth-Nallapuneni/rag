from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragpipe.config import load_settings


def test_defaults_match_spec():
    s = load_settings()
    # The spec mandates 500-800 token chunks with ~100 token overlap.
    assert 500 <= s.chunking.chunk_size <= 800
    assert s.chunking.chunk_overlap == 100
    assert s.retrieval.top_k == 5
    assert s.retrieval.mode == "hybrid"


def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("RAGPIPE_RETRIEVAL__TOP_K", "11")
    monkeypatch.setenv("RAGPIPE_LLM__PROVIDER", "openai")
    s = load_settings()
    assert s.retrieval.top_k == 11
    assert s.llm.provider == "openai"


def test_explicit_overrides_beat_env(monkeypatch):
    monkeypatch.setenv("RAGPIPE_RETRIEVAL__TOP_K", "11")
    s = load_settings(overrides={"retrieval": {"top_k": 3}})
    assert s.retrieval.top_k == 3


def test_fingerprint_is_stable_and_sensitive():
    a = load_settings()
    b = load_settings()
    assert a.fingerprint() == b.fingerprint()
    c = load_settings(overrides={"chunking": {"chunk_size": 700}})
    assert c.fingerprint() != a.fingerprint(), "config changes must change the fingerprint"


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValidationError):
        load_settings(overrides={"chunking": {"chunk_size": 400, "chunk_overlap": 400}})


def test_top_k_cannot_exceed_candidate_k():
    with pytest.raises(ValidationError):
        load_settings(overrides={"retrieval": {"top_k": 50, "candidate_k": 10}})


def test_paths_resolve_absolute():
    s = load_settings()
    assert s.vector_store.store_path.is_absolute()
    assert s.corpus.raw_path.is_absolute()

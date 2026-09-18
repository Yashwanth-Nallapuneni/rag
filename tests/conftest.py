from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import Settings, load_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep a developer's real .env / exported keys out of the test run."""
    for var in list(os.environ):
        if var.startswith("RAGPIPE_"):
            monkeypatch.delenv(var, raising=False)
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def settings() -> Settings:
    """Fully offline settings: no downloads, no keys, deterministic."""
    return load_settings(
        overrides={
            "llm": {"provider": "mock"},
            "embeddings": {"provider": "mock", "dimension": 384},
            "rerank": {"provider": "mock"},
        }
    )


@pytest.fixture(scope="session")
def corpus_chunks():
    """A slice of the real ingested corpus. Skips if it has not been built."""
    from ragpipe.ingest.pipeline import read_chunks

    path = ROOT / "data" / "processed" / "chunks.jsonl"
    if not path.exists():
        pytest.skip("run `make ingest` first")
    return read_chunks(path)[:60]


@pytest.fixture
def offline_store(tmp_path, corpus_chunks):
    """A populated Chroma collection built with the deterministic mock
    embedder, so store and retrieval tests need no downloads and no network."""
    from ragpipe.config import load_settings
    from ragpipe.index.chroma_store import ChromaStore
    from ragpipe.providers import get_embedder

    settings = load_settings(
        overrides={
            "llm": {"provider": "mock"},
            "embeddings": {"provider": "mock", "dimension": 384},
            "rerank": {"provider": "mock"},
            "vector_store": {"path": str(tmp_path / "chroma"), "collection": "test"},
        }
    )
    embedder = get_embedder(settings)
    store = ChromaStore(settings.vector_store, settings.embeddings.dimension)
    store.upsert(corpus_chunks, embedder.embed_documents([c.text for c in corpus_chunks]))
    return settings, store

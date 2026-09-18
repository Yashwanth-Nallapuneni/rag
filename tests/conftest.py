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

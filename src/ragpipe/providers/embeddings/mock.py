"""Deterministic offline embedder for CI and unit tests.

This is not a toy random-vector stub. Retrieval tests need cosine similarity
to actually track shared vocabulary -- a pure hash-of-the-whole-string
embedder makes every pair of texts equally (un)related and every retrieval
assertion meaningless. Instead this hashes individual *tokens* into buckets
(the hashing trick), so two texts sharing words land closer together than
two texts that share none, while remaining fully deterministic and offline.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _token_seed(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class MockEmbedder:
    """EmbeddingProvider that hashes text into a deterministic bag-of-words
    vector. Same text always yields the same vector; texts sharing vocabulary
    yield vectors with higher cosine similarity than unrelated texts."""

    name = "mock"

    def __init__(self, model: str = "mock-hash-embed-v1", dimension: int = 384):
        self.model = model
        self.dimension = dimension

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float64)
        toks = _tokens(text)
        if not toks:
            toks = [""]
        for tok in toks:
            seed = _token_seed(tok)
            bucket = seed % self.dimension
            # A per-token RandomState gives each token a small, fixed random
            # feature vector rather than a single scalar, which spreads
            # token identity across dimensions instead of only ever hitting
            # one bucket per token.
            rng = np.random.RandomState(seed & 0xFFFFFFFF)
            sign = 1.0 if (seed >> 32) % 2 == 0 else -1.0
            bump = rng.normal(loc=0.0, scale=1.0, size=8)
            for i, b in enumerate(bump):
                vec[(bucket + i) % self.dimension] += sign * float(b)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return [float(x) for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "dimension": self.dimension,
            "device": "cpu",
            "ready": True,
            "loaded": True,
            "offline": True,
        }

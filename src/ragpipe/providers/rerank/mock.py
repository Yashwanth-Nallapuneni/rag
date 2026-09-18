"""Deterministic offline reranker for CI and unit tests.

Scores documents by lexical token-overlap F1 against the query, so the
ordering it produces is meaningful (more shared vocabulary -> higher score)
rather than arbitrary -- retrieval and citation tests need a rerank stage
whose behaviour they can actually assert on, with no downloads or network.
"""

from __future__ import annotations

import re
from typing import Any

from ..base import RerankResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _f1(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(doc_tokens)
    recall = overlap / len(query_tokens)
    return 2 * precision * recall / (precision + recall)


class MockReranker:
    """Reranker that scores by lexical token-overlap F1."""

    name = "mock"

    def __init__(self, model: str = "mock-lexical-rerank-v1"):
        self.model = model

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        if not documents:
            return []
        q_tokens = _tokens(query)
        results = [
            RerankResult(index=i, score=_f1(q_tokens, _tokens(doc)))
            for i, doc in enumerate(documents)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "ready": True,
            "loaded": True,
            "offline": True,
        }

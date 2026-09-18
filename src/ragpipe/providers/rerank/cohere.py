"""Cohere reranker backend.

The `cohere` SDK is an optional dependency, imported lazily. This provider is
a structurally complete stub: without `cohere` installed and `COHERE_API_KEY`
set it fails cleanly with a `ProviderError`/`MissingCredentialsError` rather
than a raw SDK exception.
"""

from __future__ import annotations

import os
from typing import Any

from ..base import MissingCredentialsError, ProviderError, RerankResult, retry_with_backoff

DEFAULT_MODEL = "rerank-v3.5"


class CohereReranker:
    """Reranker backed by the Cohere rerank API."""

    name = "cohere"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.api_key = os.getenv("COHERE_API_KEY")
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise MissingCredentialsError(self.name, "COHERE_API_KEY")
        try:
            import cohere
        except ImportError as exc:
            raise ProviderError(
                "the 'cohere' package is required for provider 'cohere'. "
                "Install it with: pip install cohere"
            ) from exc
        self._client = cohere.Client(self.api_key)
        return self._client

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        if not documents:
            return []
        client = self._ensure_client()

        def _call() -> Any:
            return client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n or len(documents),
            )

        response = retry_with_backoff(_call, attempts=3)
        return [
            RerankResult(index=r.index, score=float(r.relevance_score))
            for r in response.results
        ]

    def health(self) -> dict[str, Any]:
        try:
            import cohere  # noqa: F401

            importable = True
        except ImportError:
            importable = False
        return {
            "provider": self.name,
            "model": self.model,
            "ready": importable and bool(self.api_key),
            "sdk_importable": importable,
            "has_credentials": bool(self.api_key),
            "loaded": self._client is not None,
        }

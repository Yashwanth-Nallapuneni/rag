"""OpenAI embeddings backend.

The `openai` SDK is an optional dependency, imported lazily so environments
running only `mock`/`local` providers never need it installed.
"""

from __future__ import annotations

import os
from typing import Any

from ..base import MissingCredentialsError, ProviderError, retry_with_backoff

DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbedder:
    """EmbeddingProvider backed by the OpenAI embeddings API."""

    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dimension: int = 1536,
        batch_size: int = 32,
    ):
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client: Any = None
        # Fail at construction, not at first embed call: every provider in
        # this project surfaces a missing key up front, so a misconfigured
        # run stops before it spends time ingesting a corpus.
        if not self.api_key:
            raise MissingCredentialsError(self.name, "OPENAI_API_KEY")

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise MissingCredentialsError(self.name, "OPENAI_API_KEY")
        try:
            import openai
        except ImportError as exc:
            raise ProviderError(
                "the 'openai' package is required for provider 'openai'. "
                "Install it with: pip install openai"
            ) from exc
        self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = self._ensure_client()

        def _call() -> Any:
            return client.embeddings.create(model=self.model, input=texts)

        response = retry_with_backoff(_call, attempts=3)
        return [list(item.embedding) for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def health(self) -> dict[str, Any]:
        try:
            import openai  # noqa: F401

            importable = True
        except ImportError:
            importable = False
        return {
            "provider": self.name,
            "model": self.model,
            "dimension": self.dimension,
            "device": "api",
            "ready": importable and bool(self.api_key),
            "sdk_importable": importable,
            "has_credentials": bool(self.api_key),
            "loaded": self._client is not None,
        }

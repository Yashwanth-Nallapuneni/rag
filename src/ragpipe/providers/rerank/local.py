"""Local cross-encoder reranker.

Like the local embedder, `sentence_transformers` and the model weights are
loaded lazily: constructing this provider (which happens on every cache
lookup in `get_reranker`) must not trigger a download.
"""

from __future__ import annotations

from typing import Any

from ..base import ProviderError, RerankResult

DEFAULT_MODEL = "BAAI/bge-reranker-base"


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class CrossEncoderReranker:
    """Reranker backed by a local sentence-transformers CrossEncoder.

    Scores are the model's raw logits: they may be negative and are not on
    any fixed scale, and are returned as-is (not clamped or renormalised).
    Only their relative order is meaningful.

    The activation is pinned to identity explicitly. sentence-transformers
    picks a per-model default: a sigmoid for single-label models whose config
    does not say otherwise (BAAI/bge-reranker-base gets one), identity for
    models that declare it (ms-marco-MiniLM-L-6-v2 does). The relevance gate
    (`citation.min_relevance_score`) is a threshold on the raw logit, so the
    result must not depend on that default. Inputs longer than the model's
    512-token window are truncated by the tokenizer, never an error.
    """

    name = "local"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        batch_size: int = 16,
    ):
        self.model = model
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self._cross_encoder: Any = None

    def _ensure_model(self) -> Any:
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ProviderError(
                "the 'sentence-transformers' package is required for provider "
                "'local' reranking. Install it with: pip install sentence-transformers"
            ) from exc
        import torch

        self._cross_encoder = CrossEncoder(
            self.model, device=self.device, activation_fn=torch.nn.Identity()
        )
        return self._cross_encoder

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]:
        if not documents:
            return []

        import torch

        cross_encoder = self._ensure_model()
        pairs = [(query, doc) for doc in documents]
        scores = cross_encoder.predict(
            pairs, batch_size=self.batch_size, activation_fn=torch.nn.Identity()
        )

        # `index` must point back into the caller's original `documents`
        # list -- the pipeline uses it to map results back to source chunks
        # for citations, so it must survive the sort below.
        results = [
            RerankResult(index=i, score=float(score)) for i, score in enumerate(scores)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "device": self.device,
            "ready": True,
            "loaded": self._cross_encoder is not None,
        }

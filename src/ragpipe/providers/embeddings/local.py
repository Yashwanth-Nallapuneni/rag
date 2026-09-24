"""Local sentence-transformers embedder (BGE family by default).

`sentence_transformers` and its model weights are heavy: importing the
package costs real time and loading a model can mean a multi-hundred-MB
download. Both are deferred until the first `embed_documents`/`embed_query`
call (or an explicit `_ensure_model()`), so constructing this provider --
which happens for every cache lookup in `get_embedder` -- stays cheap.
"""

from __future__ import annotations

from typing import Any

from ..base import ProviderError

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalEmbedder:
    """EmbeddingProvider backed by a local sentence-transformers model."""

    name = "local"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dimension: int = 384,
        normalize: bool = True,
        query_prefix: str = "",
        device: str = "auto",
        batch_size: int = 32,
    ):
        self.model = model
        self.dimension = dimension
        self.normalize = normalize
        self.query_prefix = query_prefix
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self._encoder: Any = None

    def _ensure_model(self) -> Any:
        if self._encoder is not None:
            return self._encoder

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ProviderError(
                "the 'sentence-transformers' package is required for provider "
                "'local'. Install it with: pip install sentence-transformers"
            ) from exc

        encoder = SentenceTransformer(self.model, device=self.device)
        # `get_sentence_embedding_dimension` was renamed to
        # `get_embedding_dimension` in newer sentence-transformers releases;
        # support both without triggering the deprecation warning.
        dim_fn = getattr(encoder, "get_embedding_dimension", None) or encoder.get_sentence_embedding_dimension
        actual_dim = dim_fn()
        if actual_dim != self.dimension:
            raise ProviderError(
                f"configured embeddings.dimension={self.dimension} does not match "
                f"the model's actual output dimension={actual_dim} for '{self.model}'. "
                "A silent mismatch here corrupts the vector store -- fix the config."
            )
        self._encoder = encoder
        return encoder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        encoder = self._ensure_model()
        vectors = encoder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # ChromaDB (and JSON serialisation generally) rejects numpy scalar
        # types, so cast down to plain Python floats before returning.
        return [[float(x) for x in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        # BGE-style models are trained asymmetrically: queries get a fixed
        # instruction prefix ("Represent this sentence for searching relevant
        # passages: ") but documents never do. Applying it to documents too
        # would collapse the asymmetry the model was trained to exploit.
        encoder = self._ensure_model()
        text_in = f"{self.query_prefix}{text}" if self.query_prefix else text
        vector = encoder.encode(
            [text_in],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return [float(x) for x in vector]

    def health(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "dimension": self.dimension,
            "device": self.device,
            "ready": True,
            "loaded": self._encoder is not None,
        }

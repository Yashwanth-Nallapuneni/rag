"""Provider registry.

Concrete backends are imported lazily, so the Anthropic/OpenAI/Cohere SDKs are
optional dependencies: running the whole pipeline offline with `mock`/`local`
providers never imports them.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .base import (
    EmbeddingProvider,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    MissingCredentialsError,
    ProviderError,
    Reranker,
    RerankResult,
)

if TYPE_CHECKING:
    from ..config import Settings

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "MissingCredentialsError",
    "ProviderError",
    "Reranker",
    "RerankResult",
    "default_model_for",
    "get_llm",
    "get_embedder",
    "get_reranker",
]


def default_model_for(provider: str) -> str | None:
    """The model a provider uses when config leaves `model` unset.

    Needed because a label like "groq:default" is useless for pricing and for
    reporting: the placeholder has to be resolved to the real model id. Done
    by importing the provider module's constant rather than constructing the
    provider, so this works with no credentials present.
    """
    modules = {
        "mock": ("mock", "mock-extractive-v1"),
        "anthropic": ("anthropic", None),
        "openai": ("openai", None),
        "groq": ("groq", None),
        "openrouter": ("openrouter", None),
        "ollama": ("ollama", None),
    }
    entry = modules.get(provider)
    if entry is None:
        return None
    module_name, literal = entry
    if literal:
        return literal
    import importlib

    try:
        module = importlib.import_module(f".llm.{module_name}", __package__)
    except ImportError:
        return None
    return getattr(module, "DEFAULT_MODEL", None)


def get_llm(settings: Settings) -> LLMProvider:
    cfg = settings.llm
    provider = cfg.provider
    if provider == "mock":
        from .llm.mock import MockLLM

        return MockLLM(model=cfg.model or "mock-extractive-v1")
    if provider == "anthropic":
        from .llm.anthropic import AnthropicLLM

        return AnthropicLLM(cfg)
    if provider == "openai":
        from .llm.openai import OpenAILLM

        return OpenAILLM(cfg)
    if provider == "groq":
        from .llm.groq import GroqLLM

        return GroqLLM(cfg)
    if provider == "openrouter":
        from .llm.openrouter import OpenRouterLLM

        return OpenRouterLLM(cfg)
    if provider == "ollama":
        from .llm.ollama import OllamaLLM

        return OllamaLLM(cfg)
    raise ProviderError(f"unknown llm provider: {provider}")


@lru_cache(maxsize=4)
def _cached_embedder(key: tuple) -> EmbeddingProvider:
    provider, model, dimension, normalize, prefix, device, batch_size = key
    if provider == "mock":
        from .embeddings.mock import MockEmbedder

        return MockEmbedder(model=model, dimension=dimension)
    if provider == "local":
        from .embeddings.local import LocalEmbedder

        return LocalEmbedder(
            model=model,
            dimension=dimension,
            normalize=normalize,
            query_prefix=prefix,
            device=device,
            batch_size=batch_size,
        )
    if provider == "openai":
        from .embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder(model=model, dimension=dimension, batch_size=batch_size)
    raise ProviderError(f"unknown embeddings provider: {provider}")


def get_embedder(settings: Settings) -> EmbeddingProvider:
    """Embedding models are expensive to load, so instances are cached by the
    settings that define them."""
    c = settings.embeddings
    return _cached_embedder(
        (
            c.provider,
            c.model,
            c.dimension,
            c.normalize,
            c.query_prefix,
            c.device,
            c.batch_size,
        )
    )


@lru_cache(maxsize=4)
def _cached_reranker(key: tuple) -> Reranker:
    provider, model, device, batch_size = key
    if provider in ("none", "mock"):
        from .rerank.mock import MockReranker

        return MockReranker(model=model)
    if provider == "local":
        from .rerank.local import CrossEncoderReranker

        return CrossEncoderReranker(model=model, device=device, batch_size=batch_size)
    if provider == "cohere":
        from .rerank.cohere import CohereReranker

        return CohereReranker(model=model)
    raise ProviderError(f"unknown rerank provider: {provider}")


def get_reranker(settings: Settings) -> Reranker:
    c = settings.rerank
    return _cached_reranker((c.provider, c.model, c.device, c.batch_size))


def clear_provider_cache() -> None:
    _cached_embedder.cache_clear()
    _cached_reranker.cache_clear()

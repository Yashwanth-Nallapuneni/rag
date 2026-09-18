"""Provider protocols.

The rest of the pipeline depends on these three interfaces, never on a
concrete SDK. That is what lets the LLM choice stay deferred: a `mock`
provider satisfies the same contract as Anthropic or OpenAI, so retrieval,
citation enforcement and the eval harness are all buildable and testable
before any API key exists.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Task = Literal["answer", "claim_check", "generic"]


@dataclass
class LLMRequest:
    system: str
    user: str
    task: Task = "generic"
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    finish_reason: str | None = None
    raw: Any = None


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def health(self) -> dict[str, Any]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    model: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def health(self) -> dict[str, Any]: ...


@dataclass
class RerankResult:
    index: int
    score: float


@runtime_checkable
class Reranker(Protocol):
    name: str
    model: str

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankResult]: ...

    def health(self) -> dict[str, Any]: ...


class ProviderError(RuntimeError):
    """Raised when a provider is misconfigured or its backend is unreachable."""


class MissingCredentialsError(ProviderError):
    """Raised when a provider needs an API key that is not set.

    Carries the env var name so the CLI and API can tell the user exactly
    what to set instead of surfacing a bare SDK stack trace.
    """

    def __init__(self, provider: str, env_var: str):
        self.provider = provider
        self.env_var = env_var
        super().__init__(
            f"Provider '{provider}' requires {env_var}. "
            f"Set it in .env, or switch to the offline default with "
            f"RAGPIPE_LLM__PROVIDER=mock."
        )


def retry_with_backoff(fn, *, attempts: int, base_delay: float = 0.5):
    """Retry transient provider failures. Credential errors are not retried."""
    last: Exception | None = None
    for i in range(max(1, attempts)):
        try:
            return fn()
        except MissingCredentialsError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider SDKs vary widely
            last = exc
            if i == attempts - 1:
                break
            time.sleep(base_delay * (2**i))
    raise ProviderError(f"provider call failed after {attempts} attempts: {last}") from last

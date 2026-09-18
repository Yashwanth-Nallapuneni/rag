"""Anthropic Claude backend.

The `anthropic` SDK is an optional dependency: it is imported lazily so the
rest of the pipeline (retrieval, citation enforcement, eval harness, and the
`mock` provider) keeps working in environments where it is not installed.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..base import (
    LLMRequest,
    LLMResponse,
    MissingCredentialsError,
    ProviderError,
    retry_with_backoff,
)

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicLLM:
    """LLMProvider backed by the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.model = cfg.model or DEFAULT_MODEL
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise MissingCredentialsError(self.name, "ANTHROPIC_API_KEY")

        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                "the 'anthropic' package is required for provider 'anthropic'. "
                "Install it with: pip install anthropic"
            ) from exc

        self._client = anthropic.Anthropic(api_key=self.api_key, timeout=cfg.timeout_s)

    def complete(self, request: LLMRequest) -> LLMResponse:
        temperature = request.temperature if request.temperature is not None else self.cfg.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else self.cfg.max_tokens

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=request.system,
            messages=[{"role": "user", "content": request.user}],
        )
        if request.stop:
            kwargs["stop_sequences"] = request.stop

        started = time.perf_counter()
        response = retry_with_backoff(
            lambda: self._client.messages.create(**kwargs),
            attempts=self.cfg.max_retries,
        )
        elapsed = (time.perf_counter() - started) * 1000

        text = ""
        if response.content:
            block = response.content[0]
            if getattr(block, "type", None) == "text":
                text = block.text

        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
        }

        return LLMResponse(
            text=text,
            model=self.model,
            usage=usage,
            latency_ms=elapsed,
            finish_reason=response.stop_reason,
            raw=response,
        )

    def health(self) -> dict[str, Any]:
        try:
            import anthropic  # noqa: F401

            importable = True
        except ImportError:
            importable = False
        return {
            "provider": self.name,
            "model": self.model,
            "ready": importable and bool(self.api_key),
            "sdk_importable": importable,
            "has_credentials": bool(self.api_key),
        }

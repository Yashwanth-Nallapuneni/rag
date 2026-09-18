"""OpenAI chat completions backend.

The `openai` SDK is an optional dependency: it is imported lazily so the rest
of the pipeline (retrieval, citation enforcement, eval harness, and the
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

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAILLM:
    """LLMProvider backed by the OpenAI chat completions API."""

    name = "openai"

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.model = cfg.model or DEFAULT_MODEL
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise MissingCredentialsError(self.name, "OPENAI_API_KEY")

        try:
            import openai
        except ImportError as exc:
            raise ProviderError(
                "the 'openai' package is required for provider 'openai'. "
                "Install it with: pip install openai"
            ) from exc

        self._client = openai.OpenAI(api_key=self.api_key, timeout=cfg.timeout_s)

    def complete(self, request: LLMRequest) -> LLMResponse:
        temperature = request.temperature if request.temperature is not None else self.cfg.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else self.cfg.max_tokens

        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
        )
        if request.stop:
            kwargs["stop"] = request.stop

        started = time.perf_counter()
        response = retry_with_backoff(
            lambda: self._client.chat.completions.create(**kwargs),
            attempts=self.cfg.max_retries,
        )
        elapsed = (time.perf_counter() - started) * 1000

        choice = response.choices[0]
        text = choice.message.content or ""

        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(response.usage, "completion_tokens", 0),
        }

        return LLMResponse(
            text=text,
            model=self.model,
            usage=usage,
            latency_ms=elapsed,
            finish_reason=choice.finish_reason,
            raw=response,
        )

    def health(self) -> dict[str, Any]:
        try:
            import openai  # noqa: F401

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

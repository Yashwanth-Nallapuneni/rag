"""Local Ollama backend.

Ollama needs no API key and no optional SDK -- it is a plain HTTP server, so
this talks to it directly over `httpx` (already a project dependency)
against the `/api/chat` endpoint rather than reaching for a client library.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from ..base import LLMRequest, LLMResponse, retry_with_backoff

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_HOST = "http://localhost:11434"


class OllamaLLM:
    """LLMProvider backed by a local Ollama server."""

    name = "ollama"

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.model = cfg.model or DEFAULT_MODEL
        self.host = os.getenv("OLLAMA_HOST", DEFAULT_HOST).rstrip("/")
        self._client = httpx.Client(timeout=cfg.timeout_s)

    def complete(self, request: LLMRequest) -> LLMResponse:
        temperature = request.temperature if request.temperature is not None else self.cfg.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else self.cfg.max_tokens

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if request.stop:
            payload["options"]["stop"] = request.stop

        def _call():
            resp = self._client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()

        started = time.perf_counter()
        data = retry_with_backoff(_call, attempts=self.cfg.max_retries)
        elapsed = (time.perf_counter() - started) * 1000

        text = data.get("message", {}).get("content", "")
        usage = {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
        }
        finish_reason = "stop" if data.get("done") else None

        return LLMResponse(
            text=text,
            model=self.model,
            usage=usage,
            latency_ms=elapsed,
            finish_reason=finish_reason,
            raw=data,
        )

    def health(self) -> dict[str, Any]:
        ready = False
        detail: str | None = None
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=2.0)
            resp.raise_for_status()
            ready = True
        except Exception as exc:  # noqa: BLE001 - health checks must never raise
            detail = str(exc)
        result: dict[str, Any] = {
            "provider": self.name,
            "model": self.model,
            "ready": ready,
            "host": self.host,
        }
        if detail:
            result["error"] = detail
        return result

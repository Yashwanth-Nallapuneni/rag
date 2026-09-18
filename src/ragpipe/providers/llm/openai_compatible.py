"""Shared client for any OpenAI-compatible Chat Completions API.

Groq, OpenRouter, DeepSeek, Together and a local vLLM server all speak the
same `chat.completions.create` wire format as OpenAI itself -- only the base
URL, the credential env var and the default model differ. `OpenAILLM`,
`GroqLLM` and `OpenRouterLLM` are all thin subclasses of `OpenAICompatibleLLM`
below; a brand-new OpenAI-compatible host needs no new code at all, just
`llm.base_url` / `llm.api_key_env` in config (see `config.py`).

This module also owns the two things a free-tier key needs to survive contact
with reality:

  * `RateLimiter` -- paces calls against RPM/TPM ceilings *before* the SDK
    ever sends a request, because Groq's free `openai/gpt-oss-120b` tier
    (30 RPM / 1,000 RPD / 8,000 TPM / 200,000 TPD) is tight enough that this
    project's own ~14,251-token samples exceed the per-minute token budget in
    a single call. Waiting for the limiter is not optional politeness, it is
    what makes the free tier usable instead of an unbroken wall of 429s.
  * 429 handling that honours `Retry-After` and tells a daily-quota
    exhaustion apart from an ordinary per-minute throttle, so a caller can
    tell "wait a minute" from "come back tomorrow" instead of retrying into
    a wall for 24 hours.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable

from ..base import LLMRequest, LLMResponse, MissingCredentialsError, ProviderError

logger = logging.getLogger(__name__)

_WINDOW_S = 60.0
_DAY_S = 86400.0


class QuotaExhaustedError(ProviderError):
    """A 429 that will not clear by waiting a few seconds -- a daily/monthly
    cap, not a per-minute throttle. Distinct from an ordinary rate limit so a
    caller can decide "retry shortly" vs "stop and come back later"."""

    def __init__(self, provider: str, detail: str):
        self.provider = provider
        super().__init__(
            f"provider '{provider}' reports its quota is exhausted, not just "
            f"rate-limited: {detail}. Retrying will not help until the quota "
            f"resets (daily caps typically reset at 00:00 UTC); reduce the "
            f"run size or switch provider/model instead."
        )


class RateLimiter:
    """Client-side pacing against per-minute request/token ceilings.

    Deliberately local and approximate: it has no visibility into the
    server's own counters, so it estimates request tokens up front (from
    `ragpipe.tokenization.count_tokens`) and reconciles against the real
    `response.usage` afterwards via `reconcile()` -- estimates drift, but
    each reconciliation pulls the running total back toward truth instead of
    compounding the drift call after call.

    `clock`/`sleep` are injectable so tests can fake a clock instead of
    actually sleeping for minutes.
    """

    def __init__(
        self,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger = logger,
    ):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._logger = logger
        self._lock = threading.Lock()

        now = self._clock()
        self._window_start = now
        self._requests_in_window = 0
        self._tokens_in_window = 0
        self.total_waits = 0
        self.total_slept_s = 0.0

    def _roll_window_locked(self, now: float) -> None:
        if now - self._window_start >= _WINDOW_S:
            self._window_start = now
            self._requests_in_window = 0
            self._tokens_in_window = 0

    def wait_for_capacity(self, estimated_tokens: int) -> None:
        """Block until both the request and token budgets allow one more
        call. An estimate that alone exceeds `tokens_per_minute` can never be
        satisfied by waiting longer, so it is let through after one window
        reset (with a warning) rather than spinning forever."""
        if self.rpm is None and self.tpm is None:
            return

        with self._lock:
            now = self._clock()
            self._roll_window_locked(now)

            if self.tpm is not None and estimated_tokens > self.tpm:
                self._logger.warning(
                    "estimated request tokens (%d) exceed the configured "
                    "tokens-per-minute ceiling (%d); this single request "
                    "cannot fit any window -- proceeding after the current "
                    "window resets instead of waiting forever",
                    estimated_tokens,
                    self.tpm,
                )
                self._wait_for_reset_locked(now)
                self._requests_in_window += 1
                self._tokens_in_window += estimated_tokens
                return

            while True:
                now = self._clock()
                self._roll_window_locked(now)
                over_requests = self.rpm is not None and self._requests_in_window >= self.rpm
                over_tokens = (
                    self.tpm is not None
                    and self._tokens_in_window + estimated_tokens > self.tpm
                )
                if not (over_requests or over_tokens):
                    break
                self._wait_for_reset_locked(now)

            self._requests_in_window += 1
            self._tokens_in_window += estimated_tokens

    def _wait_for_reset_locked(self, now: float) -> None:
        remaining = max(0.0, _WINDOW_S - (now - self._window_start))
        self.total_waits += 1
        self.total_slept_s += remaining
        if remaining > 0:
            self._sleep(remaining)
        after = self._clock()
        self._window_start = after
        self._requests_in_window = 0
        self._tokens_in_window = 0

    def reconcile(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Pull the window's token count toward the real `response.usage`
        figure once it is known, so accounting converges on truth instead of
        drifting with every estimate error."""
        delta = actual_tokens - estimated_tokens
        if delta == 0:
            return
        with self._lock:
            self._tokens_in_window = max(0, self._tokens_in_window + delta)

    def record_external_wait(self, seconds: float) -> None:
        """Account time slept for a reason outside `wait_for_capacity`
        (e.g. honouring a 429's `Retry-After`), so `as_dict()` reports the
        true total time this provider spent waiting on the network."""
        with self._lock:
            self.total_waits += 1
            self.total_slept_s += max(0.0, seconds)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests_per_minute": self.rpm,
                "tokens_per_minute": self.tpm,
                "requests_used_this_window": self._requests_in_window,
                "tokens_used_this_window": self._tokens_in_window,
                "total_waits": self.total_waits,
                "total_slept_s": round(self.total_slept_s, 3),
            }


def _retry_after_seconds(exc: Any) -> float | None:
    """Read `Retry-After` off an SDK error's response, if the SDK gave us
    one. Groq/OpenRouter both proxy the header through unchanged."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify_429(exc: Any, *, provider: str) -> QuotaExhaustedError | None:
    """Tell a daily/monthly quota exhaustion apart from an ordinary
    per-minute rate limit.

    The openai SDK surfaces a structured `.code`/`.type`/`.body` on
    `APIError` (see openai._exceptions.APIError), so those are checked first
    rather than pattern-matching the message text. No live Groq/OpenRouter
    429 body was available while building this (see module docstring caveat
    in the PR/report) -- the keyword fallback below is a best-effort guess
    at conventions OpenAI-compatible hosts commonly use ("quota", "daily",
    per-day header names), not a verified contract. Treat a very long
    `Retry-After` (over an hour) as a quota signal too: a per-minute throttle
    resets in seconds, not hours.
    """
    code = (getattr(exc, "code", None) or "").lower()
    err_type = (getattr(exc, "type", None) or "").lower()
    body = getattr(exc, "body", None)
    message = str(getattr(exc, "message", None) or exc)
    haystack = " ".join([code, err_type, message]).lower()
    if isinstance(body, dict):
        haystack += " " + str(body).lower()

    quota_keywords = ("quota", "daily", "per-day", "per day", "rpd", "tpd", "monthly")
    if any(k in haystack for k in quota_keywords):
        return QuotaExhaustedError(provider, message)

    retry_after = _retry_after_seconds(exc)
    if retry_after is not None and retry_after > 3600:
        return QuotaExhaustedError(
            provider, f"{message} (Retry-After={retry_after:.0f}s, over an hour)"
        )
    return None


class OpenAICompatibleLLM:
    """LLMProvider backed by any OpenAI-compatible chat completions API.

    Subclasses (`OpenAILLM`, `GroqLLM`, `OpenRouterLLM`) only set
    `name`/`env_var`/`base_url`/`default_model`/`default_rate_limits`; all
    request/retry/rate-limit logic lives here once.
    """

    name = "openai_compatible"
    env_var = "OPENAI_API_KEY"
    base_url: str | None = None
    default_model = "gpt-4o-mini"
    # (requests_per_minute, tokens_per_minute, requests_per_day, tokens_per_day)
    # applied only when the config leaves the corresponding field as None, so
    # a paid-tier user who never sets these is completely unaffected.
    default_rate_limits: tuple[int | None, int | None, int | None, int | None] = (
        None,
        None,
        None,
        None,
    )
    extra_headers: dict[str, str] | None = None

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.model = cfg.model or self.default_model
        env_var = getattr(cfg, "api_key_env", None) or self.env_var
        self.env_var = env_var
        self.api_key = os.getenv(env_var)
        if not self.api_key:
            raise MissingCredentialsError(self.name, env_var)

        try:
            import openai
        except ImportError as exc:
            raise ProviderError(
                f"the 'openai' package is required for provider '{self.name}'. "
                "Install it with: pip install openai"
            ) from exc
        self._openai = openai

        base_url = getattr(cfg, "base_url", None) or self.base_url
        client_kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": cfg.timeout_s}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.resolved_base_url = base_url
        if self.extra_headers:
            client_kwargs["default_headers"] = self.extra_headers
        self._client = openai.OpenAI(**client_kwargs)

        default_rpm, default_tpm, default_rpd, default_tpd = self.default_rate_limits
        rpm = getattr(cfg, "requests_per_minute", None)
        tpm = getattr(cfg, "tokens_per_minute", None)
        self.requests_per_day = getattr(cfg, "requests_per_day", None) or default_rpd
        self.tokens_per_day = getattr(cfg, "tokens_per_day", None) or default_tpd
        self.limiter = RateLimiter(
            requests_per_minute=rpm if rpm is not None else default_rpm,
            tokens_per_minute=tpm if tpm is not None else default_tpm,
        )

    def _estimate_tokens(self, request: LLMRequest, max_tokens: int) -> int:
        from ...tokenization import count_tokens

        return count_tokens(request.system) + count_tokens(request.user) + max_tokens

    def complete(self, request: LLMRequest) -> LLMResponse:
        openai = self._openai
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
        effort = getattr(self.cfg, "reasoning_effort", None)
        if effort:
            kwargs["reasoning_effort"] = effort

        estimated_tokens = self._estimate_tokens(request, max_tokens)
        max_retries = max(1, self.cfg.max_retries)
        base_delay = 0.5
        attempt = 0
        started = time.perf_counter()

        while True:
            self.limiter.wait_for_capacity(estimated_tokens)
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except openai.RateLimitError as exc:
                quota_err = _classify_429(exc, provider=self.name)
                if quota_err is not None:
                    raise quota_err from exc
                attempt += 1
                if attempt >= max_retries:
                    raise ProviderError(
                        f"provider '{self.name}' rate-limited after {attempt} attempts: {exc}"
                    ) from exc
                wait_s = _retry_after_seconds(exc)
                if wait_s is None:
                    wait_s = base_delay * (2 ** (attempt - 1))
                self.limiter.record_external_wait(wait_s)
                time.sleep(wait_s)
            except openai.BadRequestError as exc:
                # A provider that does not know `reasoning_effort` rejects the
                # whole request; drop it once and retry rather than failing.
                if "reasoning_effort" in kwargs and "reasoning" in str(exc).lower():
                    logger.warning(
                        "%s rejected reasoning_effort; retrying without it", self.name
                    )
                    kwargs.pop("reasoning_effort", None)
                    continue
                raise ProviderError(
                    f"provider '{self.name}' rejected the request: {exc}"
                ) from exc
            except (
                openai.AuthenticationError,
                openai.NotFoundError,
            ) as exc:
                # Never worth retrying: bad request/credentials/model name.
                raise ProviderError(f"provider '{self.name}' call failed: {exc}") from exc
            except openai.APIStatusError as exc:
                attempt += 1
                if attempt >= max_retries:
                    raise ProviderError(
                        f"provider '{self.name}' failed after {attempt} attempts: {exc}"
                    ) from exc
                time.sleep(base_delay * (2 ** (attempt - 1)))

        elapsed = (time.perf_counter() - started) * 1000
        choice = response.choices[0]
        text = choice.message.content or ""

        input_tokens = getattr(response.usage, "prompt_tokens", 0)
        output_tokens = getattr(response.usage, "completion_tokens", 0)
        self.limiter.reconcile(estimated_tokens, input_tokens + output_tokens)

        return LLMResponse(
            text=text,
            model=self.model,
            usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
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
            "base_url": self.resolved_base_url,
            "env_var": self.env_var,
            "ready": importable and bool(self.api_key),
            "sdk_importable": importable,
            "has_credentials": bool(self.api_key),
            "rate_limits": self.limiter.as_dict(),
        }

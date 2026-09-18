"""Groq chat completions backend (OpenAI-compatible).

Defaults target the free tier's default model, `openai/gpt-oss-120b`, whose
published free-tier limits are 30 requests/min, 1,000 requests/day,
8,000 tokens/min, 200,000 tokens/day (see Groq's rate-limit docs). This
project's own eval samples average ~14,251 tokens each -- more than one
request can fit in the per-minute token budget -- which is exactly the
scenario `RateLimiter.wait_for_capacity` in `openai_compatible.py` is built
to survive rather than deadlock on.

These defaults only apply when the user's config leaves
`requests_per_minute`/`tokens_per_minute` unset; a paid Groq tier is
unaffected by setting them explicitly (or to `null`).
"""

from __future__ import annotations

from .openai_compatible import OpenAICompatibleLLM

DEFAULT_MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"

# (requests_per_minute, tokens_per_minute, requests_per_day, tokens_per_day)
# -- Groq's published free-tier limits for openai/gpt-oss-120b.
FREE_TIER_LIMITS = (30, 8_000, 1_000, 200_000)


class GroqLLM(OpenAICompatibleLLM):
    """LLMProvider backed by Groq's OpenAI-compatible chat completions API."""

    name = "groq"
    env_var = "GROQ_API_KEY"
    base_url = BASE_URL
    default_model = DEFAULT_MODEL
    default_rate_limits = FREE_TIER_LIMITS

"""Groq/OpenRouter (OpenAI-compatible) provider tests -- all offline.

No API key exists yet, so every test here either (a) proves the registry
fails cleanly without one, (b) constructs a provider with a FAKE key and
inspects `.health()` without making a network call, or (c) drives
`RateLimiter` / `_classify_429` with a fake clock and a stubbed SDK
exception, never a real HTTP request.
"""

from __future__ import annotations

import types

import pytest

from ragpipe.config import load_settings
from ragpipe.eval.cost import (
    UnknownModelPrice,
    estimate_run_feasibility,
    load_price_table,
    resolve_price,
)
from ragpipe.providers import MissingCredentialsError, ProviderError, get_llm
from ragpipe.providers.llm.openai_compatible import (
    QuotaExhaustedError,
    RateLimiter,
    _classify_429,
    _retry_after_seconds,
)

# ---------------------------------------------------------------------------
# 1. Registry: missing credentials name the right env var
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,env_var",
    [("openai", "OPENAI_API_KEY"), ("groq", "GROQ_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")],
)
def test_missing_credentials_names_right_env_var(monkeypatch, provider, env_var):
    monkeypatch.delenv(env_var, raising=False)
    s = load_settings(overrides={"llm": {"provider": provider}})
    with pytest.raises((MissingCredentialsError, ProviderError)) as exc:
        get_llm(s)
    assert env_var in str(exc.value)


# ---------------------------------------------------------------------------
# 2. Fake key -> health() reports correct base_url/model/env_var, no network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,env_var,expected_base_url,expected_model",
    [
        ("openai", "OPENAI_API_KEY", None, "gpt-4o-mini"),
        ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
        (
            "openrouter",
            "OPENROUTER_API_KEY",
            "https://openrouter.ai/api/v1",
            "meta-llama/llama-3.1-8b-instruct",
        ),
    ],
)
def test_health_reports_correct_identity_without_network_call(
    monkeypatch, provider, env_var, expected_base_url, expected_model
):
    monkeypatch.setenv(env_var, "fake-key-not-real")
    s = load_settings(overrides={"llm": {"provider": provider}})
    llm = get_llm(s)
    health = llm.health()
    assert health["base_url"] == expected_base_url
    assert health["model"] == expected_model
    assert health["env_var"] == env_var
    assert health["ready"] is True
    assert health["has_credentials"] is True


def test_groq_default_rate_limits_are_free_tier(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    s = load_settings(overrides={"llm": {"provider": "groq"}})
    llm = get_llm(s)
    assert llm.limiter.rpm == 30
    assert llm.limiter.tpm == 8_000


def test_explicit_rate_limit_config_overrides_provider_default(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    s = load_settings(
        overrides={"llm": {"provider": "groq", "requests_per_minute": 5, "tokens_per_minute": 100}}
    )
    llm = get_llm(s)
    assert llm.limiter.rpm == 5
    assert llm.limiter.tpm == 100


# ---------------------------------------------------------------------------
# 3. RateLimiter, driven by a fake clock -- never sleeps for real
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def test_rpm_is_enforced():
    clock = FakeClock()
    limiter = RateLimiter(requests_per_minute=3, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.wait_for_capacity(0)
    assert limiter.total_waits == 0
    start = clock.t
    limiter.wait_for_capacity(0)  # 4th request in the window must wait
    assert limiter.total_waits == 1
    assert clock.t - start == pytest.approx(60.0)


def test_tpm_paces_a_sequence_of_calls():
    clock = FakeClock()
    limiter = RateLimiter(tokens_per_minute=8_000, clock=clock, sleep=clock.sleep)
    # Two 3,000-token calls fit (6,000 <= 8,000); the third does not and must
    # wait for the window to reset.
    limiter.wait_for_capacity(3_000)
    limiter.wait_for_capacity(3_000)
    assert limiter.total_waits == 0
    limiter.wait_for_capacity(3_000)
    assert limiter.total_waits == 1
    assert clock.t == pytest.approx(60.0)


def test_oversized_single_request_proceeds_after_reset_not_deadlock(caplog):
    """The actual Groq free-tier case: one sample is ~14,251 tokens against
    an 8,000 TPM ceiling. A single such request can never fit any window, so
    it must be let through after one window reset (with a warning) instead
    of blocking forever."""
    clock = FakeClock()
    limiter = RateLimiter(tokens_per_minute=8_000, clock=clock, sleep=clock.sleep)
    with caplog.at_level("WARNING"):
        limiter.wait_for_capacity(14_251)
    assert clock.t == pytest.approx(60.0)  # waited exactly one window, not forever
    assert limiter.total_waits == 1
    assert any("exceed" in rec.message for rec in caplog.records)


def test_retry_after_header_is_read_from_sdk_exception():
    class FakeResponse:
        headers = {"retry-after": "7"}

    exc = types.SimpleNamespace(response=FakeResponse())
    assert _retry_after_seconds(exc) == pytest.approx(7.0)


def test_classify_429_flags_daily_quota_by_keyword():
    exc = types.SimpleNamespace(
        code="rate_limit_exceeded",
        type="requests",
        body={"error": {"message": "You have exceeded your daily token quota"}},
        message="You have exceeded your daily token quota",
        response=None,
    )
    result = _classify_429(exc, provider="groq")
    assert isinstance(result, QuotaExhaustedError)


def test_classify_429_flags_daily_quota_by_long_retry_after():
    class FakeResponse:
        headers = {"retry-after": "7200"}  # 2 hours -- not a per-minute throttle

    exc = types.SimpleNamespace(
        code=None, type=None, body=None, message="rate limited", response=FakeResponse()
    )
    result = _classify_429(exc, provider="groq")
    assert isinstance(result, QuotaExhaustedError)


def test_classify_429_ordinary_rate_limit_is_not_quota_exhaustion():
    class FakeResponse:
        headers = {"retry-after": "2"}

    exc = types.SimpleNamespace(
        code="rate_limit_exceeded",
        type="requests",
        body=None,
        message="Rate limit reached, please retry shortly",
        response=FakeResponse(),
    )
    assert _classify_429(exc, provider="groq") is None


# ---------------------------------------------------------------------------
# 4. Feasibility check
# ---------------------------------------------------------------------------

GROQ_FREE = dict(
    requests_per_minute=30,
    tokens_per_minute=8_000,
    requests_per_day=1_000,
    tokens_per_day=200_000,
)


@pytest.mark.parametrize("n_samples", [14, 50, 150])
def test_feasibility_reports_wallclock_for_groq_free_tier(n_samples):
    est = estimate_run_feasibility(
        n_samples,
        requests_per_sample=9,
        tokens_per_sample=14_251,
        **GROQ_FREE,
    )
    assert est.total_requests == n_samples * 9
    assert est.estimated_minutes > 0


def test_feasibility_flags_150_samples_as_multi_day():
    est = estimate_run_feasibility(
        150, requests_per_sample=9, tokens_per_sample=14_251, **GROQ_FREE
    )
    assert est.fits_in_one_day is False
    assert est.estimated_days > 1


def test_feasibility_14_samples_fits_more_easily_than_150():
    small = estimate_run_feasibility(
        14, requests_per_sample=9, tokens_per_sample=14_251, **GROQ_FREE
    )
    large = estimate_run_feasibility(
        150, requests_per_sample=9, tokens_per_sample=14_251, **GROQ_FREE
    )
    assert small.estimated_minutes < large.estimated_minutes


# ---------------------------------------------------------------------------
# 5. Prices
# ---------------------------------------------------------------------------


def test_resolve_price_for_new_models():
    prices = load_price_table()
    assert resolve_price("openai/gpt-oss-120b", prices) == (0.15, 0.60)
    assert resolve_price("llama-3.3-70b-versatile", prices) == (0.59, 0.79)
    assert resolve_price("llama-3.1-8b-instant", prices) == (0.05, 0.08)
    assert resolve_price("meta-llama/llama-3.1-8b-instruct", prices) == (0.02, 0.05)


def test_resolve_price_unknown_model_still_raises():
    prices = load_price_table()
    with pytest.raises(UnknownModelPrice):
        resolve_price("totally-unpriced-model-xyz", prices)

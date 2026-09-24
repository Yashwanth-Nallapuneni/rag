"""Hard-cash guard for anything in this package that calls a paid LLM.

The user's budget is $5 total, non-negotiable, and a run that overspends or
reports a metric that was never actually computed is unacceptable. This
module exists to make both failures structurally hard:

  * `estimate_run_cost` gives a pre-flight dollar figure BEFORE any call is
    made, so a caller can refuse to start.
  * `CostTracker` accounts every call as it happens and raises the moment
    accumulated spend passes `max_usd` -- a runaway run aborts mid-flight,
    not after the invoice arrives.

Every dollar figure this module produces is labelled measured or estimated
(`CostTracker.as_dict()["measured_calls"]` / `["estimated_calls"]`). A
"measured" call means the token counts came from the provider's own
`response.usage` (a real API's real usage field). An "estimated" call means
no usage was available and the count was reconstructed with
`ragpipe.tokenization.count_tokens` on the prompt/response text -- an
approximation, never presented as a bill.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any

# Per-million-token prices (input, output), USD. This is the table of models
# this project may plausibly call. It is NOT a live pricing feed -- prices
# drift, so it must be overridable without a code change (see
# `load_price_table`).
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5-mini": (0.25, 2.00),
    # Groq models confirmed available on a real free-tier key (models.list()):
    # llama-3.3-70b-versatile is NOT, despite appearing in pricing write-ups,
    # so do not assume a model exists just because it has a published price.
    # These two figures are UNVERIFIED estimates placed here only so the cost
    # guard has something to work with -- confirm against Groq's live pricing
    # before any paid run, or override with RAGPIPE_EVAL_PRICES.
    # Verified against openrouter.ai/api/v1/models on 2026-09-23. The
    # earlier "estimates" here were 2-5x LOW -- a cost guard fed a low price
    # is a cost guard that does not guard.
    "qwen/qwen3.8-27b": (0.42, 3.00),
    "openai/gpt-oss-20b": (0.018, 0.09),
    "meta-llama/llama-3.3-70b-instruct": (0.10, 0.32),
    "gpt-5-nano": (0.05, 0.40),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "mock": (0.0, 0.0),
    # Groq published pricing (groq.com/pricing), per-million tokens. Groq's
    # FREE tier is actually $0 -- but pricing it at $0 here would make the
    # cost guard useless (BudgetExceeded never fires, every run "free"), so
    # these are the PAID on-demand figures used as the estimate basis. A user
    # who is knowingly on the free tier can override to (0, 0) via
    # RAGPIPE_EVAL_PRICES; what they cannot get from us for free is a cost
    # guard that silently stops guarding.
    "openai/gpt-oss-120b": (0.15, 0.60),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    # OpenRouter default (see providers/llm/openrouter.py for why this model
    # was chosen as the default). OpenRouter's per-model prices are set by
    # each upstream provider and change frequently, and no live pricing feed
    # was available while wiring this in -- verify at openrouter.ai/models
    # before relying on this for a real budget and override via
    # RAGPIPE_EVAL_PRICES if it has drifted.
    "meta-llama/llama-3.1-8b-instruct": (0.02, 0.05),
}

PRICE_ENV_VAR = "RAGPIPE_EVAL_PRICES"


class BudgetExceeded(RuntimeError):
    """Raised the moment tracked (or estimated) spend would pass the cap."""

    def __init__(self, spent_usd: float, max_usd: float, *, context: str = ""):
        self.spent_usd = spent_usd
        self.max_usd = max_usd
        self.context = context
        msg = (
            f"budget exceeded: ${spent_usd:.4f} spent against a ${max_usd:.2f} cap"
        )
        if context:
            msg += f" ({context})"
        super().__init__(msg)


class UnknownModelPrice(ValueError):
    """A model has no price entry and no default -- refuse to pretend it's free."""

    def __init__(self, model: str):
        self.model = model
        super().__init__(
            f"no price entry for model '{model}'. Add one via the "
            f"{PRICE_ENV_VAR} env var (JSON: {{\"model\": [in_per_m, out_per_m]}}) "
            "or pass `prices={...}` explicitly -- silently treating an unknown "
            "model as free would misstate the bill."
        )


def _env_price_overrides() -> dict[str, tuple[float, float]]:
    raw = os.environ.get(PRICE_ENV_VAR)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{PRICE_ENV_VAR} is not valid JSON: {exc}") from exc
    out: dict[str, tuple[float, float]] = {}
    for model, spec in data.items():
        if isinstance(spec, dict):
            out[model] = (float(spec["input"]), float(spec["output"]))
        else:
            in_p, out_p = spec
            out[model] = (float(in_p), float(out_p))
    return out


def load_price_table(
    overrides: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Defaults, then `RAGPIPE_EVAL_PRICES` env, then explicit overrides win."""
    table = dict(DEFAULT_PRICES)
    table.update(_env_price_overrides())
    if overrides:
        table.update(
            {k: (float(v[0]), float(v[1])) for k, v in overrides.items()}
        )
    return table


def _strip_provider_prefix(model: str) -> str:
    """Answer.model / LLMProvider identify as 'provider:model' -- match on
    the model part, falling back to the full string for a direct match."""
    return model.split(":", 1)[1] if ":" in model else model


def resolve_price(model: str, prices: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """(input_per_million, output_per_million) USD for `model`.

    Tries an exact match first (so a caller can key by 'provider:model' if it
    wants to), then the model name with any 'provider:' prefix stripped, then
    a prefix match against the table (handles dated variants like
    'gpt-5-mini-2025-11-01'). Raises `UnknownModelPrice` rather than assuming
    free -- an unpriced model must never look like a $0 model.
    """
    if model in prices:
        return prices[model]
    bare = _strip_provider_prefix(model)
    if bare in prices:
        return prices[bare]
    for key, price in prices.items():
        if bare.startswith(key) or key.startswith(bare):
            return price
    # The 'provider:model' label carries a real model name for real
    # providers (openai/anthropic/ollama), but our 'mock' provider is priced
    # as a whole regardless of the offline model string it reports (e.g.
    # 'mock:default', 'mock-extractive-v1') -- so fall back to matching the
    # provider prefix itself against the table.
    if ":" in model:
        provider = model.split(":", 1)[0]
        if provider in prices:
            return prices[provider]
    raise UnknownModelPrice(model)


@dataclass
class CostEstimate:
    """A pre-flight guess, not a bill. Every token count here is an
    assumption -- read `assumptions` before trusting the total."""

    n_samples: int
    generation_model: str
    judge_model: str
    metrics: list[str]
    generation_usd: float
    judge_usd: float
    total_usd: float
    assumptions: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "generation_model": self.generation_model,
            "judge_model": self.judge_model,
            "metrics": self.metrics,
            "generation_usd": round(self.generation_usd, 4),
            "judge_usd": round(self.judge_usd, 4),
            "total_usd": round(self.total_usd, 4),
            "estimated": True,
            "assumptions": self.assumptions,
        }


def estimate_run_cost(
    n_samples: int,
    model: str,
    *,
    judge_model: str | None = None,
    metrics: list[str] | None = None,
    # Measured on this corpus, not guessed: the rendered answer prompt averages
    # 2840 input tokens (top_k=5 over ~507-token chunks, mean context 2538).
    # A pre-flight estimate that runs LOW is the dangerous direction when the
    # user has a hard budget, so these lean on the measured figures.
    avg_gen_input_tokens: int = 2840,
    avg_gen_output_tokens: int = 250,
    avg_judge_input_tokens: int = 1100,
    avg_judge_output_tokens: int = 180,
    judge_calls_per_metric: int = 2,
    prices: dict[str, tuple[float, float]] | None = None,
) -> CostEstimate:
    """Pre-flight dollar estimate for an n-sample RAGAS run.

    Two cost sources, both heuristic:
      * generation -- one LLM call per sample to answer the question. Token
        counts are guessed from the pipeline's own defaults (context budget
        ~6000 tokens, so a filled context plus prompt overhead lands well
        under that; answers are short).
      * judge -- RAGAS issues multiple LLM calls per metric per sample
        (e.g. faithfulness extracts claims, then verifies each one;
        context_precision/recall grade each retrieved passage). There is no
        way to know the exact count before running, so `judge_calls_per_metric`
        is a deliberately round number, not a measurement.

    This is why it is called an *estimate*: change the assumptions and the
    number changes. It exists to let a caller refuse an obviously-too-large
    run before spending anything, not to predict the bill to the cent.
    """
    prices = load_price_table(prices)
    metrics = list(metrics or ["faithfulness", "answer_relevancy", "context_precision", "context_recall"])
    judge_model = judge_model or model

    gen_in_p, gen_out_p = resolve_price(model, prices)
    generation_usd = n_samples * (
        avg_gen_input_tokens * gen_in_p / 1e6 + avg_gen_output_tokens * gen_out_p / 1e6
    )

    judge_in_p, judge_out_p = resolve_price(judge_model, prices)
    judge_calls = n_samples * len(metrics) * judge_calls_per_metric
    judge_usd = judge_calls * (
        avg_judge_input_tokens * judge_in_p / 1e6 + avg_judge_output_tokens * judge_out_p / 1e6
    )

    return CostEstimate(
        n_samples=n_samples,
        generation_model=model,
        judge_model=judge_model,
        metrics=metrics,
        generation_usd=generation_usd,
        judge_usd=judge_usd,
        total_usd=generation_usd + judge_usd,
        assumptions={
            "avg_gen_input_tokens": avg_gen_input_tokens,
            "avg_gen_output_tokens": avg_gen_output_tokens,
            "avg_judge_input_tokens": avg_judge_input_tokens,
            "avg_judge_output_tokens": avg_judge_output_tokens,
            "judge_calls_per_metric": judge_calls_per_metric,
            "estimated_judge_calls": judge_calls,
            "note": (
                "heuristic pre-flight guess, not a measurement -- RAGAS's actual "
                "call count per metric varies with answer/context length"
            ),
        },
    )


def check_preflight(estimate: CostEstimate, max_usd: float | None) -> None:
    """Refuse to START a run whose estimate already exceeds the cap."""
    if max_usd is not None and estimate.total_usd > max_usd:
        raise BudgetExceeded(
            estimate.total_usd, max_usd, context="pre-flight estimate, before any call was made"
        )


@dataclass
class FeasibilityEstimate:
    """Pre-flight wall-clock/quota estimate for a rate-limited run, so a
    caller can hear "150 pairs needs ~11 days" before spending hours getting
    throttled to find that out the slow way."""

    n_samples: int
    requests_per_sample: float
    tokens_per_sample: float
    total_requests: int
    total_tokens: int
    # Time actually spent issuing calls, bounded only by the per-minute
    # ceilings. Distinct from `estimated_minutes`, which also counts the dead
    # time spent waiting for a daily allowance to reset -- conflating the two
    # reports "14 samples takes 24 hours" when it is 25 minutes of calling
    # that happens to consume a whole day's quota.
    active_minutes: float
    estimated_minutes: float
    limiting_factor: str
    quota_days: float
    fits_in_one_day: bool
    daily_request_cap_hit: bool
    daily_token_cap_hit: bool
    assumptions: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_hours(self) -> float:
        return self.estimated_minutes / 60.0

    @property
    def estimated_days(self) -> float:
        return self.estimated_minutes / (60.0 * 24.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "estimated_minutes": round(self.estimated_minutes, 1),
            "estimated_hours": round(self.estimated_hours, 2),
            "estimated_days": round(self.estimated_days, 2),
            "limiting_factor": self.limiting_factor,
            "fits_in_one_day": self.fits_in_one_day,
            "daily_request_cap_hit": self.daily_request_cap_hit,
            "daily_token_cap_hit": self.daily_token_cap_hit,
            "assumptions": self.assumptions,
        }


def estimate_run_feasibility(
    n_samples: int,
    *,
    requests_per_sample: float = 9.0,
    tokens_per_sample: float = 14_251.0,
    requests_per_minute: int | None = None,
    tokens_per_minute: int | None = None,
    requests_per_day: int | None = None,
    tokens_per_day: int | None = None,
) -> FeasibilityEstimate:
    """Estimate wall-clock duration and daily-cap feasibility for `n_samples`
    under the given rate limits, BEFORE any call is made.

    This is deliberately independent of `RateLimiter`: it does not simulate
    call-by-call pacing, it computes the same two bottlenecks a real run
    would hit --

      * per-minute ceilings (RPM/TPM) bound how fast requests can go out at
        all, so wall-clock time is `max(requests_needed / rpm,
        tokens_needed / tpm)` minutes;
      * per-day ceilings (RPD/TPD) bound how much can be done before the
        provider simply refuses more for the day, regardless of how patient
        the caller is willing to be.

    Defaults (`requests_per_sample=9`, `tokens_per_sample=14251`) are this
    project's own measured RAGAS generation+judge call pattern; override them
    for a different pipeline/metric set.
    """
    total_requests = int(round(n_samples * requests_per_sample))
    total_tokens = int(round(n_samples * tokens_per_sample))

    minutes_candidates: dict[str, float] = {}
    if requests_per_minute:
        minutes_candidates["requests_per_minute"] = total_requests / requests_per_minute
    if tokens_per_minute:
        minutes_candidates["tokens_per_minute"] = total_tokens / tokens_per_minute

    if minutes_candidates:
        limiting_factor, active_minutes = max(minutes_candidates.items(), key=lambda kv: kv[1])
    else:
        limiting_factor, active_minutes = "unbounded", 0.0

    daily_request_cap_hit = bool(requests_per_day and total_requests > requests_per_day)
    daily_token_cap_hit = bool(tokens_per_day and total_tokens > tokens_per_day)

    # If a daily cap is tighter than the per-minute pacing already implies,
    # the run cannot finish faster than "however many days it takes to trickle
    # the daily allowance out" -- stretch the estimate to reflect that instead
    # of reporting a number nobody will actually see hit.
    quota_days = 0.0
    stretched: dict[str, float] = {limiting_factor: active_minutes}
    if requests_per_day:
        days = total_requests / requests_per_day
        quota_days = max(quota_days, days)
        stretched["requests_per_day"] = days * 24 * 60
    if tokens_per_day:
        days = total_tokens / tokens_per_day
        quota_days = max(quota_days, days)
        stretched["tokens_per_day"] = days * 24 * 60
    limiting_factor, estimated_minutes = max(stretched.items(), key=lambda kv: kv[1])

    return FeasibilityEstimate(
        n_samples=n_samples,
        requests_per_sample=requests_per_sample,
        tokens_per_sample=tokens_per_sample,
        total_requests=total_requests,
        total_tokens=total_tokens,
        active_minutes=round(active_minutes, 2),
        estimated_minutes=estimated_minutes,
        limiting_factor=limiting_factor,
        quota_days=round(quota_days, 3),
        fits_in_one_day=estimated_minutes <= 24 * 60 and not (
            daily_request_cap_hit or daily_token_cap_hit
        ),
        daily_request_cap_hit=daily_request_cap_hit,
        daily_token_cap_hit=daily_token_cap_hit,
        assumptions={
            "requests_per_minute": requests_per_minute,
            "tokens_per_minute": tokens_per_minute,
            "requests_per_day": requests_per_day,
            "tokens_per_day": tokens_per_day,
            "note": (
                "heuristic pre-flight guess: assumes requests can be paced "
                "back-to-back against the per-minute ceiling with no other "
                "overhead, and that daily caps reset once every 24h"
            ),
        },
    )


@dataclass
class _ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    measured_calls: int = 0
    estimated_calls: int = 0
    usd: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
            "measured_calls": self.measured_calls,
            "estimated_calls": self.estimated_calls,
            "usd": round(self.usd, 6),
        }


class CostTracker:
    """Accumulates real spend across models and enforces a hard cap.

    `.add()` is called once per completed LLM call, after `response.usage`
    (or a token-count fallback) is known. Cost already incurred by that call
    cannot be undone -- an API request that already happened cannot be
    un-billed -- so `.add()` records it and only then raises `BudgetExceeded`
    if the running total has passed `max_usd`. That is what "aborts mid-flight"
    means here: the NEXT call never happens, and the tracker's own total is
    always the true, honest number rather than one that stops updating right
    before a breach.

    Thread-safe: RAGAS's executor can fire LLM calls from multiple threads.
    """

    def __init__(
        self,
        max_usd: float | None = None,
        prices: dict[str, tuple[float, float]] | None = None,
    ):
        self.max_usd = max_usd
        self.prices = load_price_table(prices)
        self._by_model: dict[str, _ModelUsage] = {}
        self._lock = threading.Lock()

    def add(
        self,
        model: str,
        in_tokens: int,
        out_tokens: int,
        *,
        estimated: bool = False,
    ) -> float:
        """Record one call's usage and its dollar cost. Raises
        `BudgetExceeded` if this call pushes the running total past the cap."""
        price_in, price_out = resolve_price(model, self.prices)
        cost = in_tokens * price_in / 1e6 + out_tokens * price_out / 1e6

        with self._lock:
            usage = self._by_model.setdefault(model, _ModelUsage())
            usage.input_tokens += in_tokens
            usage.output_tokens += out_tokens
            usage.calls += 1
            if estimated:
                usage.estimated_calls += 1
            else:
                usage.measured_calls += 1
            usage.usd += cost
            total = self.total_usd

        if self.max_usd is not None and total > self.max_usd:
            raise BudgetExceeded(
                total, self.max_usd, context=f"exceeded on a call to '{model}'"
            )
        return cost

    @property
    def total_usd(self) -> float:
        return sum(u.usd for u in self._by_model.values())

    @property
    def by_model(self) -> dict[str, dict[str, Any]]:
        return {model: usage.as_dict() for model, usage in self._by_model.items()}

    def as_dict(self) -> dict[str, Any]:
        measured = sum(u.measured_calls for u in self._by_model.values())
        estimated = sum(u.estimated_calls for u in self._by_model.values())
        return {
            "total_usd": round(self.total_usd, 6),
            "max_usd": self.max_usd,
            "by_model": self.by_model,
            "measured_calls": measured,
            "estimated_calls": estimated,
            "note": (
                "measured_calls used the provider's own response.usage; "
                "estimated_calls had no usage available and fell back to "
                "ragpipe.tokenization.count_tokens on the prompt/response text"
            ),
        }

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
    "gpt-5-nano": (0.05, 0.40),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "mock": (0.0, 0.0),
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

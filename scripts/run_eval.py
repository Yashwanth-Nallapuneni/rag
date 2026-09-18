#!/usr/bin/env python
"""CLI for the RAGAS evaluation harness.

Spends real money once `--yes` (or an interactive confirmation) is given, so
by design this always prints the pre-flight cost estimate first and never
calls an LLM before that gate. `--dry-run` stops right after the estimate.

Exit codes: 0 = passed every configured threshold; 1 = ran but a metric
missed its threshold (or the run failed); 2 = refused to run (budget/estimate
rejected, bad arguments) -- distinct so CI can tell "the pipeline is worse"
apart from "this invocation was misconfigured".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.eval.cost import BudgetExceeded  # noqa: E402
from ragpipe.eval.ragas_eval import compare_runs, format_summary, run_evaluation  # noqa: E402
from ragpipe.logging_utils import setup_logging  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=None, help="override evaluation.dataset_path")
    p.add_argument("--sample-size", type=int, default=None, help="override evaluation.sample_size")
    p.add_argument("--judge-model", default=None, help="judge LLM model (default: same as generation)")
    p.add_argument("--judge-provider", default=None, help="judge LLM provider (default: same as generation)")
    p.add_argument(
        "--max-usd",
        type=float,
        default=2.00,
        help="hard budget cap in USD for this run (default: $2.00); required to have a default so CI never runs unbounded",
    )
    p.add_argument(
        "--metrics",
        default=None,
        help="comma-separated metric names (default: config's evaluation.metrics)",
    )
    p.add_argument("--out", default=None, help="unused placeholder kept for interface symmetry -- result path is chosen by run_evaluation and printed")
    p.add_argument("--compare", default=None, metavar="OTHER_RESULTS_JSON", help="print a before/after delta table against this earlier result file, then exit")
    p.add_argument("--dry-run", action="store_true", help="print the pre-flight cost estimate and sample count; call nothing")
    p.add_argument("--yes", action="store_true", help="skip the interactive spend confirmation")
    p.add_argument("--rerank", dest="rerank", action="store_true", default=None)
    p.add_argument("--no-rerank", dest="rerank", action="store_false")
    p.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default=None, help="retrieval.mode override")
    p.add_argument("--env", default=None, help="RAGPIPE_ENV layer to load")
    p.add_argument("-q", "--quiet", action="store_true", help="suppress per-sample progress logging")
    return p.parse_args(argv)


def _settings(args: argparse.Namespace):
    overrides: dict = {}
    eval_over: dict = {}
    if args.dataset:
        eval_over["dataset_path"] = args.dataset
    if args.sample_size is not None:
        eval_over["sample_size"] = args.sample_size
    if args.metrics:
        eval_over["metrics"] = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if eval_over:
        overrides["evaluation"] = eval_over

    if args.rerank is not None:
        overrides["rerank"] = {"enabled": args.rerank}
    if args.mode:
        overrides["retrieval"] = {"mode": args.mode}

    return load_settings(env=args.env, overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(level="WARNING" if args.quiet else "INFO")

    settings = _settings(args)
    metric_names = list(settings.evaluation.metrics)

    result = run_evaluation(
        settings,
        sample_size=args.sample_size,
        max_usd=None,  # pre-flight estimate first; real cap applied after confirmation
        metrics=metric_names,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        progress=not args.quiet,
        dry_run=True,
    )

    est = result["estimate"]
    print(f"Pre-flight estimate: {result['n_samples']} sample(s) "
          f"({result['n_available']} available in dataset)")
    print(f"  generation model: {est['generation_model']}")
    print(f"  judge model:      {est['judge_model']}")
    print(f"  metrics:          {est['metrics']}")
    print(f"  estimated generation cost: ${est['generation_usd']:.4f}")
    print(f"  estimated judge cost:      ${est['judge_usd']:.4f}")
    print(f"  ESTIMATED TOTAL:           ${est['total_usd']:.4f}   (cap: ${args.max_usd:.2f})")
    print(f"  assumptions: {est['assumptions']}")

    if args.dry_run:
        return 0

    if est["total_usd"] > args.max_usd:
        print(
            f"\nREFUSED: estimated ${est['total_usd']:.4f} exceeds --max-usd ${args.max_usd:.2f}. "
            "Lower --sample-size, pick a cheaper --judge-model, or raise --max-usd.",
            file=sys.stderr,
        )
        return 2

    if not args.yes:
        try:
            answer = input(f"\nProceed and spend up to ${args.max_usd:.2f}? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            print("Aborted; no LLM call was made.", file=sys.stderr)
            return 2

    try:
        result = run_evaluation(
            settings,
            sample_size=args.sample_size,
            max_usd=args.max_usd,
            metrics=metric_names,
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            progress=not args.quiet,
        )
    except BudgetExceeded as exc:
        print(f"\nABORTED MID-RUN: {exc}", file=sys.stderr)
        return 2

    print()
    print(format_summary(result))
    print(f"Result written to: {result['result_path']}")
    print(f"Summary written to: {result['summary_path']}")

    if args.compare:
        print()
        print(compare_runs(args.compare, result["result_path"]))

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

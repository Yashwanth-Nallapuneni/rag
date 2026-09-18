#!/usr/bin/env python3
"""Compare an eval result JSON (produced by scripts/run_eval.py) against the
configured quality thresholds, and optionally against a baseline result to
catch regressions.

This is the script that turns a RAGAS run into a CI pass/fail decision. It
is deliberately standalone (only stdlib + PyYAML/pydantic already in
requirements.txt) so it works identically in CI and on a laptop:

    python scripts/ci_gate.py eval_results/latest.json
    python scripts/ci_gate.py eval_results/latest.json --baseline eval_results/baseline.json

Exit codes (checked by callers, including the eval.yml workflow):
    0  all metrics at/above threshold, no disallowed regression
    1  at least one metric below its configured threshold
    2  a metric regressed vs. the baseline by more than --max-regression,
       even though it may still be above the absolute threshold -- this is
       the "blocked because it regressed faithfulness from 0.91 to 0.84"
       case from the spec
    3  the results file (or, if given, the baseline file) is missing or
       is not parseable. A missing/broken results file must NEVER exit 0:
       "no result" is not the same thing as "passed", and if we let a
       missing file look like success, a CI misconfiguration (eval job
       skipped, path typo, job killed before writing output) would show as
       a green quality badge without any evaluation having actually run.

Result-file shape: `_extract_scores` reads the real schema written by
`ragpipe.eval.ragas_eval.run_evaluation` --
`result["ragas"]["aggregate"]` (a {metric_name: score} dict, values may be
`None` when RAGAS couldn't score a run -- those are dropped, not treated as
0.0) plus `result["refusal_accuracy"]["accuracy"]`. As a fallback (useful
for hand-built fixtures and any future reshaping of run_eval.py's output),
it also accepts a flatter top-level "scores"/"metrics"/"aggregate" dict, or
bare top-level numeric fields named after known metrics. If none of those
shapes match, the file is treated as malformed (exit 3) rather than
silently reporting no scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Known metric names we know how to threshold-check, in a stable display
# order. Pulled from EvalThresholds in src/ragpipe/config.py -- see below,
# where we load the *actual* threshold values from settings rather than
# hardcoding them here, so a config change is automatically honored.
KNOWN_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "refusal_accuracy",
]

EXIT_OK = 0
EXIT_THRESHOLD_FAIL = 1
EXIT_REGRESSION_FAIL = 2
EXIT_BAD_INPUT = 3


class BadInput(Exception):
    """Raised for a missing or unparseable results/baseline file."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BadInput(f"file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BadInput(f"could not read {path}: {exc}") from exc
    if not text.strip():
        raise BadInput(f"file is empty: {path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BadInput(f"malformed JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BadInput(f"expected a JSON object at the top level of {path}")
    return data


def _extract_scores(data: dict[str, Any]) -> dict[str, float]:
    """Pull a flat {metric_name: score} dict out of a run_eval.py result.

    Tries, in order:
      1. the real shape written by ragpipe.eval.ragas_eval.run_evaluation --
         result["ragas"]["aggregate"] merged with
         result["refusal_accuracy"]["accuracy"]
      2. a nested top-level "scores"/"metrics"/"aggregate"/"aggregate_scores"
         dict (fallback for hand-built fixtures / schema drift)
      3. bare top-level fields named after a known metric

    Raises BadInput if nothing recognizable is found -- an eval result with
    zero readable metrics is as useless as a missing file and must not be
    treated as a pass.
    """
    ragas = data.get("ragas")
    if isinstance(ragas, dict) and isinstance(ragas.get("aggregate"), dict):
        scores = {
            str(name): float(val)
            for name, val in ragas["aggregate"].items()
            if isinstance(val, (int, float)) and not isinstance(val, bool)
        }
        refusal = data.get("refusal_accuracy")
        if isinstance(refusal, dict) and isinstance(refusal.get("accuracy"), (int, float)):
            scores["refusal_accuracy"] = float(refusal["accuracy"])
        if scores:
            return scores

    for key in ("scores", "metrics", "aggregate", "aggregate_scores"):
        nested = data.get(key)
        if isinstance(nested, dict):
            scores = {
                str(name): float(val)
                for name, val in nested.items()
                if isinstance(val, (int, float)) and not isinstance(val, bool)
            }
            if scores:
                return scores

    flat = {
        name: float(data[name])
        for name in KNOWN_METRICS
        if name in data and isinstance(data[name], (int, float)) and not isinstance(data[name], bool)
    }
    if flat:
        return flat

    raise BadInput(
        "no recognizable per-metric scores found (looked for a 'scores' / "
        "'metrics' / 'aggregate' / 'aggregate_scores' object, or top-level "
        f"fields named after known metrics: {', '.join(KNOWN_METRICS)})"
    )


def _load_thresholds() -> dict[str, float]:
    """Load the real, current thresholds from settings -- never hardcode a
    second copy that could silently drift from src/ragpipe/config.py."""
    from ragpipe.config import load_settings

    thresholds = load_settings().evaluation.thresholds
    return thresholds.model_dump()


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _threshold_table(scores: dict[str, float], thresholds: dict[str, float]) -> tuple[str, bool]:
    """Returns (markdown table, all_pass)."""
    rows = []
    all_pass = True
    checked = [m for m in KNOWN_METRICS if m in scores and m in thresholds]
    extra = [m for m in scores if m not in KNOWN_METRICS]
    for m in checked + sorted(extra):
        score = scores[m]
        threshold = thresholds.get(m)
        if threshold is None:
            rows.append((m, score, None, None, "n/a (no threshold configured)"))
            continue
        delta = score - threshold
        ok = score >= threshold
        all_pass = all_pass and ok
        rows.append((m, score, threshold, delta, "PASS" if ok else "FAIL"))

    if not checked:
        # Nothing in the result overlaps with any configured threshold.
        raise BadInput(
            "none of the metrics in the results file have a configured "
            f"threshold (checked: {', '.join(KNOWN_METRICS)})"
        )

    lines = ["| Metric | Score | Threshold | Delta | Result |", "|---|---|---|---|---|"]
    for m, score, threshold, delta, status in rows:
        thr_s = _fmt(threshold) if threshold is not None else "-"
        delta_s = f"{delta:+.4f}" if delta is not None else "-"
        lines.append(f"| {m} | {_fmt(score)} | {thr_s} | {delta_s} | {status} |")
    return "\n".join(lines), all_pass


def _regression_table(
    scores: dict[str, float], baseline_scores: dict[str, float], max_regression: float
) -> tuple[str, bool, list[str]]:
    """Returns (markdown table, no_regression, list of human-readable regression messages)."""
    shared = [m for m in KNOWN_METRICS if m in scores and m in baseline_scores]
    shared += sorted(m for m in scores if m in baseline_scores and m not in KNOWN_METRICS)
    if not shared:
        return "(no metrics in common with baseline; regression check skipped)", True, []

    lines = ["| Metric | Baseline | Current | Delta | Result |", "|---|---|---|---|---|"]
    no_regression = True
    messages = []
    for m in shared:
        base = baseline_scores[m]
        cur = scores[m]
        delta = cur - base
        regressed = delta < -max_regression
        no_regression = no_regression and not regressed
        status = "REGRESSION" if regressed else "ok"
        lines.append(f"| {m} | {_fmt(base)} | {_fmt(cur)} | {delta:+.4f} | {status} |")
        if regressed:
            messages.append(
                f"{m} regressed from {_fmt(base)} to {_fmt(cur)} "
                f"(drop of {-delta:.4f} exceeds --max-regression {max_regression:.4f})"
            )
    return "\n".join(lines), no_regression, messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="path to the eval result JSON from scripts/run_eval.py")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="path to a previous result JSON to compare against for regressions",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.02,
        help="max allowed drop vs. baseline before a metric is flagged as a regression (default: 0.02)",
    )
    args = parser.parse_args(argv)

    summary_parts: list[str] = ["# Answer-quality gate\n"]

    # --- primary results: missing/malformed is always exit 3, no exceptions ---
    try:
        data = _load_json(args.results)
        scores = _extract_scores(data)
        thresholds = _load_thresholds()
        threshold_md, thresholds_pass = _threshold_table(scores, thresholds)
    except BadInput as exc:
        msg = f"FAILED TO EVALUATE RESULTS: {exc}"
        print(f"\n{msg}\n", file=sys.stderr)
        summary_parts.append(f"**{msg}**\n")
        _write_summary("\n".join(summary_parts))
        return EXIT_BAD_INPUT

    print(f"\nResults file: {args.results}")
    print(threshold_md)
    summary_parts.append(f"Results file: `{args.results}`\n")
    summary_parts.append(threshold_md + "\n")

    # --- optional baseline comparison ---
    regression_pass = True
    regression_messages: list[str] = []
    if args.baseline is not None:
        if not args.baseline.exists():
            note = f"No baseline found at {args.baseline} -- skipping regression check (expected on the first run)."
            print(f"\n{note}")
            summary_parts.append(f"\n{note}\n")
        else:
            try:
                baseline_data = _load_json(args.baseline)
                baseline_scores = _extract_scores(baseline_data)
            except BadInput as exc:
                # A *present but corrupt* baseline is a real problem (unlike
                # a merely absent one), so this is a hard failure too.
                msg = f"FAILED TO READ BASELINE: {exc}"
                print(f"\n{msg}\n", file=sys.stderr)
                summary_parts.append(f"**{msg}**\n")
                _write_summary("\n".join(summary_parts))
                return EXIT_BAD_INPUT

            regression_md, regression_pass, regression_messages = _regression_table(
                scores, baseline_scores, args.max_regression
            )
            print(f"\nRegression vs. baseline ({args.baseline}):")
            print(regression_md)
            summary_parts.append(f"\n## Regression vs. baseline (`{args.baseline}`)\n")
            summary_parts.append(regression_md + "\n")
            for m in regression_messages:
                print(f"  REGRESSION: {m}")
                summary_parts.append(f"\n> **Blocked:** {m}\n")

    # --- verdict ---
    # Regression takes priority in the exit code: a metric that dropped by
    # more than the tolerance is the more specific, more actionable story
    # ("this PR regressed faithfulness from 0.91 to 0.84") even when the
    # absolute threshold is still technically met. Both tables are always
    # printed above regardless of which check ends up deciding the exit code.
    if not regression_pass:
        verdict = "FAIL (regression)"
        code = EXIT_REGRESSION_FAIL
    elif not thresholds_pass:
        verdict = "FAIL (below threshold)"
        code = EXIT_THRESHOLD_FAIL
    else:
        verdict = "PASS"
        code = EXIT_OK

    print(f"\nVerdict: {verdict}\n")
    summary_parts.append(f"\n**Verdict: {verdict}**\n")
    _write_summary("\n".join(summary_parts))
    return code


def _write_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write(markdown)
        fh.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())

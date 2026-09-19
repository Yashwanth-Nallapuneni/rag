#!/usr/bin/env python
"""Measure the cross-encoder relevance gate against a QA dataset.

The gate (citation.min_relevance_score) refuses a question before generation
when the best reranked passage scores below it. This script runs retrieval +
rerank only -- no LLM call, no API key, no cost -- and reports, per candidate
threshold, how many answerable questions it would wrongly refuse and how many
unanswerable ones it would let through to generation.

It reads human-verified pairs by default. `--allow-unverified` falls back to
the raw drafted dataset, and the output says so: drafted questions are written
FROM their target chunk, so they share its vocabulary and score higher than a
real user's question would. A threshold picked on unverified drafts is biased
toward being too strict about nothing and too lenient about paraphrase.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.eval.golden import load_dataset, load_verified  # noqa: E402
from ragpipe.index.builder import get_store  # noqa: E402
from ragpipe.retrieval import get_retriever  # noqa: E402

DEFAULT_THRESHOLDS = [-11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, 0.0]


def _quantiles(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    xs = sorted(xs)
    pick = lambda q: xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]  # noqa: E731
    return {"min": xs[0], "p10": pick(0.10), "p25": pick(0.25), "median": statistics.median(xs),
            "p75": pick(0.75), "max": xs[-1], "n": len(xs)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--allow-unverified", action="store_true",
                    help="use the raw drafted dataset when nothing is verified yet")
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--show", type=int, default=5, help="print the N lowest-scoring answerable questions")
    args = ap.parse_args(argv)

    settings = load_settings()
    dataset = args.dataset or settings.evaluation.dataset_path
    pairs, verified = load_verified(dataset), True
    if not pairs:
        if not args.allow_unverified:
            print(f"no human-verified pairs in {dataset}; review them first "
                  "(scripts/review_golden.py) or pass --allow-unverified", file=sys.stderr)
            return 2
        pairs, verified = load_dataset(dataset), False
    if not pairs:
        print(f"no pairs in {dataset}", file=sys.stderr)
        return 2
    if not settings.rerank.enabled:
        print("rerank is disabled; the relevance gate has no scores to act on", file=sys.stderr)
        return 2

    retriever = get_retriever(settings, get_store(settings))
    rows = []
    for qa in pairs:
        hits = retriever.retrieve(qa.question)
        scores = [h.rerank_score for h in hits if h.rerank_score is not None]
        rows.append({"id": qa.id, "question": qa.question, "unanswerable": qa.unanswerable,
                     "category": qa.category, "top_score": max(scores) if scores else None})

    ans = [r["top_score"] for r in rows if not r["unanswerable"] and r["top_score"] is not None]
    una = [r["top_score"] for r in rows if r["unanswerable"] and r["top_score"] is not None]
    current = settings.citation.min_relevance_score

    table = []
    for t in sorted(set(args.thresholds + ([current] if current is not None else []))):
        fr = sum(s < t for s in ans)
        fp = sum(s >= t for s in una)
        table.append({"threshold": t, "false_refusals": fr, "false_refusal_rate": fr / len(ans) if ans else None,
                      "unanswerable_passed": fp, "unanswerable_pass_rate": fp / len(una) if una else None})

    print(f"relevance gate calibration -- {len(rows)} pairs "
          f"({'HUMAN-VERIFIED' if verified else 'UNVERIFIED drafts: biased toward high scores'})")
    print(f"reranker: {settings.rerank.model}   current gate: {current}")
    print(f"answerable   top-score: {_quantiles(ans)}")
    print(f"unanswerable top-score: {_quantiles(una)}")
    print(f"\n{'threshold':>9}  {'false refusals':>16}  {'unanswerable passed':>21}")
    for r in table:
        mark = "  <- current" if r["threshold"] == current else ""
        fr = f"{r['false_refusals']}/{len(ans)}"
        fp = f"{r['unanswerable_passed']}/{len(una)}"
        print(f"{r['threshold']:>9.1f}  {fr:>16}  {fp:>21}{mark}")
    print("\n'unanswerable passed' is not a failure by itself: citation enforcement is the "
          "second line of defence. The gate's job is the confidently-quoted irrelevant passage.")

    low = sorted((r for r in rows if not r["unanswerable"] and r["top_score"] is not None),
                 key=lambda r: r["top_score"])[: args.show]
    if low:
        print("\nlowest-scoring answerable questions:")
        for r in low:
            print(f"  {r['top_score']:7.2f}  [{r['category']}] {r['question'][:110]}")

    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"gate_calibration_{stamp}.json"
    out.write_text(json.dumps({
        "timestamp": stamp, "dataset_path": str(dataset), "pairs_human_verified": verified,
        "caveat": None if verified else "drafted questions share vocabulary with their target chunk; scores are optimistic",
        "reranker": settings.rerank.model, "current_gate": current,
        "answerable": _quantiles(ans), "unanswerable": _quantiles(una),
        "thresholds": table, "rows": rows,
    }, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

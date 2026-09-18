#!/usr/bin/env python
"""Draft candidate golden QA pairs from the real corpus.

This writes *draft* pairs only -- nothing here is verified. Run
`scripts/review_golden.py` afterward to have a human approve, edit, or
reject each one before the eval harness is allowed to touch it.

Safe to re-run: `--append` adds new candidates to an existing dataset file
without touching pairs a human has already decided on (the review ledger is
keyed by pair id and is never modified by this script).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.eval.golden import (  # noqa: E402
    DEFAULT_CATEGORIES,
    dataset_stats,
    default_review_path,
    draft_candidates,
    load_dataset,
    load_review_ledger,
    save_dataset,
    validate_dataset,
)
from ragpipe.ingest.pipeline import read_chunks  # noqa: E402
from ragpipe.providers import get_llm  # noqa: E402


def _provider_choices() -> list[str]:
    from typing import get_args

    from ragpipe.config import LLMConfig

    return list(get_args(LLMConfig.model_fields["provider"].annotation))

# Rough hosted-provider pricing for a cost estimate before spending real
# money; not billed anywhere, just a heads-up printed to the terminal.
_USD_PER_1K_TOKENS = {
    "anthropic": (0.003, 0.015),  # input, output -- Claude Sonnet-class
    "openai": (0.0025, 0.010),  # GPT-4o-class
}


def _parse_args() -> argparse.Namespace:
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150, help="candidates to draft")
    ap.add_argument("--unanswerable-ratio", type=float, default=0.15)
    ap.add_argument(
        "--out", type=Path, default=None, help="default: settings.evaluation.dataset"
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        # Derived from the config Literal rather than hardcoded, so adding a
        # provider cannot leave this list silently stale (it already did once).
        "--provider", choices=_provider_choices(), default=None
    )
    ap.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="default: <processed_dir>/chunks.jsonl",
    )
    ap.add_argument(
        "--categories", nargs="+", default=list(DEFAULT_CATEGORIES)
    )
    ap.add_argument("--dry-run", action="store_true", help="print candidates, write nothing")
    ap.add_argument(
        "--append", action="store_true", help="add to an existing dataset file, keeping verified pairs"
    )
    args = ap.parse_args()

    args.out = args.out or settings.evaluation.dataset
    args.chunks = args.chunks or (settings.corpus.processed_path / "chunks.jsonl")
    return args


def main() -> int:
    args = _parse_args()
    overrides = {"llm": {"provider": args.provider}} if args.provider else {}
    settings = load_settings(overrides=overrides)

    if not args.chunks.exists():
        print(f"no chunks at {args.chunks}; run `ragpipe ingest` first", file=sys.stderr)
        return 1
    chunks = read_chunks(args.chunks)
    print(f"loaded {len(chunks)} chunks from {len(set(c.doc_id for c in chunks))} documents")

    llm = get_llm(settings)
    candidates = draft_candidates(
        settings,
        chunks,
        n=args.n,
        categories=tuple(args.categories),
        llm=llm,
        unanswerable_ratio=args.unanswerable_ratio,
        seed=args.seed,
    )

    existing: list = []
    review_path = default_review_path(args.out)
    if args.append and args.out.exists():
        existing = load_dataset(args.out)
        ledger = load_review_ledger(review_path)
        verified_or_rejected_ids = {
            pid for pid, rec in ledger.items() if rec.status in ("verified", "rejected")
        }
        existing_ids = {qa.id for qa in existing}
        # Never let a freshly drafted candidate collide with a decided pair's
        # id, and don't re-add ids already present.
        candidates = [
            qa
            for qa in candidates
            if qa.id not in verified_or_rejected_ids and qa.id not in existing_ids
        ]
        combined = existing + candidates
    else:
        combined = candidates

    stats = dataset_stats(candidates)
    print("\nnew candidates by category:")
    for cat, count in sorted(stats["by_category"].items()):
        print(f"  {cat:12s} {count}")
    print(f"  unanswerable  {stats['unanswerable']} ({stats['unanswerable']/max(1,stats['total']):.0%})")
    print(f"  answerable    {stats['answerable']}")

    if llm.name in _USD_PER_1K_TOKENS:
        in_rate, out_rate = _USD_PER_1K_TOKENS[llm.name]
        # ~ one prompt+completion per answerable candidate drafted with the LLM
        est_in_tokens = stats["answerable"] * 400
        est_out_tokens = stats["answerable"] * 60
        cost = (est_in_tokens / 1000) * in_rate + (est_out_tokens / 1000) * out_rate
        print(f"\nestimated LLM cost ({llm.name}): ~${cost:.2f}")

    if args.dry_run:
        print(f"\n[dry-run] would write {len(combined)} total pairs to {args.out}; nothing written")
        for qa in candidates[:5]:
            print(f"  [{qa.category}{' UNANSWERABLE' if qa.unanswerable else ''}] {qa.question}")
            print(f"    -> {qa.ground_truth}")
        return 0

    save_dataset(combined, args.out)
    report = validate_dataset(combined)
    print(f"\nwrote {len(combined)} pairs to {args.out}")
    print(f"review ledger: {review_path} (run scripts/review_golden.py next)")
    print("\nvalidation summary:")
    print(report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

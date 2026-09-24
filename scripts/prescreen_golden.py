#!/usr/bin/env python
"""LLM pre-screen for the golden dataset review queue.

Advisory ONLY: writes flags to a sidecar file (`eval/golden_prescreen.jsonl`)
that `scripts/review_golden.py` displays to the human reviewer. This script
never writes to the review ledger and never marks anything "verified" --
`ragpipe.eval.golden.load_verified()` is completely unaffected by running
this. Human approval in the Streamlit review tool remains the only path to
verified.

Resumable: pairs already present in the sidecar are skipped unless
`--force` is passed, so a run interrupted by a 429 or a crash can just be
re-run.

Hard budget guard: refuses to start if the pre-flight estimate exceeds
`--max-usd` (default $0.30), and aborts mid-run (via CostTracker) the moment
real tracked spend passes it.

Run with:
    PYTHONPATH=src RAGPIPE_LLM__PROVIDER=openrouter \\
        RAGPIPE_LLM__MODEL=meta-llama/llama-3.3-70b-instruct \\
        .venv/bin/python scripts/prescreen_golden.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.eval.cost import (  # noqa: E402
    BudgetExceeded,
    CostTracker,
    load_price_table,
    resolve_price,
)
from ragpipe.eval.golden import load_dataset  # noqa: E402
from ragpipe.eval.prescreen import (  # noqa: E402
    build_indices,
    default_prescreen_path,
    load_prescreen,
    prescreen_pair,
    save_prescreen,
)
from ragpipe.ingest.pipeline import read_chunks  # noqa: E402
from ragpipe.providers import get_llm  # noqa: E402
from ragpipe.tokenization import count_tokens  # noqa: E402


def _parse_args() -> argparse.Namespace:
    settings = load_settings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=None, help="default: settings.evaluation.dataset")
    ap.add_argument("--chunks", type=Path, default=None, help="default: <processed_dir>/chunks.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="default: sidecar next to the dataset")
    ap.add_argument("--max-usd", type=float, default=0.30)
    ap.add_argument("--force", action="store_true", help="re-screen pairs already in the sidecar")
    ap.add_argument("--limit", type=int, default=None, help="only screen the first N pending pairs")
    ap.add_argument("--dry-run", action="store_true", help="print the cost estimate, screen nothing")
    args = ap.parse_args()
    args.dataset = args.dataset or settings.evaluation.dataset
    args.chunks = args.chunks or (settings.corpus.processed_path / "chunks.jsonl")
    args.out = args.out or default_prescreen_path(args.dataset)
    return args


def main() -> int:
    args = _parse_args()
    settings = load_settings()

    if not args.dataset.exists():
        print(f"no dataset at {args.dataset}", file=sys.stderr)
        return 1
    if not args.chunks.exists():
        print(f"no chunks at {args.chunks} -- run `ragpipe ingest` first", file=sys.stderr)
        return 1

    pairs = load_dataset(args.dataset)
    chunks = read_chunks(args.chunks)
    chunk_map, chunks_by_doc, duplicate_groups = build_indices(pairs, chunks)

    existing = load_prescreen(args.out)
    pending = [qa for qa in pairs if args.force or qa.id not in existing]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"{len(pairs)} pairs total, {len(existing)} already screened, {len(pending)} pending")
    if not pending:
        print("nothing to do")
        return 0

    llm = get_llm(settings)
    model_name = getattr(llm, "model", settings.llm.model or "")

    # Pre-flight estimate: one LLM call per answerable pair (judge over one
    # chunk) or per in-domain unanswerable pair (judge over ~5 BM25
    # candidates -- larger prompt); off-domain unanswerable pairs cost
    # nothing (no LLM call at all). Token counts are measured from the
    # actual prompts this run will send, not guessed, so this is a real
    # estimate of THIS run, not a generic heuristic.
    price_table = load_price_table()
    try:
        price_in, price_out = resolve_price(model_name, price_table)
    except Exception:
        price_in, price_out = resolve_price(
            settings.llm.model or "meta-llama/llama-3.3-70b-instruct", price_table
        )

    est_in_tokens = 0
    est_out_tokens = 0
    n_llm_calls = 0
    for qa in pending:
        if qa.unanswerable:
            if qa.doc_id is None:
                continue  # off-domain: no LLM call
            doc_chunks = chunks_by_doc.get(qa.doc_id, [])
            text = " ".join(c.text[:1200] for c in doc_chunks[:5])
        else:
            chunk = chunk_map.get(qa.expected_chunk_ids[0]) if qa.expected_chunk_ids else None
            text = chunk.text if chunk else ""
        est_in_tokens += count_tokens(text) + count_tokens(qa.question) + count_tokens(qa.ground_truth) + 150
        est_out_tokens += 120
        n_llm_calls += 1

    est_usd = est_in_tokens * price_in / 1e6 + est_out_tokens * price_out / 1e6
    print(
        f"pre-flight estimate: {n_llm_calls} LLM calls, "
        f"~{est_in_tokens} in / ~{est_out_tokens} out tokens, model={model_name}, "
        f"~${est_usd:.4f} (cap ${args.max_usd:.2f})"
    )
    if est_usd > args.max_usd:
        print(
            f"ABORTING: pre-flight estimate ${est_usd:.4f} exceeds cap ${args.max_usd:.2f}. "
            "Use --limit to screen fewer pairs per run, or raise --max-usd explicitly.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("[dry-run] not screening anything")
        return 0

    tracker = CostTracker(max_usd=args.max_usd)
    screened = 0
    suspect = 0
    for qa in pending:
        try:
            record = prescreen_pair(
                qa,
                llm=llm,
                chunk_map=chunk_map,
                chunks_by_doc=chunks_by_doc,
                duplicate_groups=duplicate_groups,
                model_name=model_name,
            )
        except Exception as exc:  # noqa: BLE001 - keep going, resumable
            print(f"  [{qa.id}] ERROR: {exc} -- skipping, will retry on next run", file=sys.stderr)
            time.sleep(1.0)
            continue

        # Attribute real usage: prescreen_pair does its own LLM call(s)
        # internally via `llm.complete`, so we cannot intercept usage there
        # without threading the tracker through; instead account the LLM's
        # own reported usage from checks when present, else fall back to a
        # rough token estimate for this one pair so the tracker stays honest
        # about which figures are measured vs. estimated.
        usage = record.checks.get("_usage")
        if isinstance(usage, dict) and usage:
            tracker.add(model_name, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        else:
            approx_in = count_tokens(qa.question) + count_tokens(qa.ground_truth) + 800
            try:
                tracker.add(model_name, approx_in, 120, estimated=True)
            except BudgetExceeded:
                raise

        existing[qa.id] = record
        save_prescreen(existing, args.out)
        screened += 1
        if record.verdict == "suspect":
            suspect += 1
            print(f"  [{qa.id}] SUSPECT: {'; '.join(record.reasons)}")
        else:
            print(f"  [{qa.id}] ok")

        if tracker.total_usd > args.max_usd:
            print(
                f"ABORTING mid-run: tracked spend ${tracker.total_usd:.4f} exceeded "
                f"cap ${args.max_usd:.2f} after {screened} pairs. Re-run to resume.",
                file=sys.stderr,
            )
            break

    print(f"\nscreened {screened} pairs this run ({suspect} suspect)")
    print(f"tracked spend this run: ${tracker.total_usd:.4f}")
    print(f"sidecar: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

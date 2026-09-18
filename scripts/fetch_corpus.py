#!/usr/bin/env python3
"""CLI to populate data/raw with a real arXiv corpus for ingestion testing.

Run standalone (no install step), matching the rest of this project's
scripts: it puts `src/` on sys.path before importing ragpipe.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.ingest.arxiv_fetch import ArxivPaper, fetch_corpus  # noqa: E402


def _parse_args() -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=["cs.CL", "cs.LG"])
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--out", type=Path, default=None, help="default: settings.corpus.raw_path")
    parser.add_argument("--manifest", type=Path, default=None, help="default: <processed_dir>/corpus_manifest.json")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between API/download requests")
    parser.add_argument("--dry-run", action="store_true", help="search only, do not download PDFs")
    args = parser.parse_args()

    args.out = args.out or settings.corpus.raw_path
    args.manifest = args.manifest or (settings.corpus.processed_path / "corpus_manifest.json")
    return args


def _progress(event: str, **kw: object) -> None:
    paper: ArxivPaper = kw.get("paper")  # type: ignore[assignment]
    if event == "dry_run":
        print(f"[dry-run]  {paper.arxiv_id}  {paper.title}")
    elif event == "cached":
        print(f"[cached]   {paper.arxiv_id}  {paper.title}")
    elif event == "downloaded":
        size = kw.get("bytes", 0)
        print(f"[fetched]  {paper.arxiv_id}  {paper.title}  ({size:,} bytes)")
    elif event == "skipped":
        print(f"[skipped]  {paper.arxiv_id}  {paper.title}  -- {kw.get('reason')}")
    elif event == "failed":
        print(f"[FAILED]   {paper.arxiv_id}  {paper.title}  -- {kw.get('reason')}")


def main() -> None:
    args = _parse_args()
    print(f"Fetching up to {args.count} papers from {', '.join(args.categories)}")
    print(f"  out={args.out}  manifest={args.manifest}  delay={args.delay}s  dry_run={args.dry_run}")

    started = time.perf_counter()
    manifest = fetch_corpus(
        categories=args.categories,
        target_count=args.count,
        dest_dir=args.out,
        manifest_path=args.manifest,
        delay_s=args.delay,
        dry_run=args.dry_run,
        progress=_progress,
    )
    elapsed = time.perf_counter() - started

    if args.dry_run:
        print(f"\nDry run: {manifest['count']} papers matched, none downloaded ({elapsed:.1f}s).")
        return

    total_bytes = sum(p["bytes"] for p in manifest["papers"])
    print(
        "\nSummary: "
        f"downloaded/cached={len(manifest['papers'])}  "
        f"skipped={len(manifest['skipped'])}  "
        f"failed={len(manifest['failed'])}  "
        f"total={total_bytes:,} bytes  "
        f"elapsed={elapsed:.1f}s"
    )
    if manifest["skipped"]:
        print("Skipped:")
        for s in manifest["skipped"]:
            print(f"  - {s['arxiv_id']}: {s['reason']}")
    if manifest["failed"]:
        print("Failed:")
        for f in manifest["failed"]:
            print(f"  - {f['arxiv_id']}: {f['reason']}")


if __name__ == "__main__":
    main()

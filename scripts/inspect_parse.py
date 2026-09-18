#!/usr/bin/env python3
"""Debugging CLI for the PDF parser: makes parse quality visible.

Parses one PDF and prints its heading tree, a sample of blocks with their
page + section_path, and everything the parser dropped (headers, footers,
page numbers). Run standalone, matching the rest of this project's scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.ingest.base import Block  # noqa: E402
from ragpipe.ingest.pdf import PDFParser  # noqa: E402


def _parse_page_range(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    n = int(spec)
    return n, n


def _print_heading_tree(blocks: list[Block]) -> None:
    print("\n=== Heading tree ===")
    for b in blocks:
        if b.kind != "heading":
            continue
        level = b.heading_level or 1
        indent = "  " * (level - 1)
        print(f"{indent}- (p.{b.page}, L{level}) {b.text}")


def _print_blocks(blocks: list[Block], page_range: tuple[int, int] | None, limit: int = 40) -> None:
    print("\n=== Sample blocks ===")
    shown = 0
    for b in blocks:
        if page_range and (b.page is None or not (page_range[0] <= b.page <= page_range[1])):
            continue
        path = " > ".join(b.section_path) if b.section_path else "(no section)"
        snippet = b.text.replace("\n", " ")
        if len(snippet) > 100:
            snippet = snippet[:100] + "..."
        print(f"[{b.order:04d}] p.{b.page} [{b.kind}] {path}\n        {snippet}")
        shown += 1
        if shown >= limit:
            print(f"        ... ({limit} shown; use --pages to narrow)")
            break


def _print_dropped(dropped: dict[str, list[str]]) -> None:
    print("\n=== Dropped furniture (repetition-based) ===")
    for key in ("headers", "footers", "page_numbers"):
        values = dropped.get(key, [])
        print(f"-- {key} ({len(values)}) --")
        for v in values[:20]:
            print(f"   {v!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True, help="path to a PDF, e.g. data/raw/2609.20584.pdf")
    ap.add_argument("--pages", default=None, help="page range to show blocks for, e.g. 1-3")
    ap.add_argument("--show-dropped", action="store_true")
    ap.add_argument("--show-headings", action="store_true")
    ap.add_argument("--max-blocks", type=int, default=40)
    args = ap.parse_args()

    cfg = load_settings().ingest
    parser = PDFParser(cfg)
    parsed = parser.parse(args.file)
    doc = parsed.document

    print(f"File:        {args.file}")
    print(f"Title:       {doc.title}")
    print(f"doc_id:      {doc.doc_id}")
    print(f"Pages:       {doc.page_count}")
    print(f"Blocks:      {len(parsed.blocks)}")
    print(f"Metadata:    {doc.metadata}")
    if parsed.warnings:
        print(f"Warnings:    {parsed.warnings}")

    if args.show_headings:
        _print_heading_tree(parsed.blocks)

    if args.show_dropped:
        _print_dropped(parsed.dropped)

    page_range = _parse_page_range(args.pages)
    _print_blocks(parsed.blocks, page_range, limit=args.max_blocks)


if __name__ == "__main__":
    main()

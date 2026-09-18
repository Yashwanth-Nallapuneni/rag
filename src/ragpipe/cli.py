"""ragpipe command line.

Subcommands are added phase by phase; `config` and `doctor` exist from the
start so the provider abstraction is inspectable without running a pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:  # allow running from a checkout
    sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.logging_utils import setup_logging  # noqa: E402


def _cmd_config(args: argparse.Namespace) -> int:
    s = load_settings(env=args.env)
    if args.full:
        print(json.dumps(s.model_dump(mode="json"), indent=2, default=str))
    else:
        print(json.dumps(s.describe(), indent=2))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Report which providers are actually usable right now.

    Deliberately never makes a paid API call -- it checks imports, keys and
    reachability so a misconfigured run fails here with a clear message
    instead of deep inside a pipeline.
    """
    from ragpipe.providers import (
        MissingCredentialsError,
        ProviderError,
        get_embedder,
        get_llm,
        get_reranker,
    )

    s = load_settings(env=args.env)
    print(f"config fingerprint: {s.fingerprint()}\n")
    ok = True
    for label, getter in (
        (f"llm ({s.llm.provider})", get_llm),
        (f"embeddings ({s.embeddings.provider})", get_embedder),
        (f"reranker ({s.rerank.provider})", get_reranker),
    ):
        try:
            health = getter(s).health()
            ready = health.get("ready", False)
            ok = ok and bool(ready)
            print(f"  [{'ok' if ready else 'XX'}] {label}: {health}")
        except (MissingCredentialsError, ProviderError) as exc:
            ok = False
            print(f"  [XX] {label}: {exc}")

    corpus = s.corpus.raw_path
    pdfs = len(list(corpus.glob('*.pdf'))) if corpus.exists() else 0
    print(f"\n  corpus: {pdfs} PDFs in {corpus}")
    print(f"\n{'all providers ready' if ok else 'some providers unavailable (see above)'}")
    return 0 if ok else 1


def _cmd_ingest(args: argparse.Namespace) -> int:
    from ragpipe.ingest.pipeline import ingest_corpus

    s = load_settings(env=args.env)
    if args.chunk_size:
        s.chunking.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        s.chunking.chunk_overlap = args.chunk_overlap

    chunks, report = ingest_corpus(s, sources=args.source or None, write=not args.dry_run)
    print(json.dumps(report, indent=2))
    if args.sample and chunks:
        print("\n--- sample chunks ---")
        step = max(1, len(chunks) // args.sample)
        for chunk in chunks[::step][: args.sample]:
            print(f"\n[{chunk.chunk_id}] {chunk.locator()}  ({chunk.token_count} tokens)")
            print(f"  {chunk.text[:300].strip()}...")
    return 0 if report["parsed"] and not report["failed"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragpipe", description="Production-grade RAG pipeline")
    p.add_argument("--env", default=None, help="config/<env>.yaml layer to apply")
    p.add_argument("--log-level", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("config", help="show resolved configuration")
    c.add_argument("--full", action="store_true", help="dump every setting")
    c.set_defaults(func=_cmd_config)

    d = sub.add_parser("doctor", help="check provider availability")
    d.set_defaults(func=_cmd_doctor)

    i = sub.add_parser("ingest", help="parse and chunk the corpus")
    i.add_argument("source", nargs="*", help="files or URLs (default: the whole corpus)")
    i.add_argument("--chunk-size", type=int, default=None)
    i.add_argument("--chunk-overlap", type=int, default=None)
    i.add_argument("--sample", type=int, default=0, help="print N sample chunks")
    i.add_argument("--dry-run", action="store_true", help="do not write chunks.jsonl")
    i.set_defaults(func=_cmd_ingest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    s = load_settings(env=args.env)
    setup_logging(args.log_level or s.logging.level, s.logging.json_output)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

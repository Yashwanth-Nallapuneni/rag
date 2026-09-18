#!/usr/bin/env python
"""Compare retrieval configurations on the known-item diagnostic.

Read the caveats in ragpipe.eval.retrieval_bench before quoting any number
from this: queries are synthesised from the target chunks, which favours
lexical matching. It is a tuning instrument, not an answer-quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.eval.retrieval_bench import (  # noqa: E402
    build_abstract_queries,
    build_queries,
    format_table,
    run_config,
)
from ragpipe.index.builder import get_store  # noqa: E402
from ragpipe.ingest.pipeline import read_chunks  # noqa: E402
from ragpipe.logging_utils import setup_logging  # noqa: E402


def _settings(**over):
    retrieval = {
        "mode": over.pop("mode", "hybrid"),
        "fusion": over.pop("fusion", "rrf"),
        "dense_weight": over.pop("dense_weight", 0.5),
        "sparse_weight": over.pop("sparse_weight", 0.5),
        "candidate_k": over.pop("candidate_k", 30),
        "top_k": 10,
    }
    rerank = {"enabled": over.pop("rerank", False)}
    return load_settings(overrides={"retrieval": retrieval, "rerank": rerank, **over})


def _retriever(settings, store, shared):
    """Reuse the BM25 index and rerank model across configurations so the
    comparison measures retrieval, not repeated setup cost."""
    from ragpipe.retrieval.dense import DenseRetriever
    from ragpipe.retrieval.hybrid import HybridRetriever

    if settings.retrieval.mode == "dense" and not settings.rerank.enabled:
        return DenseRetriever(settings, store)

    rerank_stage = None
    if settings.rerank.enabled:
        if "rerank" not in shared:
            from ragpipe.retrieval.rerank import RerankStage

            shared["rerank"] = RerankStage(settings)
        rerank_stage = shared["rerank"]

    sparse = None
    if settings.retrieval.mode in ("sparse", "hybrid"):
        if "bm25" not in shared:
            from ragpipe.retrieval.bm25 import BM25Retriever

            shared["bm25"] = BM25Retriever.build_or_load(settings)
        sparse = shared["bm25"]

    dense = DenseRetriever(settings, store) if settings.retrieval.mode != "sparse" else None
    return HybridRetriever(
        settings, store, dense=dense, sparse=sparse, reranker=rerank_stage
    )


CONFIGS: list[tuple[str, dict]] = [
    ("dense only", {"mode": "dense"}),
    ("sparse only (BM25)", {"mode": "sparse"}),
    ("hybrid RRF", {"mode": "hybrid", "fusion": "rrf"}),
    ("hybrid weighted 50/50", {"mode": "hybrid", "fusion": "weighted_sum"}),
    ("dense + rerank", {"mode": "dense", "rerank": True}),
    ("hybrid RRF + rerank", {"mode": "hybrid", "fusion": "rrf", "rerank": True}),
]

SWEEP = [
    ("weighted 0.9d/0.1s", {"fusion": "weighted_sum", "dense_weight": 0.9, "sparse_weight": 0.1}),
    ("weighted 0.7d/0.3s", {"fusion": "weighted_sum", "dense_weight": 0.7, "sparse_weight": 0.3}),
    ("weighted 0.5d/0.5s", {"fusion": "weighted_sum", "dense_weight": 0.5, "sparse_weight": 0.5}),
    ("weighted 0.3d/0.7s", {"fusion": "weighted_sum", "dense_weight": 0.3, "sparse_weight": 0.7}),
    ("weighted 0.1d/0.9s", {"fusion": "weighted_sum", "dense_weight": 0.1, "sparse_weight": 0.9}),
    ("rrf weights 0.5/0.5", {"fusion": "rrf"}),
    ("rrf weights 0.7d/0.3s", {"fusion": "rrf", "dense_weight": 0.7, "sparse_weight": 0.3}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=50)
    ap.add_argument("--sweep", action="store_true", help="tune fusion weights instead")
    ap.add_argument("--out", default="eval_results/retrieval_bench.json")
    args = ap.parse_args()

    setup_logging("WARNING")
    base = load_settings()
    chunks = read_chunks(base.corpus.processed_path / "chunks.jsonl")
    queries = build_queries(chunks, per_family=args.per_family)
    queries += build_abstract_queries(chunks, limit=args.per_family)
    print(f"{len(queries)} queries over {len(chunks)} chunks from "
          f"{len({c.doc_id for c in chunks})} documents\n")

    store = get_store(base)
    shared: dict = {}
    todo = SWEEP if args.sweep else CONFIGS
    results = []
    for label, over in todo:
        if args.sweep:
            over = {"mode": "hybrid", **over}
        settings = _settings(**dict(over))
        started = time.perf_counter()
        results.append(run_config(label, settings, _retriever(settings, store, shared), queries))
        print(f"  ran {label:<26} in {time.perf_counter() - started:5.1f}s")

    print("\n" + format_table(results))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "queries": len(queries),
                "per_family": args.per_family,
                "chunks": len(chunks),
                "config_fingerprint": base.fingerprint(),
                "caveat": (
                    "Known-item diagnostic. Queries are synthesised from their "
                    "target chunks, which favours lexical matching. Not an "
                    "answer-quality or faithfulness measurement."
                ),
                "results": [
                    {
                        "label": r.label,
                        "config": r.config,
                        "families": [f.as_row() for f in r.families],
                        "overall": r.overall.as_row() if r.overall else None,
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

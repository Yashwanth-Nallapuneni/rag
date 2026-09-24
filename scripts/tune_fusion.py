#!/usr/bin/env python
"""Tune hybrid fusion weights against the golden set. Retrieval-only, no LLM
calls, $0 to run.

Splits the golden set into a TUNE set (pick the best config) and a held-out
TEST set (report that config's score honestly, next to the 0.5/0.5 RRF
baseline, with a paired bootstrap 95% CI). Recommends a change only if the
CI excludes 0 -- see docs/STATE.md §6 and src/ragpipe/eval/fusion_tuning.py.

Reads human-verified pairs by default (`ragpipe.eval.golden.load_verified`).
`--allow-unverified` falls back to the raw drafted dataset; every report
then says pairs_human_verified: false; drafted questions were written FROM
their target chunk, so they are lexically biased toward BM25 -- see
docs/retrieval_findings.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.eval.fusion_tuning import (  # noqa: E402
    BASELINE,
    RerankCache,
    bootstrap_ci,
    build_grid,
    collect_candidates,
    decide,
    per_category,
    pick_best,
    run_config,
    stratified_split,
    sweep,
)
from ragpipe.eval.golden import load_dataset, load_verified  # noqa: E402
from ragpipe.index.builder import get_store  # noqa: E402

CAVEAT = (
    "CAVEAT: pairs are NOT human-verified. Drafted questions were written "
    "FROM their target chunk (lexical overlap bias favoring BM25) -- see "
    "docs/retrieval_findings.md. Treat this run as a dry run, not evidence."
)


def _build_dense_sparse(settings, store):
    from ragpipe.retrieval.bm25 import BM25Retriever
    from ragpipe.retrieval.dense import DenseRetriever

    dense = DenseRetriever(settings, store) if settings.retrieval.mode in ("dense", "hybrid") else None
    sparse = None
    indexed = store.iter_chunks()
    if indexed and settings.retrieval.mode in ("sparse", "hybrid"):
        sparse = BM25Retriever.build_or_load(settings, indexed)
    return dense, sparse


def _build_reranker(settings, enabled: bool):
    if not enabled:
        return None
    from ragpipe.retrieval.rerank import RerankStage

    return RerankStage(settings)


def _metric_attr(name: str) -> str:
    return {"recall@1": "recall_1", "recall@5": "recall_5", "mrr@10": "mrr_10", "doc_hit@5": "doc_hit_5"}[name]


def _paired_diff_list(chosen_metrics, baseline_metrics, metric_name: str) -> list[float]:
    attr = _metric_attr(metric_name)
    baseline_by_id = {m.qa_id: m for m in baseline_metrics}
    diffs = []
    for m in chosen_metrics:
        b = baseline_by_id.get(m.qa_id)
        if b is None:
            continue
        diffs.append(getattr(m, attr) - getattr(b, attr))
    return diffs


def _view(pairs, settings, store, *, rerank_enabled: bool, tune_ratio: float, primary_metric: str):
    dense, sparse = _build_dense_sparse(settings, store)
    reranker = _build_reranker(settings, rerank_enabled)
    candidate_k = settings.retrieval.candidate_k
    top_k = settings.retrieval.top_k

    answerable = [qa for qa in pairs if not qa.unanswerable and qa.expected_chunk_ids]
    tune_pairs, test_pairs = stratified_split(answerable, tune_ratio=tune_ratio)

    t0 = time.perf_counter()
    tune_candidates = collect_candidates(tune_pairs, dense, sparse, candidate_k)
    test_candidates = collect_candidates(test_pairs, dense, sparse, candidate_k)
    first_pass_s = time.perf_counter() - t0

    grid = build_grid()
    cache = RerankCache(reranker)

    t0 = time.perf_counter()
    tune_rows = sweep(
        tune_candidates, grid, candidate_k=candidate_k, top_k=top_k,
        rerank_cache=cache, primary_metric=primary_metric,
    )
    sweep_s = time.perf_counter() - t0
    best_row = pick_best(tune_rows, primary_metric=primary_metric)

    from ragpipe.eval.fusion_tuning import FusionConfig

    best_cfg = FusionConfig(**best_row["config"])

    t0 = time.perf_counter()
    baseline_test_metrics = run_config(
        test_candidates, BASELINE, candidate_k=candidate_k, top_k=top_k, rerank_cache=cache
    )
    chosen_test_metrics = run_config(
        test_candidates, best_cfg, candidate_k=candidate_k, top_k=top_k, rerank_cache=cache
    )
    test_eval_s = time.perf_counter() - t0

    from ragpipe.eval.fusion_tuning import aggregate

    baseline_test_agg = aggregate(baseline_test_metrics)
    chosen_test_agg = aggregate(chosen_test_metrics)

    ci_by_metric = {}
    for m in ("recall@1", "recall@5", "mrr@10", "doc_hit@5"):
        diffs = _paired_diff_list(chosen_test_metrics, baseline_test_metrics, m)
        ci_by_metric[m] = bootstrap_ci(diffs)

    primary_ci = ci_by_metric[primary_metric]
    recommend_change = decide(primary_ci)

    return {
        "rerank_enabled": rerank_enabled,
        "tune_n": len(tune_pairs),
        "test_n": len(test_pairs),
        "tune_top10": tune_rows[:10],
        "best_config_on_tune": best_row,
        "test_baseline": {"config": BASELINE.as_dict(), **baseline_test_agg},
        "test_chosen": {"config": best_cfg.as_dict(), **chosen_test_agg},
        "test_ci_by_metric": ci_by_metric,
        "primary_metric": primary_metric,
        "decision": {
            "recommend_change": recommend_change,
            "reason": (
                f"test-set {primary_metric} CI excludes 0 "
                f"(mean={primary_ci['mean']:+.4f}, 95% CI=[{primary_ci['lo']:+.4f}, {primary_ci['hi']:+.4f}])"
                if recommend_change else
                f"test-set {primary_metric} CI includes 0 "
                f"(mean={primary_ci['mean']:+.4f}, 95% CI=[{primary_ci['lo']:+.4f}, {primary_ci['hi']:+.4f}]); "
                "data does not support changing the config -- keep 0.5/0.5"
            ),
        },
        "per_category_test": {
            "baseline": per_category(baseline_test_metrics),
            "chosen": per_category(chosen_test_metrics),
        },
        "timing_s": {
            "first_pass_retrieval": round(first_pass_s, 2),
            "tune_grid_sweep": round(sweep_s, 2),
            "test_eval": round(test_eval_s, 2),
            "rerank_cache_calls": cache.calls,
        },
    }


def _print_view(name: str, view: dict) -> None:
    print(f"\n{'=' * 70}\nVIEW: {name} (rerank {'ON' if view['rerank_enabled'] else 'OFF'})\n{'=' * 70}")
    print(f"tune_n={view['tune_n']}  test_n={view['test_n']}  primary_metric={view['primary_metric']}")
    print("\nTop 10 configs on TUNE:")
    print(f"{'config':<28} {'n':>4} {'recall@1':>9} {'recall@5':>9} {'mrr@10':>8} {'doc_hit@5':>10}")
    for r in view["tune_top10"]:
        c = r["config"]["label"]
        print(f"{c:<28} {r['n']:>4} {r['recall@1']:>9.4f} {r['recall@5']:>9.4f} {r['mrr@10']:>8.4f} {r['doc_hit@5']:>10.4f}")

    print(f"\nBest config on TUNE: {view['best_config_on_tune']['config']['label']}")
    print("\nTEST-set comparison (baseline 0.5/0.5 RRF vs chosen config):")
    b, c = view["test_baseline"], view["test_chosen"]
    print(f"{'metric':<10} {'baseline':>10} {'chosen':>10} {'diff mean':>10} {'95% CI':>22}")
    for m in ("recall@1", "recall@5", "mrr@10", "doc_hit@5"):
        ci = view["test_ci_by_metric"][m]
        print(f"{m:<10} {b[m]:>10.4f} {c[m]:>10.4f} {ci['mean']:>+10.4f} [{ci['lo']:>+.4f}, {ci['hi']:>+.4f}]")

    print(f"\nDECISION: {'CHANGE' if view['decision']['recommend_change'] else 'KEEP 0.5/0.5'}")
    print(f"  {view['decision']['reason']}")

    print("\nPer-category TEST recall@5 (baseline vs chosen):")
    cats = sorted(set(view["per_category_test"]["baseline"]) | set(view["per_category_test"]["chosen"]))
    for cat in cats:
        bc = view["per_category_test"]["baseline"].get(cat, {})
        cc = view["per_category_test"]["chosen"].get(cat, {})
        print(f"  {cat:<12} n={bc.get('n', 0):>3}  baseline recall@5={bc.get('recall@5', 0):.4f}  chosen recall@5={cc.get('recall@5', 0):.4f}")

    t = view["timing_s"]
    print(f"\ntiming: first_pass={t['first_pass_retrieval']}s  tune_sweep={t['tune_grid_sweep']}s  "
          f"test_eval={t['test_eval']}s  rerank_cache_calls={t['rerank_cache_calls']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=None)
    ap.add_argument("--allow-unverified", action="store_true",
                     help="use the raw drafted dataset when nothing is verified yet")
    ap.add_argument("--tune-ratio", type=float, default=0.6)
    ap.add_argument("--primary-metric", default="recall@5",
                     choices=["recall@1", "recall@5", "mrr@10", "doc_hit@5"])
    ap.add_argument("--out", type=Path, default=None)
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

    store = get_store(settings)

    print(f"fusion tuning -- {len(pairs)} pairs from {dataset} "
          f"({'HUMAN-VERIFIED' if verified else 'UNVERIFIED DRAFTS'})")
    if not verified:
        print(CAVEAT)

    views = {}
    for rerank_enabled, name in ((True, "rerank_on_PRIMARY"), (False, "rerank_off_secondary")):
        view = _view(
            pairs, settings, store,
            rerank_enabled=rerank_enabled,
            tune_ratio=args.tune_ratio,
            primary_metric=args.primary_metric,
        )
        view["pairs_human_verified"] = verified
        if not verified:
            view["caveat"] = CAVEAT
        views[name] = view
        _print_view(name, view)

    result = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "pairs_human_verified": verified,
        "n_pairs": len(pairs),
        "views": views,
    }
    if not verified:
        result["caveat"] = CAVEAT

    out_dir = ROOT / "eval_results"
    out_dir.mkdir(exist_ok=True)
    out_path = args.out or (out_dir / f"fusion_tuning_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

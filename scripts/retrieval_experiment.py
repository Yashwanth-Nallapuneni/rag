#!/usr/bin/env python
"""Retrieval-only reranker / fusion / k experiment on the human-verified golden
set. $0, no LLM calls. Reuses the tune/test split, metrics and bootstrap
decision rule from ragpipe.eval.fusion_tuning (same seed, same stratification).

Winner is chosen on TUNE only (recall@5, ties by mrr@10); its TEST result is
reported against the production baseline (RRF 0.5/0.5 + MiniLM-L-6, ck30, k5).
Also measures the relevance-gate score separation (answerable vs unanswerable)
for each reranker, since the gate is a raw logit tied to the reranker model.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.eval.fusion_tuning import (  # noqa: E402
    BASELINE,
    FusionConfig,
    bootstrap_ci,
    build_grid,
    decide,
    stratified_split,
)
from ragpipe.eval.golden import load_verified  # noqa: E402
from ragpipe.index.builder import get_store  # noqa: E402
from ragpipe.retrieval.fusion import fuse  # noqa: E402

MODELS = {
    "minilm-l6": ("cross-encoder/ms-marco-MiniLM-L-6-v2", "Apache-2.0"),
    "minilm-l12": ("cross-encoder/ms-marco-MiniLM-L-12-v2", "Apache-2.0"),
    "bge-base": ("BAAI/bge-reranker-base", "MIT"),
    "bge-v2-m3": ("BAAI/bge-reranker-v2-m3", "Apache-2.0"),
}
METRICS = ("recall@1", "recall@5", "recall@8", "mrr@10")


class Scorer:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.scores: dict = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
        self.models: dict = {}
        self.calls = 0

    def model(self, key):
        if key not in self.models:
            import torch
            from sentence_transformers import CrossEncoder
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self.models[key] = CrossEncoder(MODELS[key][0], device=dev)
        return self.models[key]

    def score(self, key, qa, chunks):
        need = [rc for rc in chunks if (key, qa.id, rc.chunk_id) not in self.scores]
        if need:
            self.calls += 1
            ce = self.model(key)
            out = ce.predict([(qa.question, rc.chunk.text) for rc in need], batch_size=16)
            for rc, s in zip(need, out, strict=True):
                self.scores[(key, qa.id, rc.chunk_id)] = float(s)
        vals = [self.scores[(key, qa.id, rc.chunk_id)] for rc in chunks]
        if key.startswith("bge"):
            # sentence-transformers applies a sigmoid to these models; the cache holds
            # probabilities. Convert to logits (monotone, so ordering is unchanged) so the
            # gate table is on a comparable raw-logit scale.
            import math
            vals = [math.log(min(max(v, 1e-7), 1 - 1e-7) / (1 - min(max(v, 1e-7), 1 - 1e-7))) for v in vals]
        return vals

    def save(self):
        self.cache_path.write_bytes(pickle.dumps(self.scores))


def fused_shortlist(cands, fc: FusionConfig, ck):
    d, s = cands[ck]
    return fuse(d, s, method=fc.method, rrf_k=fc.rrf_k, dense_weight=fc.dense_weight,
                sparse_weight=fc.sparse_weight)[:ck]


def order(scorer, qa, cands, fc, ck, model, blend_k):
    fused = fused_shortlist(cands, fc, ck)
    if model is None:
        return fused, None
    sc = scorer.score(model, qa, fused)
    idx = sorted(range(len(fused)), key=lambda i: -sc[i])  # stable: ties keep fusion order
    if blend_k:
        rr = {i: r for r, i in enumerate(idx, 1)}
        idx = sorted(range(len(fused)), key=lambda i: -(1 / (blend_k + i + 1) + 1 / (blend_k + rr[i])))
    return [fused[i] for i in idx], sc


def per_query(ranked, qa):
    ids = [rc.chunk_id for rc in ranked]
    exp = set(qa.expected_chunk_ids)
    def rec(k): return 1.0 if set(ids[:k]) & exp else 0.0
    mrr = next((1 / i for i, c in enumerate(ids[:10], 1) if c in exp), 0.0)
    return {"recall@1": rec(1), "recall@5": rec(5), "recall@8": rec(8), "mrr@10": mrr,
            "tok@5": sum(rc.chunk.token_count for rc in ranked[:5]),
            "tok@8": sum(rc.chunk.token_count for rc in ranked[:8])}


def mean(rows, key): return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def auroc(pos, neg):
    if not pos or not neg:
        return None
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--cache", type=Path, default=Path("/tmp/none"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    settings = load_settings()
    store = get_store(settings)
    from ragpipe.retrieval.bm25 import BM25Retriever
    from ragpipe.retrieval.dense import DenseRetriever
    dense = DenseRetriever(settings, store)
    sparse = BM25Retriever.build_or_load(settings, store.iter_chunks())

    pairs = load_verified(settings.evaluation.dataset_path)
    ans = [q for q in pairs if not q.unanswerable and q.expected_chunk_ids]
    una = [q for q in pairs if q.unanswerable]
    tune, test = stratified_split(ans, tune_ratio=0.6)
    print(f"answerable={len(ans)} tune={len(tune)} test={len(test)} unanswerable={len(una)}", flush=True)

    t0 = time.perf_counter()
    C = {}  # qa.id -> {ck: (dense, sparse)}
    for qa in ans + una:
        C[qa.id] = {ck: (dense.retrieve(qa.question, k=ck), sparse.retrieve(qa.question, k=ck)) for ck in (30, 50)}
    print(f"first pass {time.perf_counter()-t0:.1f}s", flush=True)

    scorer = Scorer(args.cache)
    result = {"timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
              "n_answerable": len(ans), "n_unanswerable": len(una), "tune_n": len(tune), "test_n": len(test),
              "models": {k: {"id": v[0], "licence": v[1]} for k, v in MODELS.items()}}

    # ---- latency (warm, 8 queries, mps) ----
    lat = {}
    for key in args.models:
        sample = tune[:8]
        scorer.score(key, sample[0], fused_shortlist(C[sample[0].id], BASELINE, 30))  # warmup (cached afterwards; use fresh ids below)
        row = {}
        for ck in (30, 50):
            ts = []
            ce = scorer.model(key)
            for qa in sample[1:]:
                fz = fused_shortlist(C[qa.id], BASELINE, ck)
                t = time.perf_counter()
                ce.predict([(qa.question, rc.chunk.text) for rc in fz], batch_size=16)
                ts.append(time.perf_counter() - t)
            row[f"s_per_query_ck{ck}"] = round(sum(ts) / len(ts), 3)
        lat[key] = row
        print("latency", key, row, flush=True)
    result["latency_mps"] = lat

    # ---- configs ----
    configs = []  # (name, fusion cfg, ck, model, blend_k)
    for fc in build_grid():
        configs.append((f"off|{fc.label}|ck30", fc, 30, None, 0))

    def evalset(cfg, qs):
        _, fc, ck, model, bk = cfg
        return {qa.id: per_query(order(scorer, qa, C[qa.id], fc, ck, model, bk)[0], qa) for qa in qs}

    tune_res = {}
    for cfg in configs:
        tune_res[cfg[0]] = evalset(cfg, tune)

    def agg(res):
        rows = list(res.values())
        return {m: mean(rows, m) for m in (*METRICS, "tok@5", "tok@8")}

    off_best = max([c for c in configs if c[3] is None], key=lambda c: (agg(tune_res[c[0]])["recall@5"], agg(tune_res[c[0]])["mrr@10"]))
    fstar = off_best[1]
    print("OFF winner on tune:", off_best[0], flush=True)
    result["off_winner_fusion"] = off_best[0]

    for key in args.models:
        for fc in (BASELINE, fstar) if fstar.label != BASELINE.label else (BASELINE,):
            for ck in (30, 50):
                for bk in (0, 60, 10):
                    nm = f"on:{key}|{fc.label}|ck{ck}" + (f"|blend{bk}" if bk else "")
                    configs.append((nm, fc, ck, key, bk))
                print("scoring", key, fc.label, ck, flush=True)
                for qa in tune + test:
                    scorer.score(key, qa, fused_shortlist(C[qa.id], fc, ck))
                scorer.save()
        print("done model", key, time.perf_counter() - t0, flush=True)

    prod = next(c for c in configs if c[0] == f"on:minilm-l6|{BASELINE.label}|ck30")
    full = {}
    for cfg in configs:
        tr = tune_res.get(cfg[0]) or evalset(cfg, tune)
        te = evalset(cfg, test)
        full[cfg[0]] = (tr, te)
    base_te = full[prod[0]][1]
    rows = []
    for cfg in configs:
        tr, te = full[cfg[0]]
        row = {"name": cfg[0], "model": cfg[3], "fusion": cfg[1].as_dict(), "candidate_k": cfg[2], "blend_k": cfg[4],
               "tune": agg(tr), "test": agg(te), "test_diff": {}}
        for m in (*METRICS, "tok@5", "tok@8"):
            d = [te[q][m] - base_te[q][m] for q in te]
            row["test_diff"][m] = bootstrap_ci(d)
        row["test_diff_recall5_ci_excludes_0_above"] = decide(row["test_diff"]["recall@5"])
        rows.append(row)
    rows.sort(key=lambda r: (-r["tune"]["recall@5"], -r["tune"]["mrr@10"]))
    result["configs"] = rows
    result["baseline"] = prod[0]
    result["winner_on_tune"] = rows[0]["name"]
    result["winner_on_tune_top5"] = [r["name"] for r in rows[:5]]
    # best ON per model (tune) for convenience
    result["best_per_model_on_tune"] = {}
    for key in [None, *args.models]:
        rs = [r for r in rows if r["model"] == key]
        if rs:
            result["best_per_model_on_tune"][str(key)] = rs[0]["name"]

    # ---- gate separation ----
    gates = {}
    def gate_for(model, fc, ck, shown_k):
        a, u = [], []
        for qs, out in ((ans, a), (una, u)):
            for qa in qs:
                fused = fused_shortlist(C[qa.id], fc, ck)
                sc = scorer.score(model, qa, fused)
                if shown_k:  # gate over only the chunks fusion shows the LLM
                    sc = sc[:shown_k]
                out.append(max(sc))
        lo, hi = min(a + u), max(a + u)
        step = max(0.5, round((hi - lo) / 14 * 2) / 2)
        ths, t = [], lo
        while t <= hi + 1e-9:
            ths.append(round(t, 1))
            t += step
        table = [{"threshold": t, "false_refusals": sum(s < t for s in a), "unanswerable_passed": sum(s >= t for s in u)} for t in ths]
        best = min(table, key=lambda r: (r["false_refusals"] + r["unanswerable_passed"], r["threshold"]))
        def qs_(xs):
            return {q: sorted(xs)[min(len(xs) - 1, int(p * (len(xs) - 1) + .5))] for q, p in (("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1))}
        return {"answerable": qs_(a), "unanswerable": qs_(u), "auroc_ans_vs_unans": auroc(a, u), "n_ans": len(a), "n_unans": len(u),
                "table": table, "best_threshold_min_errors": best}
    for key in args.models:
        gates[f"{key}|{BASELINE.label}|ck30|shortlist"] = gate_for(key, BASELINE, 30, None)
    # no-rerank-ordering alternative: MiniLM-L6 gate over only the top-5 / top-8 shown by fusion
    for fc in {BASELINE.label: BASELINE, fstar.label: fstar}.values():
        for sk in (5, 8):
            gates[f"minilm-l6|{fc.label}|ck30|fusion_top{sk}_only"] = gate_for("minilm-l6", fc, 30, sk)
        gates[f"minilm-l6|{fc.label}|ck30|shortlist"] = gate_for("minilm-l6", fc, 30, None)
    result["gate"] = gates
    scorer.save()

    out = args.out or ROOT / "eval_results" / f"retrieval_experiment_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)

    print(f"\n{'config':<62} tR@5 tR@8 tMRR | R@1  R@5  R@8  MRR  dR@5 [CI] tok5 tok8")
    for r in rows[:40]:
        t, e, d = r["tune"], r["test"], r["test_diff"]["recall@5"]
        print(f"{r['name']:<62} {t['recall@5']:.3f} {t['recall@8']:.3f} {t['mrr@10']:.3f} | {e['recall@1']:.3f} {e['recall@5']:.3f} {e['recall@8']:.3f} {e['mrr@10']:.3f} {d['mean']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] {e['tok@5']:.0f} {e['tok@8']:.0f}")


if __name__ == "__main__":
    main()

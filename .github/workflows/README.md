# CI workflows

Two separate workflows, two separate badges, two separate meanings:

| Workflow | Badge | Answers |
|---|---|---|
| [`tests.yml`](./tests.yml) | tests | "Does the code work?" (lint + 233 offline unit/integration tests) |
| [`eval.yml`](./eval.yml) | answer quality | "Are the answers still good?" (RAGAS eval vs. the golden dataset) |

They're split on purpose. `tests.yml` is free, fast, and needs no secret —
it runs the deterministic `mock` LLM/embedder path, so it's the right gate
for every commit. `eval.yml` needs a real judge model, a real corpus, and a
real vector index, so it's slower and costs money per run — but it's the
one that actually answers the spec's question: *did this PR make the
answers worse?*

## `tests.yml`

- Triggers: every PR, every push to `main`.
- No secrets required.
- Steps: `ruff check`, then `pytest -q -m "not slow"` (the 233-test offline
  suite — no network, no model downloads, no API key, because everything
  runs through ragpipe's `mock` provider).
- Typical runtime: well under a minute of actual test time, plus dependency
  install.

## `eval.yml` — the answer-quality gate

- Triggers: every PR (per spec), every push to `main`, and
  `workflow_dispatch` (manual re-run with custom `sample_size`,
  `judge_model`, `max_usd` inputs — see "Re-running by hand" below).
- **Requires the `OPENROUTER_API_KEY` repository secret.** This is the judge
  model RAGAS uses to score faithfulness / relevancy / precision / recall.
  Add it under **Settings → Secrets and variables → Actions →
  `OPENROUTER_API_KEY`**.
- **If the secret is missing, the job fails immediately** with a clear
  `::error::` message, before any other step runs. This is deliberate: the
  alternative — quietly falling back to the mock judge, or skipping the eval
  step — would let a PR merge with a green "answer quality" badge despite no
  real evaluation ever having happened. A missing secret is a repo
  misconfiguration and must look like one, not like success.
- Spending is capped by `--max-usd` (default `1.00`, overridable via the
  `workflow_dispatch` input or the `MAX_USD` env in the workflow), so a CI
  misconfiguration can't run up an open-ended bill.
- `scripts/run_eval.py` chooses its own output filename internally
  (`eval_results/eval_<timestamp>_<config-fingerprint>.json`) and prints it
  as `Result written to: <path>`; the workflow greps that line out of the
  step's log rather than assuming a fixed filename. If `run_eval.py` exits
  without ever printing that line (crashed, or refused to run because the
  cost estimate exceeded `--max-usd`), the workflow points `ci_gate.py` at
  a path that deliberately does not exist, so the gate fails loudly (exit
  3) instead of silently passing on no data.

### What gets cached, and why

The corpus and the vector index are **not** committed to git (only
`data/processed/corpus_manifest.json` is — see the comment in `.gitignore`).
Rebuilding them from scratch on every PR would make the gate both slow and
pointlessly repetitive, so the workflow caches aggressively and only rebuilds
on an actual change:

| Cache | Path(s) | Key | Rebuilt with |
|---|---|---|---|
| HF model weights | `~/.cache/huggingface`, `~/.cache/torch/sentence_transformers` | hash of the embedder + reranker model names (`config/default.yaml`: `embeddings.model`, `rerank.model`) | downloaded automatically the first time they're used |
| Corpus | `data/raw` | hash of `data/processed/corpus_manifest.json` | `python scripts/fetch_corpus.py --count 40` — respects arXiv's ~1 req/3s rate limit, ~4 minutes, deliberately not parallelised |
| Chunks + vector index | `data/chroma`, `data/processed/chunks.jsonl` | manifest hash **+** hash of the `chunking:`/`ingest:` sections of `config/default.yaml` | `make ingest PY=python && make index PY=python` |

The index cache key includes the chunking config on top of the manifest
hash so that a `chunk_size`/`chunk_overlap`/etc. change invalidates the
index even when the underlying papers haven't changed — otherwise you'd
silently evaluate against a stale index built with the old chunking.

### Baseline for regression detection

To make "this PR regressed faithfulness from 0.91 to 0.84" possible, the
workflow needs last-known-good numbers to compare against. Rather than
committing eval results to the repo, it relies on a documented property of
GitHub Actions cache: **caches saved on the default branch (`main`) are
visible to PR branches via `restore-keys` prefix matching.** So:

- On every push to `main`, if the gate passed, the run's result JSON is
  saved to the cache under `eval-baseline-<sha>`.
- On every run (PR or push), the workflow tries to restore a cache matching
  the `eval-baseline-` prefix, which resolves to the most recently saved
  one — i.e., main's last good run.
- `scripts/ci_gate.py` then compares the current run against that baseline
  and fails (exit code 2) if any metric dropped by more than
  `--max-regression` (default `0.02`), even if the metric is still above
  its absolute threshold.

Caveat: GitHub evicts caches after ~7 days of no access, or once the repo's
overall cache quota is exceeded. If no baseline is found (first run ever,
or an evicted cache), the regression check is skipped with a note in the
log and job summary — it does not fail the build. Absolute thresholds are
still enforced regardless.

### Interpreting a blocked PR

A red `eval` check on a PR means one of:

- **Exit 1 — below threshold.** Some metric (e.g. `faithfulness`) is below
  the value configured in `EvalThresholds` (`src/ragpipe/config.py` /
  `config/default.yaml`, currently faithfulness 0.80, answer_relevancy
  0.70, context_precision 0.65, context_recall 0.65, refusal_accuracy
  0.80).
- **Exit 2 — regression.** Every metric is still above its absolute
  threshold, but one dropped by more than `--max-regression` (0.02) versus
  main's last good run — e.g. faithfulness 0.91 → 0.84.
- **Exit 3 — no usable result.** The eval script didn't produce a readable
  result file (crashed, timed out, wrote malformed JSON). Treated as a
  hard failure, never as a pass — a missing result is not evidence of
  quality, so it must never look green.

Every run uploads build artifacts (`eval-results-<run id>`, 90-day
retention) as evidence for whichever of the above happened: the raw result
JSON that `scripts/run_eval.py` wrote (`eval_results/eval_<timestamp>_
<config-fingerprint>.json` — the workflow discovers this exact path from
the script's own "Result written to: ..." log line, since it isn't a fixed
filename), the human-readable pass/fail table `ci_gate.py` produced
(`eval_results/ci_gate_summary.txt`), and the full `run_eval.py` console log.
The same pass/fail table is written to the run's Job Summary page, and — on
same-repo PRs — posted/updated as a PR comment (fork PRs get a read-only
token and can't be commented on; the workflow logs a warning and still
enforces pass/fail correctly via the required check, it just can't leave
the comment).

### Re-running by hand

Use **Actions → eval → Run workflow** to trigger `workflow_dispatch`. You
can override:

- `sample_size` — how many golden-dataset questions to evaluate (default 20)
- `judge_model` — which model RAGAS uses as judge (default `meta-llama/llama-3.3-70b-instruct` on OpenRouter; generation is `openai/gpt-oss-120b`)
- `max_usd` — spending cap for the run (default `1.00`)

This is useful for a full-dataset run, or to re-check quality after a
prompt/config change without opening a throwaway PR.

## `scripts/ci_gate.py`

The threshold/regression comparator both workflows (well, just `eval.yml`)
rely on, and the piece meant to be run locally too:

```bash
python scripts/ci_gate.py eval_results/eval_20260918T000000Z_abc123def456.json
python scripts/ci_gate.py eval_results/eval_20260918T000000Z_abc123def456.json \
  --baseline eval_results/baseline.json --max-regression 0.02
```

See the module docstring in `scripts/ci_gate.py` for the exact exit-code
contract (0 pass / 1 threshold fail / 2 regression fail / 3 bad input) and
the accepted result-JSON shapes.

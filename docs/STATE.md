# Project state and handoff

Last updated: 2026-10-17. **Project complete.** 363 passed offline (`pytest -m "not slow"`, 4 slow tests deselected). OpenRouter spend: $3.80 of the $5 cap (real `/key` delta).

This file exists so a fresh session can resume without re-deriving anything or
undoing a decision that was made for a measured reason. Read it before changing
retrieval, enforcement thresholds or the evaluation harness.

---

## 1. Standing instructions from the project owner

These override convenience. They were stated explicitly and repeatedly.

- **The spec (`production_rag_project_spec.md`) is the golden standard.** Follow
  it precisely. No shortcuts, no scope reduction.
- **No false claims.** Any number that has not been measured stays absent or
  marked *pending measurement*. An estimate presented as a measurement is the
  single worst failure mode here, because the project's entire thesis is that
  its quality claims are checkable.
- **Budget is hard: under $5 total.** Pre-flight cost estimates must lean high,
  never low.
- **Commits are authored by the owner** (`Yashwanth Nallapuneni
  <yashwanthatwork@gmail.com>`), with no AI co-author trailer. History was
  rewritten once to remove that; do not reintroduce it.
- Build-out work is delegated to Sonnet subagents, but **their claims are
  verified independently**. Several reported things that were not true (see §8).

---

## 2. What is built and working

| Layer | State |
|---|---|
| Corpus | 40 arXiv PDFs, manifest versioned in git, PDFs are not |
| Ingestion | PDF / Markdown / HTML+URL → Block IR (page + heading stack) |
| Chunking | 1054 chunks, 650 tokens / 100 overlap, sentence-aligned |
| Index | ChromaDB, BGE-small-en-v1.5 on MPS, idempotent rebuild |
| Retrieval | Hybrid dense + BM25, RRF fusion, cross-encoder rerank (30 → 5) |
| Orchestration | LangGraph StateGraph, 8 nodes, 4 conditional refuse edges |
| Enforcement | Per-claim coverage + numeric + negation, LLM judge on the ambiguous middle, plus a separate relevance gate |
| API / UI | FastAPI (`/query`, `/chunk/{id}`, `/health`, `/stats`) + Streamlit demo |
| Eval | Golden-set tooling, RAGAS harness, cost guard, CI gate, all proven end to end against a real judge |
| CI | `tests.yml` (free, secretless) + `eval.yml` (quality gate, every PR) |

Resolved config (do not change without re-measuring):

```
chunking        650 / 100
retrieval       hybrid, rrf, weights 0.5/0.5, top_k 5, candidate_k 30
rerank          on, BAAI/bge-reranker-base (raw logits)
citation        enforce on, min_supported_ratio 0.80, lexical_threshold 0.45,
                min_relevance_score -2.5
prompts         answer/v3
thresholds      faithfulness 0.80, answer_relevancy 0.70,
                context_precision 0.65, context_recall 0.65,
                refusal_accuracy 0.80
```

---

## 3. Spec compliance

**Complete:** 1.1 ingestion (PDF/MD/web), 1.2 chunking (500-800 tok, ~100
overlap, full metadata), 1.3 ChromaDB, 1.4 top-k retrieval with citations and
click-through, 2.1 hybrid BM25+vector with both RRF and weighted fusion, 2.2
cross-encoder reranking over a 20-30 candidate pool, 2.3 citation enforcement
with refusal, 2.4 versioned prompts in `prompts/`. Tech stack: LangGraph,
ChromaDB, sentence-transformers, RAGAS, FastAPI, GitHub Actions.

**One item is partial:** fusion weight tuning (spec 2.1). Both mechanisms
exist and a full sweep was run, but the weights were deliberately left at
0.5/0.5 (see §6). The golden dataset, faithfulness figures and before/after
reranking comparison are now done (§4). The repo is public.

---

## 4. Final measurement (done)

The golden set was reviewed (171 human-verified pairs: 145 answerable, 26
unanswerable) and the final runs were made on it. Same generator
(`openai/gpt-oss-120b`), judge (`meta-llama/llama-3.3-70b-instruct`), prompt
v3, hybrid RRF, `candidate_k` 30, `top_k` 5.

| Metric | Rerank on | Rerank off |
|---|---|---|
| Faithfulness | 0.933 | 0.931 |
| Answer relevancy | 0.820 | 0.813 |
| Context precision | 0.910 | 0.871 |
| Context recall | 0.964 | 0.985 |
| Refusal accuracy | 0.953 | 0.930 |
| False refusals / false answers | 8 / 0 | 12 / 0 |

- Rerank on: `docs/results/eval_20261017T011212Z_5f0a41881655.json` (commit d276390).
- Rerank off: `docs/results/eval_20261017T033055Z_4d290a449c2f.json` (commit 50fc0ca, same code as d276390 plus a decision-log commit).
- Reading: reranking does not improve faithfulness. It improves context
  precision, refusal accuracy and false refusals (12 to 8). Context recall is
  slightly higher with rerank off. Generation is nondeterministic, so a few
  pairs of difference are within noise. Full table and subset numbers in the
  README; reasoning in `docs/decisions.md` (D18, D19).
- Held-out retrieval: bge-reranker-base vs MiniLM reranker Recall@5 0.879 vs
  0.707 (test n=58, 95% bootstrap CI on the difference [+0.05, +0.29]).
- Spend: the OFF run cost $0.422 real against $0.329 reported by the harness;
  the gap is verifier judge calls the harness does not count. Total project
  spend is $3.80 of $5.

---

## 5. Remaining (all optional)

- Merge Dependabot PR #2 and confirm the CI checks pass (being done by the
  orchestrator after this commit): status pending.
- Fusion weight tuning (spec 2.1) stays at 0.5/0.5 RRF by measurement (see §6).
- Any further false-refusal work is inside run-to-run noise (decisions D19).

Run config for every paid run (OpenRouter):
`RAGPIPE_LLM__PROVIDER=openrouter RAGPIPE_LLM__MODEL=openai/gpt-oss-120b
RAGPIPE_LLM__REASONING_EFFORT=low RAGPIPE_LLM__MAX_TOKENS=1400`, then
`--judge-provider openrouter --judge-model meta-llama/llama-3.3-70b-instruct
--ragas-workers 2`. A full 171-pair run costs about $0.33 as reported and
about $0.42 real.

---

## 6. Calibration traps: read before touching a threshold

**The relevance gate (`-7.0`, the MiniLM-era value; now `-2.5` for bge-reranker-base, see docs/decisions.md D9/D10) is NOT what refused the answerable question.**
An earlier version of this file said it was. That was wrong: the refuse node
dropped token usage, so a refusal *after* generation (by citation enforcement)
showed zero tokens and looked like one *before* it. Fixed, with a regression
test. `scripts/calibrate_gate.py` (retrieval + rerank only, no LLM, no cost)
then showed every answerable smoke question scoring >= 0.61, far above -7.0.
So the live false refusal came from **citation enforcement on a paraphrasing
model**. See the enforcement paragraph below.

On all 180 UNVERIFIED drafts: 0/153 answerable below -7.0 (lowest -4.39); all
13 off-domain probes at -10 to -11, caught; the 14 in-domain "uncovered
detail" probes pass the gate, as they should: they name a real paper, so
the passages ARE on topic; refusing those is enforcement's job. Loosening to
-5 changes nothing; tightening to 0 refuses 13 answerable. Gate unchanged.

Still re-run `calibrate_gate.py` on the verified golden set: drafted
questions share vocabulary with their chunk, so their scores are optimistic.
Do not simply loosen the gate either way: it is what stops a faithfully-quoted
but irrelevant passage from becoming a confident answer (asked "what is the
capital of Mongolia?", an early build retrieved a Toronto weather-station
passage, quoted it accurately and scored 100% supported).

**Do NOT tune fusion weights on the known-item benchmark.** `make bench`
reports BM25 crushing dense retrieval and improves monotonically toward
`sparse_weight=0.9`. That is an artifact: its queries are synthesised from
their target chunks, so they share vocabulary by construction. A fourth family
built specifically to escape the bias (abstract → body) did not escape it
either. BM25 scored a perfect 1.000, because accepting any of a paper's 10-25
body chunks makes the task document identification, which each paper's
distinctive jargon settles trivially. Full reasoning in
`docs/retrieval_findings.md`. Tune against the golden set instead.

**Pilot false refusals, 40 unverified pairs: 13 -> 10 -> 7**, with zero
unanswerable questions wrongly answered throughout. Every fix was in parsing,
none in thresholds: `【S1†L1-L4】` markers, "U.S." as a sentence end,
zero-width and narrow spaces inside markers, curly quotes, numbers glued to
identifiers and a Unicode minus in the passages. Of the remaining 7: 3 are
the model itself declining (INSUFFICIENT_CONTEXT), and 4 contain a genuinely
uncited sentence, which the traceability rule is right to reject. Both are
policy questions to settle on VERIFIED data, not by loosening on drafts.

**Fusion weights: data says keep 0.5/0.5 RRF.** `scripts/tune_fusion.py`
(retrieval-only, $0) tunes on a 60% split and changes the config only if the
held-out gain's 95% bootstrap CI is entirely above 0. With rerank on
(production), the best tune-split config lost 0.033 Recall@5 on test (CI
[-0.10, +0.03]). With rerank off, RRF 0.1/0.9 toward BM25 won clearly, which
is the lexical bias from §6 again: the drafts were written from their chunks.
Re-run on verified pairs before calling deviation 1 closed.

**Enforcement thresholds were also implicitly tuned to an extractive mock**,
which copies sentences verbatim and scores ~1.0 coverage. Real models
paraphrase. If false refusals appear, prefer fixing escalation to the LLM judge
over lowering `min_supported_ratio`.

---

## 7. Provider facts, measured on a live key

- **OpenRouter is the paid path.** Generation `openai/gpt-oss-120b`
  ($0.15/$0.60 per M), judge `meta-llama/llama-3.3-70b-instruct`
  ($0.10/$0.32). Different families, so the judge never grades its own
  family. Prices in `cost.py` were verified against the live
  `/api/v1/models` on 2026-09-23; the earlier Qwen "estimate" had been 2-5x
  too low. Re-verify before large runs.
- **A published price is not evidence a model exists for your key.**
  `llama-3.3-70b-versatile` is in every Groq price list and 404s on this
  account.
- **gpt-oss-120b is a reasoning model.** At `max_tokens=40` the whole budget
  went to reasoning and the content came back empty. Keep 1400 and
  `REASONING_EFFORT=low`.
- **It writes typographic Unicode everywhere**: `【S1†L1-L4】`, `[\u200bS2]`,
  `[\u202fS1\u202f]`, U+2011 hyphens, curly quotes. All handled by
  `normalize_typography` + marker normalisation. A new marker shape shows
  up as a 0% support refusal with top rerank scores > 4; check raw output first.
- **Groq free tier (200K tokens/day) cannot run the spec's evals**: 150
  pairs would need ~10.7 days of quota. Useful only for free drafting.

---

## 8. Bugs already fixed: do not reintroduce

Each has a regression test. Several were only findable with a real model.

| Bug | Consequence if reintroduced |
|---|---|
| Citation markers as `[n]` | A paper's own "[4, 15, 20]" references parse as citations to passages that may not exist |
| ASCII-only marker regex | Groq cites as 【S1】 (CJK brackets) → 0% support → **100% refusal rate**, looking like a retrieval fault |
| Claim splitter breaking on any period | Email dots shatter one answer into ten bogus uncited claims |
| Context header regex requiring balanced parens | A heading containing "(RE3 & RE5)" merges passages and mis-attributes every sentence in one |
| Trailing marker attributed to the next sentence | Every claim verified against its neighbour's passage |
| BM25 built from `chunks.jsonl` instead of the store | Indexes drift; a hit the store lacks gives a citation that 404s |
| `.env` not loaded for provider keys | Keys present but invisible; every script fails |
| RAGAS default concurrency + 180s timeout | All metrics NaN behind a rate-limited judge that is working fine |
| Cost estimate assuming 1200 generation tokens | Under-estimates a hard budget; measured is 2840 |
| Hardcoded provider lists in CLIs | Silently stale the moment a provider is added |
| Refuse node dropping token usage | Post-generation refusals look pre-generation; cost under-reported (misdiagnosed the relevance gate once) |
| Unanswerable probes drawn with replacement | 27 probes = 11 unique questions; refusal accuracy measured on repeats |
| `【S1†L1-L4】`, `[ S1 ]`, zero-width chars in markers | Citations invisible -> 0% support -> refusal despite perfect retrieval |
| "U.S." read as a sentence end | Content half loses its markers; one grounded claim scored 50% |
| Strict number extraction on the evidence side / Unicode minus | "Qwen 3.6-35B" or "T^-1/2" fails as a figure "absent from the passage", which is a hard fail |
| CI fetching the corpus by arXiv search | A cache miss builds this week's papers; the golden set is asked of the wrong corpus. Use `--from-manifest` |
| Fusion `decide()` accepting a CI entirely below 0 | Recommends a significantly WORSE config |
| `ruff` missing from requirements.txt | Lint step exits 127 ("command not found"); the test suite never runs in CI |
| No LICENSE file behind the README's "MIT" | No legal effect; GitHub detects no license |

---

## 9. Commands

```bash
make doctor          # provider readiness, no paid calls
make ingest          # parse + chunk  -> 1054 chunks
make index           # embed + store  -> ChromaDB (~35s, idempotent)
make ask Q="..."     # one question, with citations
make serve           # FastAPI on :8000
make ui              # Streamlit demo on :8501
make test            # offline suite, no key needed
make bench           # retrieval diagnostic (read §6 first)

# Live eval (Groq). --dry-run prints the cost estimate and calls nothing.
export RAGPIPE_LLM__PROVIDER=groq RAGPIPE_LLM__REASONING_EFFORT=low RAGPIPE_LLM__MAX_TOKENS=1400
python scripts/run_eval.py --dataset eval/golden_dataset.jsonl \
  --judge-provider groq --judge-model qwen/qwen3.8-27b \
  --ragas-workers 1 --ragas-timeout 900 --max-usd 1.00 --dry-run

python scripts/ci_gate.py <results.json> --baseline <previous.json>
# exit 0 pass | 1 below threshold | 2 regression | 3 missing/malformed
```

---

## 10. Numbers: what may and may not be claimed

**Measured, quotable:**
- 40 documents, 1054 chunks; 100% carry a page number, 98.8% a section
- 97.9% of adjacent chunk pairs share overlap text (mean 85 words)
- Known-item Recall@1: BM25 0.300 vs dense 0.060 on exact identifiers
- Known-item Recall@1: reranking lifts dense 0.256 to 0.483 (earlier MiniLM
  experiment, superseded setup)
- Held-out Recall@5: bge-reranker-base 0.879 vs MiniLM 0.707 (CI [+0.05, +0.29])
- Final runs on 171 human-verified pairs: see §4
- False refusals 32 -> 23 -> 9 -> 8 with thresholds never loosened
- 363 passed offline

Call the known-item figures **"known-item Recall@1"**, never "precision". The
diagnostic is lexically biased and the honest framing is a strength: it shows
the limits of one's own benchmark.

Do not claim reranking improves faithfulness: the OFF run shows 0.933 vs
0.931. Every result file records `pairs_human_verified` plus a caveat string.

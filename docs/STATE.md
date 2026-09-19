# Project state and handoff

Last updated: 2026-09-19. **274 passed** (offline, no API key).

This file exists so a fresh session can resume without re-deriving anything or
undoing a decision that was made for a measured reason. Read it before changing
retrieval, enforcement thresholds, or the evaluation harness.

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
  verified independently** — several reported things that were not true (see §8).

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
| Eval | Golden-set tooling, RAGAS harness, cost guard, CI gate — all proven end to end against a real judge |
| CI | `tests.yml` (free, secretless) + `eval.yml` (quality gate, every PR) |

Resolved config (do not change without re-measuring):

```
chunking        650 / 100
retrieval       hybrid, rrf, weights 0.5/0.5, top_k 5, candidate_k 30
rerank          on, cross-encoder/ms-marco-MiniLM-L-6-v2
citation        enforce on, min_supported_ratio 0.80, lexical_threshold 0.45,
                min_relevance_score -7.0
prompts         answer/v2
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

**Open — two items:**

1. **Fusion weight tuning (spec 2.1) is PARTIAL.** Both mechanisms exist and a
   full sweep was run, but the weights were deliberately left at 0.5/0.5. See
   §6 — this is blocked on the golden dataset, not forgotten.
2. **Spec Phase 3 deliverables need real numbers:** the golden dataset itself,
   the faithfulness figures, before/after reranking faithfulness, the CI
   screenshot, and the resume bullet's X.

**Also required by the spec and not yet done: the repo must be PUBLIC.** It is
currently private (unauthenticated API returns 404). The CI badges and the
Mermaid architecture diagrams cannot be seen or verified until it is public.

---

## 4. Blocked on the project owner

1. **OpenRouter key** in `.env` as `OPENROUTER_API_KEY`. Groq's free tier
   cannot produce the spec's numbers (§7). Projected total spend ~$2.
2. **Make the GitHub repo public.**
3. **1-2 hours reviewing the golden set** -- DRAFTED, waiting on review:
   `eval/golden_dataset.jsonl`, 180 pairs (153 answerable, 27 unanswerable),
   drafted by `qwen/qwen3.8-27b` on Groq. Run
   `PYTHONPATH=src .venv/bin/streamlit run scripts/review_golden.py`.
   Review targets: one duplicate pair ("What is On-Demand Attention?", reject
   one); vague questions that name no paper (see the lowest scores in
   `eval_results/gate_calibration_*.json`); and each in-domain unanswerable
   probe must be checked as truly absent from its paper. `load_verified()`
   returns only human-approved pairs and a fresh file yields zero — so this
   cannot be faked, and must not be worked around.

---

## 5. Next actions, in dependency order

```
1. Owner verifies the 180 drafts (need >= 150 approved)         scripts/review_golden.py
2. Re-run the gate calibration on VERIFIED pairs                 scripts/calibrate_gate.py
3. Diagnose enforcement false refusals on real generations      (§6)
4. First real eval run                                           scripts/run_eval.py
5. Tune fusion weights against the golden set                    (closes deviation 1)
6. Before/after reranking faithfulness  (--rerank / --no-rerank, same dataset+judge)
7. Push, open a PR, capture the CI screenshot
8. Fill the resume bullet's measured X
```

---

## 6. Calibration traps — read before touching a threshold

**The relevance gate (`-7.0`) is NOT what refused the answerable question.**
An earlier version of this file said it was. That was wrong: the refuse node
dropped token usage, so a refusal *after* generation (by citation enforcement)
showed zero tokens and looked like one *before* it. Fixed, with a regression
test. `scripts/calibrate_gate.py` (retrieval + rerank only, no LLM, no cost)
then showed every answerable smoke question scoring >= 0.61, far above -7.0.
So the live false refusal came from **citation enforcement on a paraphrasing
model** -- see the enforcement paragraph below.

On all 180 UNVERIFIED drafts: 0/153 answerable below -7.0 (lowest -4.39); all
13 off-domain probes at -10 to -11, caught; the 14 in-domain "uncovered
detail" probes pass the gate, as they should -- they name a real paper, so
the passages ARE on topic; refusing those is enforcement's job. Loosening to
-5 changes nothing; tightening to 0 refuses 13 answerable. Gate unchanged.

Still re-run `calibrate_gate.py` on the verified golden set: drafted
questions share vocabulary with their chunk, so their scores are optimistic.
Do not simply loosen the gate either way: it is what stops a faithfully-quoted
but irrelevant passage from becoming a confident answer (asked "what is the
capital of Mongolia?", an early build retrieved a Toronto weather-station
passage, quoted it accurately, and scored 100% supported).

**Do NOT tune fusion weights on the known-item benchmark.** `make bench`
reports BM25 crushing dense retrieval and improves monotonically toward
`sparse_weight=0.9`. That is an artifact: its queries are synthesised from
their target chunks, so they share vocabulary by construction. A fourth family
built specifically to escape the bias (abstract → body) did not escape it
either — BM25 scored a perfect 1.000, because accepting any of a paper's 10-25
body chunks makes the task document identification, which each paper's
distinctive jargon settles trivially. Full reasoning in
`docs/retrieval_findings.md`. Tune against the golden set instead.

**Enforcement thresholds were also implicitly tuned to an extractive mock**,
which copies sentences verbatim and scores ~1.0 coverage. Real models
paraphrase. If false refusals appear, prefer fixing escalation to the LLM judge
over lowering `min_supported_ratio`.

---

## 7. Provider facts, measured on a live key

- **A published price is not evidence a model exists for your key.**
  `llama-3.3-70b-versatile` appears in every Groq pricing write-up and returns
  404 on this account. Always check `client.models.list()`.
- **13 models available:** `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
  `openai/gpt-oss-safeguard-20b`, `qwen/qwen3.8-27b`, `groq/compound`,
  `groq/compound-mini`, `allam-2-7b`, plus whisper/prompt-guard models.
- **Generation:** `openai/gpt-oss-120b`. It is a *reasoning* model — a
  three-word answer cost 148 reasoning + 15 content tokens, and at
  `max_tokens=40` the whole budget went to reasoning, returning empty content.
  Keep `max_tokens` generous (1400 used) and set
  `RAGPIPE_LLM__REASONING_EFFORT=low` (~40% fewer reasoning tokens, identical
  verdicts).
- **Judge:** `qwen/qwen3.8-27b`. Different family from the generator (so the
  judge stays independent), no reasoning overhead (86 output tokens vs 279),
  and a far higher free-tier token-per-minute ceiling.
- **Groq free tier is too small for the spec.** `gpt-oss-120b` allows 30 RPM /
  1K RPD / 8K TPM / **200K TPD**, and one eval sample is ~14,251 tokens:
  14 pairs ≈ 25 min of calling but a full day's quota; 50 pairs ≈ 3.6 days;
  150 pairs ≈ 10.7 days. A 40-pair CI gate is 2.9× over the daily cap, so
  spec 3.3 ("every pull request") is impossible on free.
- **Prices for `qwen/qwen3.8-27b` and `openai/gpt-oss-20b` in
  `DEFAULT_PRICES` are UNVERIFIED estimates.** Confirm against live pricing
  before any paid run, or override via `RAGPIPE_EVAL_PRICES`.

---

## 8. Bugs already fixed — do not reintroduce

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
- Known-item Recall@1 — BM25 0.300 vs dense 0.060 on exact identifiers
- Known-item Recall@1 — reranking lifts dense 0.256 → 0.483 (+89%)
- Section-label noise 0.0% (was 3.6%)
- 274 passed offline

Call the retrieval figures **"known-item Recall@1"**, never "precision". The
diagnostic is lexically biased and the honest framing is a strength: it shows
the limits of one's own benchmark.

**NOT measured — must stay blank:**
- Faithfulness, answer relevance, context precision/recall on the golden set
- Before/after reranking faithfulness
- Refusal accuracy (the 0.25 → 0.75 → 0.67 figures are 2-4 sample diagnostics)
- The resume bullet's X

The only real-judge RAGAS run so far graded **one** sample (faithfulness 1.000,
answer_relevancy 0.916) on **unverified** pairs. It proves the harness runs; it
measures nothing. Every result file records `pairs_human_verified` plus a
caveat string for exactly this reason.

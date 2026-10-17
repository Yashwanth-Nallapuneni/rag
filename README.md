# ragpipe: a production-grade RAG system

[![tests](https://github.com/Yashwanth-Nallapuneni/rag/actions/workflows/tests.yml/badge.svg)](https://github.com/Yashwanth-Nallapuneni/rag/actions/workflows/tests.yml)
[![answer quality](https://github.com/Yashwanth-Nallapuneni/rag/actions/workflows/eval.yml/badge.svg)](https://github.com/Yashwanth-Nallapuneni/rag/actions/workflows/eval.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Ask questions over an arXiv ML corpus and get answers that **cite the exact
passage they came from**, or an explicit refusal when the retrieved passages
don't support an answer.

The point of this project is not the demo. It is the discipline around it:
hybrid retrieval, cross-encoder reranking, enforced citations and an
evaluation harness that **gates CI on answer faithfulness**.

> **Status: complete.** Ingestion, hybrid retrieval, reranking, enforced
> citations, the FastAPI service, the Streamlit demo, the RAGAS harness and
> the CI quality gate all work, with 363 offline tests. The final numbers
> below come from 171 human-verified golden QA pairs; the run files are named
> in the Results section.

## Why the provider abstraction

Every backend (LLM, embeddings, reranker) sits behind a protocol in
`src/ragpipe/providers/`. A deterministic offline `mock` LLM and `mock`
embedder satisfy the same contracts as Anthropic/OpenAI/local models, which
means the entire pipeline, including citation enforcement and the eval
harness, runs **with no API key, at no cost, with no run-to-run variance**,
and therefore runs in CI. Switching to a hosted model is one config line.

## Quickstart

```bash
make install
make doctor      # which providers are usable right now
make corpus      # download the arXiv corpus (rate-limited, ~4 min)
make ingest      # parse + chunk  -> 1054 chunks from 40 papers
make index       # embed + store  -> ChromaDB, ~35s
make ask Q="what problem does self-attention solve?"
make serve       # API on :8000, interactive docs at /docs
make ui          # Streamlit demo on :8501
make test        # 363 offline tests
```

> **Resuming work on this?** [`docs/STATE.md`](docs/STATE.md) is the current
> handoff: what is built, what is blocked and the calibration traps that will
> damage retrieval quality if a threshold is changed without re-measuring.

## The problem this solves

Ask a question of a document collection and a typical RAG system gives you a
fluent paragraph with no way to tell whether it is true. It will answer
confidently when the documents say nothing on the subject, because nothing in
the pipeline is checking.

This system is built so that every answer is either **traceable to an exact
passage** or **explicitly refused**:

- Each sentence carries a citation marker resolving to one indexed chunk, and
  `GET /chunk/{id}` returns that chunk's full text, so a reader can land on
  the paragraph a claim came from.
- Claims are verified against the passage they cite *after* generation. An
  answer whose claims are not supported is refused rather than shipped.
- Relevance is gated separately from grounding, because a faithful quotation of
  an irrelevant passage passes every citation check and still fails the user.
- The evaluation harness measures this, and **CI fails the build when answer
  faithfulness regresses**, so the property is enforced continuously, not
  asserted once in a README.

## Architecture

Three diagrams (query path, ingestion path, evaluation path) in
[`docs/architecture.md`](docs/architecture.md).

Orchestration is a LangGraph state machine: `retrieve → relevance_gate →
build_context → generate → parse_citations → check_attribution → verify → finalize`, with
conditional edges routing to a terminal `refuse` node at five distinct points.

## Results

### Corpus and retrieval diagnostics

| | |
|---|---|
| Documents / chunks | 40 arXiv papers, 1054 chunks |
| Chunks carrying a page number | 100% |
| Chunks carrying a section | 98.8% |
| Adjacent chunks sharing overlap text | 97.9% (mean 85 words) |
| BM25 vs dense on exact identifiers (known-item R@1) | **0.300 vs 0.060** |
| Reranking lift on a weak first pass (known-item R@1, earlier MiniLM experiment) | 0.256 → 0.483 |
| Held-out Recall@5, bge-reranker-base vs MiniLM reranker (test n=58) | **0.879 vs 0.707**, bootstrap 95% CI on the difference [+0.05, +0.29] |

The known-item figures come from a synthetic benchmark that is lexically
biased and from the superseded MiniLM setup. The held-out Recall@5 row is the
one that decided the reranker choice
(`eval_results/retrieval_experiment_20261016T200538Z*`).

### Answer quality: reranking on vs off

Evaluated on 171 human-verified golden pairs (145 answerable, 26
unanswerable). Generator `openai/gpt-oss-120b`, judge
`meta-llama/llama-3.3-70b-instruct`, answer prompt v3, hybrid BM25 + dense
with RRF fusion, `candidate_k` 30, `top_k` 5. The two runs differ only in the
reranker.

| Metric | Rerank on | Rerank off |
|---|---|---|
| Faithfulness | 0.933 | 0.931 |
| Answer relevancy | 0.820 | 0.813 |
| Context precision | 0.910 | 0.871 |
| Context recall | 0.964 | 0.985 |
| Refusal accuracy | 0.953 | 0.930 |
| Refusal precision | 0.765 | 0.684 |
| Refusal recall | 1.000 | 1.000 |
| Refusal F1 | 0.867 | 0.813 |
| Answered | 137 | 133 |
| False refusals (answerable, refused) | 8 | 12 |
| False answers (unanswerable, answered) | 0 | 0 |

On the 128 pairs answered in both runs: faithfulness 0.929 vs 0.929, context
precision 0.905 vs 0.875, context recall 0.961 vs 0.984 (on vs off).

Run files: rerank on `eval_results/eval_20261017T011212Z_5f0a41881655.json`
(commit d276390, `BAAI/bge-reranker-base`); rerank off
`eval_results/eval_20261017T033055Z_4d290a449c2f.json` (commit 50fc0ca, the
same code plus a decision-log commit, run with `--no-rerank`). Result files
are git-ignored, so they exist on the machine that ran them rather than in a
clone.

**What the comparison shows.** Reranking does not improve faithfulness: 0.933
vs 0.931, and identical on the pairs both runs answered. What it improves is
context precision (0.910 vs 0.871) and refusal behaviour (accuracy 0.953 vs
0.930, false refusals 12 down to 8). Context recall is slightly higher with
reranking off (0.985 vs 0.964).

**Nondeterminism.** The generator is not deterministic. Two earlier,
near-identical runs differed on 13 pairs, so differences of a few pairs
between single runs (including the 4 fewer false refusals here) are within
noise. These are single-run numbers.

False refusals across the iteration, with thresholds never loosened:
32 → 23 → 9 → 8 (see `docs/decisions.md`, D19).

Retrieval numbers and the reason the most eye-catching one in them is a trap:
[`docs/retrieval_findings.md`](docs/retrieval_findings.md).

## Retrieval

Dense (BGE embeddings over ChromaDB) and sparse (BM25) each propose 30
candidates; reciprocal rank fusion merges them; a cross-encoder reranks that
shortlist down to the 5 passages the model sees. Wide-then-narrow matters: a
reranker can only reorder what it is given, so a passage dense retrieval put
12th can never reach first place if the shortlist was only 5 long.

Both indexes are built from the **same** source (the sparse index reads the
vector store's contents, not the chunk file) because two independently built
indexes drift, and a BM25 hit for a chunk the store lacks yields a citation
whose click-through 404s.

Measured retrieval numbers, and why the most eye-catching number in them is
a trap worth understanding, are in
[`docs/retrieval_findings.md`](docs/retrieval_findings.md).

## Citations

Every answer sentence carries a `[S1]`-style marker that resolves to an exact
passage, and `GET /chunk/{chunk_id}` returns that passage in full so a reader
can land on the paragraph the claim came from.

The `S` prefix is load-bearing. Academic prose is dense with its own reference
markers, such as *"generative retrieval in industrial search [4, 15, 20, 26]"*, so
a plain `[n]` scheme cannot distinguish a citation to passage 4 from the source
paper's own bibliography entry 4. Quoting a passage verbatim then yields an
answer that appears to cite passages which may not exist. `[S4]` cannot
collide.

## Refusing to answer

The system declines rather than guessing, and it checks two *different*
properties to decide.

**Grounding**: is each sentence actually supported by the passage it cites?
Every claim is verified independently, cheapest check first:

1. *Content-word coverage* against the cited passage. Fast, but a weak proxy
   for entailment on its own: "X improves Y" and "X does not improve Y"
   overlap almost completely.
2. *Numeric agreement*: every figure in a claim must appear in the cited
   passage. The highest-precision cheap check available on an academic corpus:
   a claim of "2.5 BLEU" against a passage saying "3.1 BLEU" has flawless word
   overlap and is simply false.
3. *Negation parity*: a claim that negates where its passage does not is
   unsupported however well the words match. This is the case coverage cannot
   see.

An LLM judge settles only the ambiguous middle, so its cost tracks ambiguity
rather than answer length. Hard numeric and negation failures are never
escalated: a model must not be able to talk itself into accepting 2.5 where
the passage says 3.1. If fewer than `min_supported_ratio` (0.8) of claims
survive, the answer is refused and the failing verdicts are shown.

**Relevance**: do the passages address the question at all? This is a
separate property, and conflating the two is a real trap: asked *"what is the
capital of Mongolia?"*, an early build retrieved a passage about Toronto
weather stations, quoted it faithfully and scored **100% supported**. It was
perfectly grounded and completely useless. The cross-encoder settles this,
since query-passage relevance is exactly what it is trained for. Measured on
the 171 human-verified golden pairs with `BAAI/bge-reranker-base`, answerable
questions score −1.3 to +10.3 for their best passage and unanswerable ones
−9.9 to +7.5. The two overlap, so the gate is not a clean separator: it sits at
−2.5 (1.2 logits below the lowest answerable score, 0 of 145 false refusals,
9 of 26 unanswerable stopped) and citation enforcement handles the rest. (The
earlier MiniLM reranker gated at −7.0.) That number is a raw logit for one
specific reranker model, so changing the model means it must be re-measured.

On a 14-question probe (8 answerable, 6 not), run with the earlier MiniLM reranker and not repeated since, the system answered 8/8
and refuses 6/6. That is a smoke test, not an evaluation; the labelled version
arrives with the golden dataset in Phase 6.

Refusals are detected from a sentinel the prompt mandates rather than by
pattern-matching apologetic prose, which varies by model and is unreliable to
parse. Over the API a refusal is a **200 with `status: refused_*`**. It is a
product behaviour the client renders, not an HTTP failure.

## Configuration

All settings live in [`config/default.yaml`](config/default.yaml), typed in
`src/ragpipe/config.py`. Override any of them with env vars:

```bash
RAGPIPE_RETRIEVAL__TOP_K=8 RAGPIPE_LLM__PROVIDER=anthropic make doctor
```

`Settings.fingerprint()` hashes the whole resolved config and is recorded with
every evaluation run, so a metric change can always be traced to a config
change.

## Layout

```
config/          versioned configuration
prompts/         versioned prompt templates (Phase 5)
src/ragpipe/
  config.py      typed settings, YAML + env layering
  schemas.py     Chunk / RetrievedChunk / Citation / Answer contracts
  providers/     LLM, embedding and reranker backends behind protocols
  ingest/        PDF / Markdown / HTML parsing (Phase 2)
  chunking/      token-window chunking with overlap (Phase 2)
  index/         embedding + vector store (Phase 3)
  retrieval/     hybrid BM25 + dense, reranking (Phase 4)
  generation/    answer synthesis, citation enforcement (Phase 5)
  eval/          RAGAS harness and CI gate (Phase 6)
scripts/         corpus fetch and pipeline entry points
tests/
```

## License

[MIT](LICENSE) © 2026 Yashwanth Sai Nallapuneni

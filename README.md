# ragpipe — a production-grade RAG system

[![tests](https://github.com/Yashwanth-Nallapuneni/rag/actions/workflows/tests.yml/badge.svg)](https://github.com/Yashwanth-Nallapuneni/rag/actions/workflows/tests.yml)
[![answer quality](https://github.com/Yashwanth-Nallapuneni/rag/actions/workflows/eval.yml/badge.svg)](https://github.com/Yashwanth-Nallapuneni/rag/actions/workflows/eval.yml)

Ask questions over an arXiv ML corpus and get answers that **cite the exact
passage they came from** — or an explicit refusal when the retrieved passages
don't support an answer.

The point of this project is not the demo. It is the discipline around it:
hybrid retrieval, cross-encoder reranking, enforced citations, and an
evaluation harness that **gates CI on answer faithfulness**.

> **Status: Phase 6 of 6 in progress.** Ingestion, hybrid retrieval,
> reranking, enforced citations, a FastAPI service and a Streamlit demo all
> work end to end (233 tests). The golden dataset, RAGAS harness and CI
> quality gate are being built now.
>
> Numbers marked **pending measurement** below are not yet measured. They stay
> blank until a real evaluation run produces them — an estimated number in
> this README would make every other number in it worthless.

## Why the provider abstraction

Every backend — LLM, embeddings, reranker — sits behind a protocol in
`src/ragpipe/providers/`. A deterministic offline `mock` LLM and `mock`
embedder satisfy the same contracts as Anthropic/OpenAI/local models, which
means the entire pipeline, including citation enforcement and the eval
harness, runs **with no API key, at no cost, with no run-to-run variance** —
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
make test        # 156 offline tests
```

> **Resuming work on this?** [`docs/STATE.md`](docs/STATE.md) is the current
> handoff: what is built, what is blocked, and the calibration traps that will
> damage retrieval quality if a threshold is changed without re-measuring.

## The problem this solves

Ask a question of a document collection and a typical RAG system gives you a
fluent paragraph with no way to tell whether it is true. It will answer
confidently when the documents say nothing on the subject, because nothing in
the pipeline is checking.

This system is built so that every answer is either **traceable to an exact
passage** or **explicitly refused**:

- Each sentence carries a citation marker resolving to one indexed chunk, and
  `GET /chunk/{id}` returns that chunk's full text — so a reader can land on
  the paragraph a claim came from.
- Claims are verified against the passage they cite *after* generation. An
  answer whose claims are not supported is refused rather than shipped.
- Relevance is gated separately from grounding, because a faithful quotation of
  an irrelevant passage passes every citation check and still fails the user.
- The evaluation harness measures this, and **CI fails the build when answer
  faithfulness regresses** — so the property is enforced continuously, not
  asserted once in a README.

## Architecture

Three diagrams — query path, ingestion path, evaluation path — in
[`docs/architecture.md`](docs/architecture.md).

Orchestration is a LangGraph state machine: `retrieve → relevance_gate →
build_context → generate → parse_citations → verify → finalize`, with
conditional edges routing to a terminal `refuse` node at four distinct points.

## Results

| | |
|---|---|
| Documents / chunks | 40 arXiv papers, 1054 chunks |
| Chunks carrying a page number | 100% |
| Chunks carrying a section | 98.8% |
| Adjacent chunks sharing overlap text | 97.9% (mean 85 words) |
| BM25 vs dense on exact identifiers (R@1) | **0.300 vs 0.060** |
| Reranking lift on a weak first pass (R@1) | **0.256 → 0.483 (+89%)** |
| Faithfulness (RAGAS) | *pending measurement* |
| Faithfulness before/after reranking | *pending measurement* |
| Answer relevance / context precision / recall | *pending measurement* |
| Refusal accuracy on the golden set | *pending measurement* |

Retrieval numbers and the reason the most eye-catching one in them is a trap:
[`docs/retrieval_findings.md`](docs/retrieval_findings.md).

## Retrieval

Dense (BGE embeddings over ChromaDB) and sparse (BM25) each propose 30
candidates; reciprocal rank fusion merges them; a cross-encoder reranks that
shortlist down to the 5 passages the model sees. Wide-then-narrow matters: a
reranker can only reorder what it is given, so a passage dense retrieval put
12th can never reach first place if the shortlist was only 5 long.

Both indexes are built from the **same** source -- the sparse index reads the
vector store's contents, not the chunk file -- because two independently built
indexes drift, and a BM25 hit for a chunk the store lacks yields a citation
whose click-through 404s.

Measured before/after numbers, and why the most eye-catching number in them is
a trap worth understanding, are in
[`docs/retrieval_findings.md`](docs/retrieval_findings.md).

## Citations

Every answer sentence carries a `[S1]`-style marker that resolves to an exact
passage, and `GET /chunk/{chunk_id}` returns that passage in full so a reader
can land on the paragraph the claim came from.

The `S` prefix is load-bearing. Academic prose is dense with its own reference
markers -- *"generative retrieval in industrial search [4, 15, 20, 26]"* -- so
a plain `[n]` scheme cannot distinguish a citation to passage 4 from the source
paper's own bibliography entry 4. Quoting a passage verbatim then yields an
answer that appears to cite passages which may not exist. `[S4]` cannot
collide.

## Refusing to answer

The system declines rather than guessing, and it checks two *different*
properties to decide.

**Grounding** -- is each sentence actually supported by the passage it cites?
Every claim is verified independently, cheapest check first:

1. *Content-word coverage* against the cited passage. Fast, but a weak proxy
   for entailment on its own: "X improves Y" and "X does not improve Y"
   overlap almost completely.
2. *Numeric agreement* -- every figure in a claim must appear in the cited
   passage. The highest-precision cheap check available on an academic corpus:
   a claim of "2.5 BLEU" against a passage saying "3.1 BLEU" has flawless word
   overlap and is simply false.
3. *Negation parity* -- a claim that negates where its passage does not is
   unsupported however well the words match. This is the case coverage cannot
   see.

An LLM judge settles only the ambiguous middle, so its cost tracks ambiguity
rather than answer length. Hard numeric and negation failures are never
escalated: a model must not be able to talk itself into accepting 2.5 where
the passage says 3.1. If fewer than `min_supported_ratio` (0.8) of claims
survive, the answer is refused and the failing verdicts are shown.

**Relevance** -- do the passages address the question at all? This is a
separate property, and conflating the two is a real trap: asked *"what is the
capital of Mongolia?"*, an early build retrieved a passage about Toronto
weather stations, quoted it faithfully, and scored **100% supported**. It was
perfectly grounded and completely useless. The cross-encoder settles this,
since query-passage relevance is exactly what it is trained for. Measured on
this corpus, on-topic questions score −4.8 to +7.2 for their best passage and
off-topic questions −11.0 to −7.8, so the gate sits at −7.0. That number is a
raw logit for one specific reranker model -- change the model and it must be
re-measured.

On a 14-question probe (8 answerable, 6 not) the system currently answers 8/8
and refuses 6/6. That is a smoke test, not an evaluation; the labelled version
arrives with the golden dataset in Phase 6.

Refusals are detected from a sentinel the prompt mandates rather than by
pattern-matching apologetic prose, which varies by model and is unreliable to
parse. Over the API a refusal is a **200 with `status: refused_*`** -- it is a
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

MIT

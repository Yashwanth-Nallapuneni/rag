# ragpipe — a production-grade RAG system

Ask questions over an arXiv ML corpus and get answers that **cite the exact
passage they came from** — or an explicit refusal when the retrieved passages
don't support an answer.

The point of this project is not the demo. It is the discipline around it:
hybrid retrieval, cross-encoder reranking, enforced citations, and an
evaluation harness that **gates CI on answer faithfulness**.

> Status: Phase 3 of 6 complete. Ingestion, chunking, embedding, dense
> retrieval, cited answers and a FastAPI service all work end to end.
> Hybrid retrieval and reranking are Phase 4; the RAGAS CI gate is Phase 6.
> 156 tests passing.

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
make test        # 156 offline tests
```

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

When the retrieved passages do not support an answer, the system refuses. The
refusal is detected from a sentinel the prompt mandates rather than by
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

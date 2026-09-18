# ragpipe — a production-grade RAG system

Ask questions over an arXiv ML corpus and get answers that **cite the exact
passage they came from** — or an explicit refusal when the retrieved passages
don't support an answer.

The point of this project is not the demo. It is the discipline around it:
hybrid retrieval, cross-encoder reranking, enforced citations, and an
evaluation harness that **gates CI on answer faithfulness**.

> Status: Phase 1 of 6 complete (scaffolding, config, provider abstraction, corpus).
> Numbers and architecture diagram land in Phase 6.

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
make test        # fast offline tests
```

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

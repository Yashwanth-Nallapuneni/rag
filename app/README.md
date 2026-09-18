# ragpipe Streamlit demo

## Run it

```bash
cd /Users/yashwanth/Desktop/claude/rag
PYTHONPATH=src .venv/bin/streamlit run app/streamlit_app.py
```

The index must already exist (`make ingest && make index`) — if the vector
store is empty the app shows an actionable message instead of a traceback.

## Caveat: answers are extractive, not generated

The default LLM provider (`llm.provider: mock`) is a deterministic, offline,
**extractive** model (`src/ragpipe/providers/llm/mock.py`). It does not write
prose — it selects and stitches together the best-matching *sentences already
present in the retrieved passages*, each tagged with the `[Sn]` marker of the
passage it came from. There is no paraphrasing and no synthesis.

That is deliberate: it lets the whole pipeline — retrieval, fusion,
reranking, citation resolution, claim verification, and refusal — run
end-to-end with no API key, no cost, and no run-to-run variance. This demo
exists to make **that** machinery visible, not to show off fluent writing.
Point `llm.provider` at `anthropic`/`openai`/`ollama` in config to see
generated prose instead; the UI is unchanged either way.

## What each panel shows

**Sidebar — Pipeline configuration**
`top_k`, retrieval mode (dense/sparse/hybrid), fusion method, reranking
on/off, citation enforcement on/off, and the answer prompt version. Changing
any of these rebuilds the `Answerer` (see cache key note below) and the next
question runs against the new configuration.

**Sidebar — Config / Store / Provider health**
`settings.describe()` and the config fingerprint (`settings.fingerprint()`),
so a viewer always knows exactly which configuration produced what's on
screen; the vector store's `stats()` and document count; and the LLM
provider's `health()`.

**Ask a question**
A text box plus clickable presets. The "on-topic" presets exercise a normal
answer; the "questions the corpus cannot answer" presets deliberately trigger
the refusal path (e.g. asking an ML-paper corpus for a football score) so a
visitor can see it on purpose, not by accident.

**Answer**
The generated text with its `[Sn]` markers, followed by a marker legend
mapping each one to its source document — the visible link between a claim
and its evidence.

**Refusal (when it happens)**
Refusal is a first-class outcome of this pipeline, not an error, so it is
rendered as the headline: the `AnswerStatus`, the human-readable
`refusal_reason` (e.g. "best passage scored -10.98 for relevance... below
the -7.00 threshold"), and — critically — the passages that *were* retrieved
anyway, so you can see why they weren't good enough.

**Citations**
One expander per cited passage: the `[Sn]` marker, document title, section
path, page number, every retrieval score that's present (dense / sparse /
fusion / rerank / combined), and the full passage text. This is the
click-through — you can land on the exact paragraph a claim came from.

**Verification (claim-by-claim)**
The most important panel. One row per `ClaimVerdict` from
`answer.claim_verdicts`: the claim text, supported yes/no, the support
score, the reason string (e.g. "83% of the claim's terms appear in the cited
passage", or "negates what the cited passage states"), and which chunk ids
it was checked against. The overall supported ratio is shown against
`settings.citation.min_supported_ratio` so it's obvious how close the answer
came to being refused for insufficient support.

**Retrieval detail**
Every retrieved passage, not just the cited ones, each tagged `cited` or
`retrieved, not cited`, with its full per-stage score breakdown — this is
where the effect of fusion and reranking becomes visible.

**Timings**
Per-stage wall time from `answer.timings_ms` (retrieval / context /
generation / verification), plus overall wall time for the request.

## Cache key (why the sidebar controls actually take effect)

The `Answerer` is expensive to build — it loads a sentence-transformers
embedder and a cross-encoder reranker — so it's cached with
`@st.cache_resource`, keyed on a tuple of exactly the settings fields the
sidebar exposes: retrieval mode, `top_k`, fusion method, rerank enabled,
citation enforcement, and prompt version. Anything not in that tuple (env
vars, eval config, logging, etc.) is not something the UI lets a viewer
change, so excluding it avoids spurious cache misses and model reloads; the
full `overrides` dict is also passed through as a cache argument, which is
redundant with the explicit tuple but harmless. See the comment on
`_get_answerer` in `streamlit_app.py` for the reasoning in full.

## Scope

This file and `streamlit_app.py` are the only files in `app/`. Nothing
outside `app/` was modified to build this demo; the app calls the pipeline
through its public API (`build_answerer`, `Answerer.answer`,
`VectorStore.get/count/stats/document_ids`, `Settings.describe`) and does
not reimplement retrieval, verification, or citation parsing.

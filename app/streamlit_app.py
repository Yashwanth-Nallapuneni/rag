"""Streamlit demo UI for the ragpipe RAG pipeline.

This file is a *view* over the pipeline in `src/ragpipe`. It builds an
`Answerer` from `Settings` and renders whatever it returns -- retrieval hits,
citations, claim verdicts, refusals -- faithfully. It does not re-implement
retrieval, fusion, reranking, citation parsing or claim verification: all of
that already lives in the library and is exercised here through its public
API (`build_answerer`, `Answerer.answer`, `VectorStore.get/count/stats`,
`Settings.describe`). Search this file for "ragpipe." to see every call out
to the pipeline.

Run with:
    PYTHONPATH=src streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import streamlit as st

# The app lives in app/, the package in src/ -- add src/ to the path the same
# way `PYTHONPATH=src` would, so `streamlit run app/streamlit_app.py` works
# even if the caller forgot the env var.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ragpipe.config import Settings, load_settings  # noqa: E402
from ragpipe.generation.answerer import Answerer, build_answerer  # noqa: E402
from ragpipe.index.builder import get_store  # noqa: E402
from ragpipe.schemas import Answer, RetrievedChunk  # noqa: E402

st.set_page_config(page_title="ragpipe demo", layout="wide")

ON_TOPIC_EXAMPLES = [
    "what is FL-Net?",
    "how are the models evaluated?",
    "what is federated learning used for in clinical research?",
]
OFF_TOPIC_EXAMPLES = [
    "What is the capital city of Mongolia?",
    "Who won the 1998 FIFA World Cup final?",
]


# --------------------------------------------------------------------------
# Settings <-> sidebar controls
# --------------------------------------------------------------------------
def _sidebar_overrides() -> dict:
    """Render the sidebar controls and return a `load_settings(overrides=...)`
    -compatible dict. Nothing here touches retrieval/citation logic -- it only
    edits the config that the real pipeline consumes."""
    st.sidebar.header("Pipeline configuration")

    top_k = st.sidebar.slider("top_k (passages to the generator)", 1, 10, 5)
    mode = st.sidebar.selectbox("Retrieval mode", ["hybrid", "dense", "sparse"], index=0)
    fusion = st.sidebar.selectbox("Fusion method", ["rrf", "weighted_sum"], index=0)
    rerank_on = st.sidebar.checkbox("Reranking enabled", value=True)
    enforce_citations = st.sidebar.checkbox("Citation enforcement", value=True)
    prompt_version = st.sidebar.selectbox("Answer prompt version", ["v2", "v1"], index=0)

    return {
        "retrieval": {"mode": mode, "top_k": top_k, "fusion": fusion},
        "rerank": {"enabled": rerank_on},
        "citation": {"enforce": enforce_citations},
        "prompts": {"answer_version": prompt_version},
    }


@st.cache_resource(show_spinner="Loading models and building the answerer...")
def _get_answerer(cache_key: tuple, overrides: dict) -> Answerer:
    """Cache the Answerer, not just the Settings.

    `build_answerer` loads a sentence-transformers embedder and a
    cross-encoder reranker (seconds, not milliseconds) plus a BM25 index off
    the store, so this must not run per Streamlit rerun (every widget
    interaction triggers one).

    Cache key: `cache_key` is a flat tuple of exactly the settings fields
    that change *what gets built* -- retrieval mode/top_k/fusion, whether
    reranking is on, whether citation enforcement is on, and the prompt
    version -- i.e. precisely the knobs exposed in the sidebar. `overrides`
    (the dict actually passed to `load_settings`) is also part of the
    Streamlit cache key by value, which is redundant with `cache_key` but
    harmless; `cache_key` is what we reason about by hand.

    Deliberately NOT in the key: env-only settings, eval config, logging,
    etc. -- fields a viewer cannot change from this UI, so including them
    would only cause spurious cache misses. Too coarse a key (e.g. caching
    once globally) would make the sidebar controls silently do nothing;
    too fine (e.g. keying on the whole Settings object, which includes
    machine-specific paths) would reload the cross-encoder on unrelated
    changes. This tuple is the exact set of levers the sidebar exposes.
    """
    settings = load_settings(overrides=overrides)
    return build_answerer(settings)


def _settings_cache_key(settings: Settings) -> tuple:
    return (
        settings.retrieval.mode,
        settings.retrieval.top_k,
        settings.retrieval.fusion,
        settings.rerank.enabled,
        settings.citation.enforce,
        settings.prompts.answer_version,
    )


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def _score_row(rc: RetrievedChunk) -> dict:
    return {
        "rank": rc.rank,
        "retriever": rc.retriever,
        "dense": rc.dense_score,
        "sparse": rc.sparse_score,
        "fusion": rc.fusion_score,
        "rerank": rc.rerank_score,
        "combined score": rc.score,
    }


def _render_answer_text(answer: Answer) -> None:
    """Render the answer text with its [S1]-style markers, each linked to an
    anchor in the citation panel below via Streamlit's markdown anchors."""
    st.subheader("Answer")
    text = answer.text
    # Streamlit markdown doesn't support real in-page anchors reliably across
    # versions, so instead we render the raw text (markers visible as-is)
    # and immediately follow with a marker legend that maps each [Sn] to
    # its citation entry -- the "obvious at a glance" link the spec asks for.
    st.markdown(f"> {text}")
    if answer.citations:
        legend = "  ".join(
            f"**[S{c.marker}]** → *{c.doc_title}*" for c in answer.citations
        )
        st.caption(f"Markers: {legend}")


def _render_refusal(answer: Answer) -> None:
    """A refusal is the headline, not an error banner."""
    st.warning(f"**Refused to answer** — status: `{answer.status.value}`")
    st.markdown(f"**Why:** {answer.refusal_reason}")
    st.markdown(f"*Message shown to the user:* {answer.text}")
    if answer.contexts:
        st.markdown("**Passages retrieved anyway** (this is why it refused, not despite it):")
        for rc in answer.contexts:
            with st.expander(f"{rc.chunk.doc_title} — {rc.chunk.locator()}"):
                st.write(rc.chunk.text)
                st.json(_score_row(rc))
    else:
        st.caption("No passages cleared the retrieval threshold at all.")


def _render_citations(answer: Answer) -> None:
    st.subheader("Citations (click-through to source passages)")
    if not answer.citations:
        st.caption("No citations on this answer.")
        return
    # Build a lookup so we can show retrieval scores alongside each citation.
    by_id = {rc.chunk_id: rc for rc in answer.contexts}
    for c in answer.citations:
        rc = by_id.get(c.chunk_id)
        header = f"[S{c.marker}] {c.doc_title}"
        if c.section_path:
            header += f" › {' > '.join(c.section_path)}"
        if c.page_start is not None:
            header += f" (p. {c.page_start})"
        with st.expander(header):
            st.markdown(f"**Locator:** {c.locator}")
            if rc is not None:
                cols = st.columns(5)
                labels = ["dense", "sparse", "fusion", "rerank", "combined"]
                values = [rc.dense_score, rc.sparse_score, rc.fusion_score, rc.rerank_score, rc.score]
                for col, label, val in zip(cols, labels, values):
                    col.metric(label, f"{val:.3f}" if val is not None else "—")
                st.markdown("**Full passage text:**")
                st.write(rc.chunk.text)
            else:
                st.caption("Chunk not present in this answer's retrieved contexts.")


def _render_verification(answer: Answer, settings: Settings) -> None:
    """The most important panel: visible evidence of the citation-enforcement
    trust property (see src/ragpipe/generation/verify.py)."""
    st.subheader("Verification (claim-by-claim)")
    if not answer.claim_verdicts:
        st.caption(
            "No claim verdicts on this answer (verifier disabled, or the "
            "answer was refused before verification ran)."
        )
        return

    ratio = sum(1 for v in answer.claim_verdicts if v.supported) / len(answer.claim_verdicts)
    threshold = settings.citation.min_supported_ratio
    delta = ratio - threshold
    st.metric(
        "Supported ratio vs. refusal threshold",
        f"{ratio:.0%}",
        delta=f"{delta:+.0%} vs. {threshold:.0%} threshold",
        delta_color="normal" if delta >= 0 else "inverse",
    )
    rows = [
        {
            "claim": v.claim,
            "supported": "✅" if v.supported else "❌",
            "support score": round(v.support_score, 3),
            "cited chunk ids": ", ".join(v.cited_chunk_ids) or "—",
            "reason": v.reason,
        }
        for v in answer.claim_verdicts
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_retrieval_detail(answer: Answer) -> None:
    st.subheader("Retrieval detail (every retrieved passage, not just cited)")
    if not answer.contexts:
        st.caption("Nothing was retrieved.")
        return
    cited_ids = {c.chunk_id for c in answer.citations}
    with st.expander(f"{len(answer.contexts)} retrieved passage(s)", expanded=False):
        for rc in answer.contexts:
            cited = rc.chunk_id in cited_ids
            tag = "cited" if cited else "retrieved, not cited"
            st.markdown(f"**{rc.chunk.doc_title}** — {rc.chunk.locator()}  `[{tag}]`")
            st.json(_score_row(rc))
            st.divider()


def _render_timings(answer: Answer) -> None:
    if not answer.timings_ms:
        return
    st.subheader("Per-stage timings")
    cols = st.columns(len(answer.timings_ms))
    for col, (stage, ms) in zip(cols, answer.timings_ms.items()):
        col.metric(stage, f"{ms:.1f} ms")


def _render_sidebar_health(settings: Settings, answerer: Answerer | None) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("Config")
    st.sidebar.caption(f"Fingerprint: `{settings.fingerprint()}`")
    st.sidebar.json(settings.describe())

    st.sidebar.subheader("Store")
    try:
        store = get_store(settings)
        stats = store.stats()
        st.sidebar.json(stats)
        st.sidebar.caption(f"{len(store.document_ids())} distinct documents indexed")
    except Exception as exc:  # store misconfigured/unreachable
        st.sidebar.error(f"Store unavailable: {exc}")

    st.sidebar.subheader("Provider health")
    if answerer is not None:
        try:
            st.sidebar.json(answerer.llm.health())
        except Exception as exc:
            st.sidebar.error(f"LLM health check failed: {exc}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    st.title("ragpipe — retrieval-augmented QA demo")
    st.caption(
        "Offline mock LLM: answers are EXTRACTIVE (sentences copied from "
        "retrieved passages), not fluent generated prose. This demo showcases "
        "retrieval, citation and refusal behaviour rather than writing "
        "quality. See app/README.md for details."
    )

    overrides = _sidebar_overrides()

    # Build a throwaway Settings just to compute the cache key and to have
    # something to describe/inspect even before an Answerer exists.
    try:
        probe_settings = load_settings(overrides=overrides)
    except Exception as exc:
        st.error(f"Invalid configuration: {exc}")
        return

    # Index-not-built guard, before we try to load models against it.
    try:
        store = get_store(probe_settings)
        if store.count() == 0:
            st.error(
                "The vector store is empty — no index has been built yet.\n\n"
                "Run `make ingest && make index` from the project root, then reload."
            )
            _render_sidebar_health(probe_settings, None)
            return
    except Exception as exc:
        st.error(
            "Could not reach the configured vector store: "
            f"{exc}\n\nRun `make ingest && make index` from the project root."
        )
        return

    cache_key = _settings_cache_key(probe_settings)
    try:
        answerer = _get_answerer(cache_key, overrides)
    except Exception as exc:
        st.error(f"Failed to build the answerer: {exc}")
        st.code(traceback.format_exc())
        return

    _render_sidebar_health(answerer.settings, answerer)

    st.subheader("Ask a question")
    st.caption("On-topic examples:")
    cols = st.columns(len(ON_TOPIC_EXAMPLES))
    for col, q in zip(cols, ON_TOPIC_EXAMPLES):
        if col.button(q, key=f"ontopic_{q}"):
            st.session_state["question"] = q

    st.caption("Questions the corpus cannot answer (watch the refusal path):")
    cols = st.columns(len(OFF_TOPIC_EXAMPLES))
    for col, q in zip(cols, OFF_TOPIC_EXAMPLES):
        if col.button(q, key=f"offtopic_{q}"):
            st.session_state["question"] = q

    question = st.text_input("Question", key="question")
    ask = st.button("Ask", type="primary")

    if not (ask and question.strip()):
        return

    # Never let a pipeline exception crash the app -- catch, show, stay usable.
    try:
        started = time.perf_counter()
        answer = answerer.answer(question)
        wall_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:
        st.error(f"The pipeline raised an exception while answering: {exc}")
        st.code(traceback.format_exc())
        return

    st.divider()
    if answer.refused:
        _render_refusal(answer)
    else:
        _render_answer_text(answer)
        _render_citations(answer)

    _render_verification(answer, answerer.settings)
    _render_retrieval_detail(answer)
    _render_timings(answer)
    st.caption(f"Wall time: {wall_ms:.0f} ms · model: {answer.model} · prompt: {answer.prompt_version}")


if __name__ == "__main__":
    main()

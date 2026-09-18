#!/usr/bin/env python
"""Human verification tool for the golden evaluation dataset.

One pair per screen: question, editable ground truth, and the full source
chunk with its locator so the reviewer can check correctness against the
corpus without leaving the screen. Every decision (approve / edit-and-approve
/ reject) is written to disk immediately -- there is no "save at the end" to
lose work to a closed tab.

Nothing here ever marks a pair verified on its own. Only a reviewer clicking
Approve does that; loading, drafting, or re-running never implicitly does.

Run with:
    PYTHONPATH=src .venv/bin/streamlit run scripts/review_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ragpipe.config import load_settings  # noqa: E402
from ragpipe.eval.golden import (  # noqa: E402
    default_review_path,
    load_dataset,
    load_review_ledger,
    record_decision,
    review_status,
    save_dataset,
    validate_dataset,
)
from ragpipe.ingest.pipeline import read_chunks  # noqa: E402

st.set_page_config(page_title="Golden dataset review", layout="wide")

SETTINGS = load_settings()


@st.cache_data(show_spinner=False)
def _load_chunk_map(chunks_path_str: str, mtime: float) -> dict:
    chunks = read_chunks(Path(chunks_path_str))
    return {c.chunk_id: c for c in chunks}


def _chunk_map() -> dict:
    chunks_path = SETTINGS.corpus.processed_path / "chunks.jsonl"
    if not chunks_path.exists():
        return {}
    return _load_chunk_map(str(chunks_path), chunks_path.stat().st_mtime)


def _inject_shortcuts() -> None:
    """Best-effort keyboard shortcuts (a=approve, r=reject, s=skip) by
    clicking the matching button in the parent document. Streamlit has no
    native keybinding API; this degrades gracefully to mouse-only if it
    doesn't attach."""
    st.components.v1.html(
        """
        <script>
        const doc = window.parent.document;
        if (!doc.__golden_shortcuts_bound) {
            doc.__golden_shortcuts_bound = true;
            doc.addEventListener('keydown', (e) => {
                if (e.target && ['TEXTAREA', 'INPUT'].includes(e.target.tagName)) return;
                const map = {a: 'Approve', e: 'Save edits & approve', r: 'Reject', s: 'Skip'};
                const label = map[e.key.toLowerCase()];
                if (!label) return;
                const buttons = Array.from(doc.querySelectorAll('button'));
                const btn = buttons.find(b => b.innerText.trim().startsWith(label));
                if (btn) { btn.click(); }
            });
        }
        </script>
        """,
        height=0,
    )


def main() -> None:
    st.title("Golden dataset review")

    with st.sidebar:
        st.text_input("Dataset path", value=str(SETTINGS.evaluation.dataset), key="dataset_path")
        st.text_input("Reviewer name / initials", value="", key="reviewer")
        only_draft = st.checkbox("Show only draft pairs", value=True, key="only_draft")
        page = st.radio("View", ["Review", "Summary"], index=0)

    dataset_path = Path(st.session_state["dataset_path"])
    review_path = default_review_path(dataset_path)

    if not dataset_path.exists():
        st.warning(f"No dataset at {dataset_path} yet. Run scripts/draft_golden.py first.")
        return

    pairs = load_dataset(dataset_path)
    ledger = load_review_ledger(review_path)
    by_id = {qa.id: qa for qa in pairs}

    total = len(pairs)
    verified = sum(1 for qa in pairs if review_status(qa.id, ledger) == "verified")
    rejected = sum(1 for qa in pairs if review_status(qa.id, ledger) == "rejected")
    st.progress(verified / total if total else 0.0)
    st.caption(f"{verified} / {total} verified  |  {rejected} rejected  |  {total - verified - rejected} remaining")

    if page == "Summary":
        st.subheader("Validation report")
        store = None
        try:
            from ragpipe.index.builder import get_store

            store = get_store(SETTINGS)
        except Exception:
            st.info("Vector store unavailable -- expected_chunk_ids not cross-checked against the index.")
        report = validate_dataset(pairs, store=store)
        st.code(report.summary(), language=None)
        if report.duplicate_questions:
            st.write("Duplicate question groups:", report.duplicate_questions)
        if report.missing_ground_truth:
            st.write("Missing ground truth:", report.missing_ground_truth)
        if report.unknown_chunk_ids:
            st.write("Unknown expected_chunk_ids:", report.unknown_chunk_ids)
        if report.unanswerable_with_expected_chunks:
            st.write(
                "Unanswerable pairs wrongly carrying expected chunks:",
                report.unanswerable_with_expected_chunks,
            )
        st.success("Set OK for eval harness use.") if report.ok else st.error(
            "Not ready -- fix the issues above (or keep reviewing until count is in range)."
        )
        return

    queue = sorted(
        (qa for qa in pairs if not only_draft or review_status(qa.id, ledger) == "draft"),
        key=lambda qa: qa.id,
    )

    if not queue:
        st.success("Nothing left to review with the current filter.")
        return

    queue_ids = [qa.id for qa in queue]
    if st.session_state.get("current_id") not in queue_ids:
        st.session_state["current_id"] = queue_ids[0]
    current_id = st.session_state["current_id"]
    idx = queue_ids.index(current_id)
    current = by_id[current_id]

    st.caption(f"Pair {idx + 1} / {len(queue)} in current filter  --  id: {current.id}")

    def _advance() -> None:
        nxt = idx + 1
        st.session_state["current_id"] = queue_ids[nxt] if nxt < len(queue_ids) else None

    col_q, col_src = st.columns([1, 1])

    with col_q:
        st.markdown(f"**Category:** `{current.category}`")
        if current.unanswerable:
            st.error(
                "UNANSWERABLE candidate -- there is no source chunk. Confirm the "
                "corpus genuinely cannot answer this before approving."
            )
        st.markdown("**Question**")
        st.info(current.question)
        gt = st.text_area("Ground truth (editable)", value=current.ground_truth, height=150, key=f"gt_{current.id}")
        notes = st.text_area(
            "Reviewer notes", value=current.notes, height=80, key=f"notes_{current.id}"
        )

        def _persist_edit(new_notes: str) -> None:
            if gt != current.ground_truth or new_notes != current.notes:
                current.ground_truth = gt
                current.notes = new_notes
                save_dataset(pairs, dataset_path)

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Approve", use_container_width=True):
            _persist_edit(notes)
            record_decision(review_path, current.id, "verified", reviewer=st.session_state["reviewer"], notes=notes)
            _advance()
            st.rerun()
        if b2.button("Save edits & approve", use_container_width=True):
            _persist_edit(notes or "edited during review")
            record_decision(review_path, current.id, "verified", reviewer=st.session_state["reviewer"], notes=notes)
            _advance()
            st.rerun()
        if b3.button("Reject", use_container_width=True):
            _persist_edit(notes)
            record_decision(review_path, current.id, "rejected", reviewer=st.session_state["reviewer"], notes=notes)
            _advance()
            st.rerun()
        if b4.button("Skip", use_container_width=True):
            _advance()
            st.rerun()

    with col_src:
        st.markdown("**Source**")
        if current.unanswerable:
            st.warning("No source chunk -- this pair is meant to be unanswerable from the corpus.")
            if current.doc_id:
                st.caption(f"Related document id: {current.doc_id}")
        elif current.expected_chunk_ids:
            chunk_map = _chunk_map()
            chunk = chunk_map.get(current.expected_chunk_ids[0])
            if chunk is None:
                st.error(f"Chunk {current.expected_chunk_ids[0]} not found in chunks.jsonl.")
            else:
                st.caption(chunk.locator())
                st.text_area("Full chunk text", value=chunk.text, height=400, disabled=True)
        else:
            st.error("No expected_chunk_ids set on an answerable pair -- flag for rejection.")

    _inject_shortcuts()
    st.caption("Shortcuts: a = approve, e = save edits & approve, r = reject, s = skip (when not typing).")


if __name__ == "__main__":
    main()

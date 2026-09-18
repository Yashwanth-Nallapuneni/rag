from __future__ import annotations

import pytest

from ragpipe.generation.answerer import Answerer
from ragpipe.retrieval.dense import DenseRetriever
from ragpipe.schemas import AnswerStatus


@pytest.fixture
def answerer(offline_store):
    settings, store = offline_store
    return Answerer(settings, store, retriever=DenseRetriever(settings, store))


@pytest.fixture
def grounded_question(corpus_chunks):
    """A question built from real indexed text, so support genuinely exists."""
    return " ".join(corpus_chunks[0].text.split()[:14])


def test_answers_a_grounded_question(answerer, grounded_question):
    a = answerer.answer(grounded_question)
    assert a.status == AnswerStatus.ANSWERED
    assert a.text.strip()
    assert a.contexts


def test_answer_cites_only_supplied_passages(answerer, grounded_question):
    a = answerer.answer(grounded_question)
    supplied = {rc.chunk_id for rc in a.contexts}
    assert a.citations, "a grounded answer must carry at least one citation"
    for c in a.citations:
        assert c.chunk_id in supplied


def test_every_citation_is_resolvable_to_a_real_chunk(
    answerer, grounded_question, offline_store
):
    """This is the click-through guarantee: a citation the user cannot follow
    is not a citation."""
    _, store = offline_store
    a = answerer.answer(grounded_question)
    for c in a.citations:
        fetched = store.get([c.chunk_id])
        assert fetched, f"citation {c.chunk_id} resolves to nothing"
        assert fetched[0].text.strip()
        assert c.locator


def test_refuses_when_nothing_relevant_is_retrieved(answerer):
    a = answerer.answer("What is the capital city of Mongolia and its population?")
    assert a.refused
    assert a.status in (
        AnswerStatus.REFUSED_BY_MODEL,
        AnswerStatus.REFUSED_NO_CONTEXT,
        AnswerStatus.REFUSED_LOW_SUPPORT,
    )
    assert a.refusal_reason
    assert not a.citations


def test_refusal_uses_the_configured_message(answerer, settings):
    a = answerer.answer("Completely unrelated Mongolian geography trivia?")
    if a.refused:
        assert a.text.strip().startswith(
            settings.citation.refusal_message.strip().split()[0]
        )


def test_refuses_on_an_empty_index(corpus_chunks, tmp_path):
    from ragpipe.config import load_settings
    from ragpipe.index.chroma_store import ChromaStore

    s = load_settings(
        overrides={
            "llm": {"provider": "mock"},
            "embeddings": {"provider": "mock", "dimension": 384},
            "rerank": {"provider": "mock"},
            "vector_store": {"path": str(tmp_path / "empty"), "collection": "empty"},
        }
    )
    store = ChromaStore(s.vector_store, 384)
    a = Answerer(s, store).answer("anything at all")
    assert a.status == AnswerStatus.REFUSED_NO_CONTEXT


def test_answer_is_auditable(answerer, grounded_question):
    """Provenance on the answer itself is what makes a metric shift
    explainable later."""
    a = answerer.answer(grounded_question)
    assert a.prompt_version == "answer/v2"
    assert a.model and a.model.startswith("mock:")
    assert a.config_fingerprint
    assert {"retrieval", "context", "generation"} <= set(a.timings_ms)


def test_top_k_is_respected(answerer, grounded_question):
    assert len(answerer.answer(grounded_question, k=2).contexts) <= 2


def test_is_deterministic_offline(answerer, grounded_question):
    """CI can only gate on faithfulness if the same input gives the same
    answer."""
    first = answerer.answer(grounded_question)
    second = answerer.answer(grounded_question)
    assert first.text == second.text
    assert [c.chunk_id for c in first.citations] == [c.chunk_id for c in second.citations]


def test_empty_question_refuses(answerer):
    assert answerer.answer("   ").status == AnswerStatus.REFUSED_NO_CONTEXT


def test_source_reference_markers_never_become_citations(answerer, corpus_chunks):
    """Real corpus text contains "[4, 15, 20]"; none of those may show up as
    a citation, and none may be reported as unresolved."""
    a = answerer.answer(" ".join(corpus_chunks[3].text.split()[:14]))
    supplied = set(range(1, len(a.contexts) + 1))
    for c in a.citations:
        assert c.marker in supplied
    assert a.usage.get("unresolved_citations", 0) == 0

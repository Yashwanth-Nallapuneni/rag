from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ragpipe.api.app import create_app


@pytest.fixture
def client(offline_store):
    """The app is built against the populated temp store, so the API is
    exercised end to end with no network and no model downloads."""
    settings, _ = offline_store
    app = create_app(settings)
    with TestClient(app) as c:
        c.ragpipe_settings = settings
        yield c


@pytest.fixture
def grounded_question(corpus_chunks):
    return " ".join(corpus_chunks[0].text.split()[:14])


def test_index_lists_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200
    body = str(r.json())
    for route in ("/query", "/chunk", "/health", "/stats"):
        assert route in body


def test_health_is_200_and_reports_providers(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "providers" in body or "llm" in str(body)
    assert "ready" in body


def test_stats_reports_corpus_size(client, corpus_chunks):
    r = client.get("/stats")
    assert r.status_code == 200
    assert str(len(corpus_chunks)) in str(r.json())


def test_query_returns_a_cited_answer(client, grounded_question):
    r = client.post("/query", json={"question": grounded_question})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]["text"].strip()
    assert body["answer"]["status"] == "answered"
    assert body["answer"]["citations"]


def test_citations_point_into_the_returned_contexts(client, grounded_question):
    body = client.post("/query", json={"question": grounded_question}).json()
    # Contexts are RetrievedChunk objects, so the chunk is nested.
    context_ids = {c["chunk"]["chunk_id"] for c in body["answer"]["contexts"]}
    for citation in body["answer"]["citations"]:
        assert citation["chunk_id"] in context_ids


def test_citation_click_through_returns_the_full_passage(client, grounded_question):
    """The Phase 1 deliverable in the spec: click a citation, land on the
    exact paragraph. A snippet would not satisfy that."""
    body = client.post("/query", json={"question": grounded_question}).json()
    citation = body["answer"]["citations"][0]
    context = next(
        c["chunk"]
        for c in body["answer"]["contexts"]
        if c["chunk"]["chunk_id"] == citation["chunk_id"]
    )

    r = client.get(f"/chunk/{citation['chunk_id']}")
    assert r.status_code == 200
    chunk = r.json()["chunk"]
    assert chunk["text"] == context["text"], "click-through text must be the full chunk"
    assert chunk["doc_title"]
    assert chunk["page_start"] is not None or chunk["section_path"]
    assert len(chunk["text"]) > 200, "a citation target must be the passage, not a preview"


def test_unknown_chunk_is_404(client):
    assert client.get("/chunk/does-not-exist").status_code == 404


def test_blank_question_is_422(client):
    assert client.post("/query", json={"question": "   "}).status_code == 422
    assert client.post("/query", json={}).status_code == 422


def test_overlong_question_is_rejected(client):
    r = client.post("/query", json={"question": "a" * 5000})
    assert r.status_code == 422


def test_refusal_is_a_200_not_an_error(client):
    """A refusal is a product behaviour the client must be able to render,
    so it cannot be signalled as an HTTP failure."""
    r = client.post(
        "/query", json={"question": "What is the capital city of Mongolia?"}
    )
    assert r.status_code == 200
    answer = r.json()["answer"]
    assert answer["status"].startswith("refused")
    assert not answer["citations"]


def test_request_id_is_returned(client, grounded_question):
    r = client.post("/query", json={"question": grounded_question})
    assert r.headers.get("X-Request-ID")


def test_top_k_is_honoured(client, grounded_question):
    body = client.post("/query", json={"question": grounded_question, "k": 2}).json()
    assert len(body["answer"]["contexts"]) <= 2


def test_openapi_documents_every_route(client):
    spec = client.get("/openapi.json").json()
    for path in ("/query", "/chunk/{chunk_id}", "/health", "/stats", "/"):
        assert path in spec["paths"], f"{path} missing from OpenAPI"


def test_answerer_is_built_once(offline_store, monkeypatch):
    """Rebuilding per request would reload a sentence-transformers model on
    every call."""
    settings, _ = offline_store
    import ragpipe.api.app as app_module

    calls = {"n": 0}
    real = app_module.build_answerer

    def counting(s):
        calls["n"] += 1
        return real(s)

    monkeypatch.setattr(app_module, "build_answerer", counting)
    app = create_app(settings)
    with TestClient(app) as c:
        for _ in range(3):
            c.get("/health")
            c.post("/query", json={"question": "attention mechanism"})
    assert calls["n"] == 1, f"answerer built {calls['n']} times"

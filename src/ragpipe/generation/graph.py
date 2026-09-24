"""LangGraph orchestration for the answer pipeline.

`Answerer.answer()` used to be one long function with refusal checks as
early `return`s buried between stages. That reads fine linearly but hides
the actual shape of the pipeline: it is a state machine with four points
where it can bail out before finishing. This module makes that shape
explicit as a `StateGraph` -- one node per stage, real conditional edges
for every branch -- and `Answerer` delegates to the compiled graph instead
of re-implementing the flow inline.

Semantics are preserved exactly, not approximated: every refusal reason
string, status value, timings key and usage key here is copied verbatim
from the function this replaces. The graph is compiled once per `Answerer`
(construction is not free) and re-invoked per query.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..logging_utils import get_logger, timed
from ..providers import LLMRequest
from ..schemas import AnswerStatus, Citation, ClaimVerdict, RetrievedChunk
from .citations import (
    has_suspect_markers,
    normalize_citation_markers,
    resolve_citations,
    split_claims,
)
from .context import RenderedContext, render_context

if TYPE_CHECKING:
    from .answerer import Answerer

log = get_logger(__name__)

# Cheap presence check so the "unrecognised marker format" warning only fires
# when no usable marker survived normalisation.
_MARKER_PRESENT_RE = re.compile(r"\[S\d{1,3}\]")


class GraphState(TypedDict, total=False):
    """Everything a stage might read or write. `total=False`: a stage only
    sets the keys it owns, and earlier keys just carry forward untouched."""

    question: str
    k: int | None
    where: dict[str, Any] | None

    candidates: list[RetrievedChunk]
    contexts: list[RetrievedChunk]
    rendered: RenderedContext
    response_text: str
    usage: dict[str, int]
    citations: list[Citation]
    unknown_citations: list[int]
    uncited_claims: list[str]
    claim_verdicts: list[ClaimVerdict]
    supported_ratio: float

    timings_ms: dict[str, float]

    # Set by any stage that decides to refuse; its presence is what the
    # conditional edges branch on.
    status: AnswerStatus
    refusal_reason: str
    refusal_contexts: list[RetrievedChunk]

    answer: Any  # Answer, left untyped here to avoid importing pydantic model cycles


def _route_on_status(state: GraphState) -> str:
    return "refuse" if state.get("status") else "continue"


# -- node factories -------------------------------------------------------
# Each node closes over the `Answerer` instance so it can call the real
# retriever / LLM / verifier through their current attributes (tests
# monkeypatch `answerer.llm.complete` and `answerer.retriever.retrieve`
# after construction, so lookups must stay dynamic rather than snapshotted).


def _make_retrieve(answerer: Answerer):
    def retrieve(state: GraphState) -> dict[str, Any]:
        timings = dict(state.get("timings_ms") or {})
        with timed(timings, "retrieval"):
            candidates = answerer.retriever.retrieve(
                state["question"], where=state.get("where")
            )
        top_k = state.get("k") or answerer.settings.retrieval.top_k
        contexts = candidates[:top_k]

        update: dict[str, Any] = {
            "candidates": candidates,
            "contexts": contexts,
            "timings_ms": timings,
        }
        if not contexts:
            update["status"] = AnswerStatus.REFUSED_NO_CONTEXT
            update["refusal_reason"] = "retrieval returned no passages above threshold"
            update["refusal_contexts"] = []
        return update

    return retrieve


def _make_relevance_gate(answerer: Answerer):
    def relevance_gate(state: GraphState) -> dict[str, Any]:
        # Checked BEFORE generation so an off-topic question costs nothing to
        # refuse. Grounding and relevance are different properties: a
        # faithful quotation of an irrelevant passage passes every citation
        # check and still fails the user.
        contexts = state["contexts"]
        cfg = answerer.settings.citation
        gate = cfg.min_relevance_score
        if cfg.enforce and gate is not None:
            scored = [c.rerank_score for c in contexts if c.rerank_score is not None]
            if scored and max(scored) < gate:
                return {
                    "status": AnswerStatus.REFUSED_NO_CONTEXT,
                    "refusal_reason": (
                        f"best passage scored {max(scored):.2f} for relevance to "
                        f"this question, below the {gate:.2f} threshold: the "
                        f"corpus does not appear to cover it"
                    ),
                    "refusal_contexts": contexts,
                }
        return {}

    return relevance_gate


def _make_build_context(answerer: Answerer):
    def build_context(state: GraphState) -> dict[str, Any]:
        timings = dict(state.get("timings_ms") or {})
        with timed(timings, "context"):
            rendered = render_context(
                state["contexts"],
                answerer.settings.generation.max_context_tokens,
                include_locators=answerer.settings.generation.include_locators,
            )
        if rendered.dropped:
            log.info(
                "context budget dropped %d of %d passages",
                len(rendered.dropped),
                len(state["contexts"]),
            )
        return {"rendered": rendered, "timings_ms": timings}

    return build_context


def _make_generate(answerer: Answerer):
    def generate(state: GraphState) -> dict[str, Any]:
        rendered: RenderedContext = state["rendered"]
        system, user = answerer.prompt.render(
            context=rendered.text, question=state["question"]
        )

        timings = dict(state.get("timings_ms") or {})
        with timed(timings, "generation"):
            response = answerer.llm.complete(
                LLMRequest(
                    system=system,
                    user=user,
                    task="answer",
                    temperature=answerer.settings.llm.temperature,
                    max_tokens=answerer.settings.llm.max_tokens,
                )
            )

        # Normalise citation markers before anything downstream reads them.
        # Groq's gpt-oss-120b cites as 【S1】 (CJK brackets), which the ASCII
        # parser cannot see -- left unnormalised, every answer looks uncited
        # and gets refused despite being correctly attributed.
        text = normalize_citation_markers(response.text).strip()
        if not _MARKER_PRESENT_RE.search(text) and has_suspect_markers(text):
            log.warning(
                "answer cites in an unrecognised marker format; citations will "
                "not resolve. First 160 chars: %r",
                text[:160],
            )
        update: dict[str, Any] = {
            "response_text": text,
            "usage": dict(response.usage),
            "timings_ms": timings,
        }
        if answerer.sentinel and answerer.sentinel in text:
            update["status"] = AnswerStatus.REFUSED_BY_MODEL
            update["refusal_reason"] = (
                "model reported the passages do not support an answer"
            )
            update["refusal_contexts"] = rendered.used
        return update

    return generate


def _make_parse_citations(answerer: Answerer):
    def parse_citations(state: GraphState) -> dict[str, Any]:
        rendered: RenderedContext = state["rendered"]
        text = state["response_text"]
        citations, unknown = resolve_citations(text, rendered)
        claims = split_claims(text)
        uncited = [c.text for c in claims if c.uncited]
        return {
            "citations": citations,
            "unknown_citations": unknown,
            "uncited_claims": uncited,
        }

    return parse_citations


def _make_verify(answerer: Answerer):
    def verify(state: GraphState) -> dict[str, Any]:
        if answerer.verifier is None:
            # No verifier configured: nothing to check, so nothing is
            # unsupported. Mirrors the un-enforced default in `answer()`.
            return {"claim_verdicts": [], "supported_ratio": 1.0}

        rendered: RenderedContext = state["rendered"]
        timings = dict(state.get("timings_ms") or {})
        with timed(timings, "verification"):
            verdicts, supported_ratio = answerer.verifier.verify(
                state["response_text"], rendered, state["question"]
            )

        update: dict[str, Any] = {
            "claim_verdicts": verdicts,
            "supported_ratio": supported_ratio,
            "timings_ms": timings,
        }
        cfg = answerer.settings.citation
        if cfg.enforce and supported_ratio < cfg.min_supported_ratio:
            update["status"] = AnswerStatus.REFUSED_LOW_SUPPORT
            update["refusal_reason"] = (
                f"only {supported_ratio:.0%} of claims are supported by the "
                f"retrieved passages (threshold {cfg.min_supported_ratio:.0%})"
            )
            update["refusal_contexts"] = rendered.used
        return update

    return verify


def _make_finalize(answerer: Answerer):
    def finalize(state: GraphState) -> dict[str, Any]:
        answer = answerer._finalize(
            question=state["question"],
            text=state["response_text"],
            citations=state.get("citations", []),
            contexts=state["rendered"].used,
            verdicts=state.get("claim_verdicts", []),
            timings=state.get("timings_ms", {}),
            usage=state.get("usage", {}),
            unknown=state.get("unknown_citations", []),
            uncited=state.get("uncited_claims", []),
        )
        return {"answer": answer}

    return finalize


def _make_refuse(answerer: Answerer):
    def refuse(state: GraphState) -> dict[str, Any]:
        answer = answerer._refuse(
            state["question"],
            state["status"],
            state["refusal_reason"],
            state.get("refusal_contexts", []),
            state.get("timings_ms", {}),
            state.get("claim_verdicts"),
            state.get("usage"),
        )
        return {"answer": answer}

    return refuse


# -- graph assembly ---------------------------------------------------------

_NODES = (
    "retrieve",
    "relevance_gate",
    "build_context",
    "generate",
    "parse_citations",
    "verify",
    "finalize",
    "refuse",
)

# The stages with a conditional exit to `refuse`; kept alongside the node
# list so `describe_graph()` and `build_graph()` cannot drift apart.
_GATED_NODES = ("retrieve", "relevance_gate", "generate", "verify")


def build_graph(answerer: Answerer):
    """Compile the state machine for one `Answerer`. Call once at
    construction -- compilation is not free, and the graph has no per-query
    state of its own to invalidate."""
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", _make_retrieve(answerer))
    graph.add_node("relevance_gate", _make_relevance_gate(answerer))
    graph.add_node("build_context", _make_build_context(answerer))
    graph.add_node("generate", _make_generate(answerer))
    graph.add_node("parse_citations", _make_parse_citations(answerer))
    graph.add_node("verify", _make_verify(answerer))
    graph.add_node("finalize", _make_finalize(answerer))
    graph.add_node("refuse", _make_refuse(answerer))

    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve", _route_on_status, {"refuse": "refuse", "continue": "relevance_gate"}
    )
    graph.add_conditional_edges(
        "relevance_gate",
        _route_on_status,
        {"refuse": "refuse", "continue": "build_context"},
    )
    graph.add_edge("build_context", "generate")
    graph.add_conditional_edges(
        "generate", _route_on_status, {"refuse": "refuse", "continue": "parse_citations"}
    )
    graph.add_edge("parse_citations", "verify")
    graph.add_conditional_edges(
        "verify", _route_on_status, {"refuse": "refuse", "continue": "finalize"}
    )
    graph.add_edge("finalize", END)
    graph.add_edge("refuse", END)

    return graph.compile()


def describe_graph() -> dict[str, Any]:
    """Static topology, independent of any `Answerer` instance -- for docs
    and for tests that want to assert the graph's real shape."""
    edges: list[tuple[str, str]] = [("__start__", "retrieve")]
    linear_next = {
        "retrieve": "relevance_gate",
        "relevance_gate": "build_context",
        "build_context": "generate",
        "generate": "parse_citations",
        "parse_citations": "verify",
        "verify": "finalize",
    }
    for node, nxt in linear_next.items():
        edges.append((node, nxt))
    for node in _GATED_NODES:
        edges.append((node, "refuse"))
    edges.append(("finalize", "__end__"))
    edges.append(("refuse", "__end__"))

    return {
        "nodes": list(_NODES),
        "edges": edges,
        "conditional_nodes": list(_GATED_NODES),
        "terminal_nodes": ["finalize", "refuse"],
    }

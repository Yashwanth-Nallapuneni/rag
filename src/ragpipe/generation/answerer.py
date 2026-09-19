"""The question-answering pipeline: retrieve, ground, generate, attribute.

Every stage records what it did on the returned `Answer` -- the passages used,
the markers that resolved, the ones that did not, per-stage timings, the
prompt version and the config fingerprint. That auditability is the point: an
answer you cannot trace is indistinguishable from a confident hallucination.

Refusal is a first-class outcome, not an error. The pipeline refuses when
retrieval returns nothing and when the model emits the prompt's refusal
sentinel. Claim-level verification (Phase 5) plugs in through `verifier`
without changing this flow.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..config import Settings
from ..index.base import VectorStore
from ..logging_utils import get_logger
from ..prompts import load_prompt
from ..providers import get_llm
from ..retrieval import Retriever, get_retriever
from ..schemas import Answer, AnswerStatus, Citation, ClaimVerdict, RetrievedChunk
from .context import RenderedContext
from .graph import build_graph

log = get_logger(__name__)

DEFAULT_REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"


class Verifier(Protocol):
    """Phase 5 citation enforcement plugs in here."""

    def verify(
        self, answer_text: str, rendered: RenderedContext, question: str
    ) -> tuple[list[ClaimVerdict], float]: ...


class Answerer:
    def __init__(
        self,
        settings: Settings,
        store: VectorStore,
        retriever: Retriever | None = None,
        verifier: Verifier | None = None,
    ):
        self.settings = settings
        self.store = store
        self.retriever = retriever or get_retriever(settings, store)
        self.llm = get_llm(settings)
        self.verifier = verifier
        self.prompt = load_prompt(
            "answer", settings.prompts.answer_version, str(settings.prompts.path)
        )
        self.sentinel = self.prompt.metadata.get(
            "refusal_sentinel", DEFAULT_REFUSAL_SENTINEL
        )
        # Compiled once: the graph IS the pipeline, not a decorative parallel
        # path, so `answer()` below only invokes it.
        self._graph = build_graph(self)

    # -- helpers --------------------------------------------------------
    def _refuse(
        self,
        question: str,
        status: AnswerStatus,
        reason: str,
        contexts: list[RetrievedChunk],
        timings: dict[str, float],
        verdicts: list[ClaimVerdict] | None = None,
        usage: dict[str, int] | None = None,
    ) -> Answer:
        # A refusal after generation still spent the generation tokens; drop
        # the usage here and the cost tracker under-reports a hard budget.
        return Answer(
            question=question,
            text=self.settings.citation.refusal_message.strip(),
            status=status,
            contexts=contexts,
            claim_verdicts=verdicts or [],
            refusal_reason=reason,
            prompt_version=self.prompt.id,
            model=f"{self.llm.name}:{self.llm.model}",
            timings_ms=timings,
            usage=dict(usage or {}),
            config_fingerprint=self.settings.fingerprint(),
        )

    def _finalize(
        self,
        question: str,
        text: str,
        citations: list[Citation],
        contexts: list[RetrievedChunk],
        verdicts: list[ClaimVerdict],
        timings: dict[str, float],
        usage: dict[str, int],
        unknown: list[int],
        uncited: list[str],
    ) -> Answer:
        answer = Answer(
            question=question,
            text=text,
            status=AnswerStatus.ANSWERED,
            citations=citations,
            contexts=contexts,
            claim_verdicts=verdicts,
            prompt_version=self.prompt.id,
            model=f"{self.llm.name}:{self.llm.model}",
            timings_ms=timings,
            usage=usage,
            config_fingerprint=self.settings.fingerprint(),
        )
        # Surfaced, not swallowed: a marker citing a passage that was never
        # supplied means the answer only looks grounded.
        if unknown:
            log.warning("answer cited unknown passages %s", unknown)
            answer.usage["unresolved_citations"] = len(unknown)
        if uncited:
            log.info("%d answer sentence(s) carry no citation", len(uncited))
            answer.usage["uncited_sentences"] = len(uncited)
        if verdicts:
            answer.usage["claims_checked"] = len(verdicts)
            answer.usage["claims_supported"] = sum(1 for v in verdicts if v.supported)
        return answer

    # -- main -----------------------------------------------------------
    def answer(
        self,
        question: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> Answer:
        """Run the graph. This IS the pipeline -- retrieve, gate, build
        context, generate, parse citations, verify, finalize-or-refuse --
        not a wrapper around a separate hand-rolled path."""
        result = self._graph.invoke(
            {"question": question, "k": k, "where": where, "timings_ms": {}}
        )
        return result["answer"]


def build_answerer(settings: Settings) -> Answerer:
    from ..index.builder import get_store
    from .verify import get_verifier

    return Answerer(settings, get_store(settings), verifier=get_verifier(settings))

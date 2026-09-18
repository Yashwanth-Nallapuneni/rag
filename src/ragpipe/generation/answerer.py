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
from ..logging_utils import get_logger, timed
from ..prompts import load_prompt
from ..providers import LLMRequest, get_llm
from ..retrieval import Retriever, get_retriever
from ..schemas import Answer, AnswerStatus, ClaimVerdict, RetrievedChunk
from .citations import resolve_citations, split_claims
from .context import RenderedContext, render_context

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

    # -- helpers --------------------------------------------------------
    def _refuse(
        self,
        question: str,
        status: AnswerStatus,
        reason: str,
        contexts: list[RetrievedChunk],
        timings: dict[str, float],
        verdicts: list[ClaimVerdict] | None = None,
    ) -> Answer:
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
            config_fingerprint=self.settings.fingerprint(),
        )

    # -- main -----------------------------------------------------------
    def answer(
        self,
        question: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> Answer:
        timings: dict[str, float] = {}
        top_k = k or self.settings.retrieval.top_k

        with timed(timings, "retrieval"):
            candidates = self.retriever.retrieve(question, where=where)
        contexts = candidates[:top_k]

        if not contexts:
            return self._refuse(
                question,
                AnswerStatus.REFUSED_NO_CONTEXT,
                "retrieval returned no passages above threshold",
                [],
                timings,
            )

        # Relevance gate, checked BEFORE generation so an off-topic question
        # costs nothing to refuse. Grounding and relevance are different
        # properties: a faithful quotation of an irrelevant passage passes
        # every citation check and still fails the user.
        gate = self.settings.citation.min_relevance_score
        if self.settings.citation.enforce and gate is not None:
            scored = [c.rerank_score for c in contexts if c.rerank_score is not None]
            if scored and max(scored) < gate:
                return self._refuse(
                    question,
                    AnswerStatus.REFUSED_NO_CONTEXT,
                    f"best passage scored {max(scored):.2f} for relevance to this "
                    f"question, below the {gate:.2f} threshold: the corpus does "
                    f"not appear to cover it",
                    contexts,
                    timings,
                )

        with timed(timings, "context"):
            rendered = render_context(
                contexts,
                self.settings.generation.max_context_tokens,
                include_locators=self.settings.generation.include_locators,
            )
        if rendered.dropped:
            log.info(
                "context budget dropped %d of %d passages",
                len(rendered.dropped),
                len(contexts),
            )

        system, user = self.prompt.render(context=rendered.text, question=question)

        with timed(timings, "generation"):
            response = self.llm.complete(
                LLMRequest(
                    system=system,
                    user=user,
                    task="answer",
                    temperature=self.settings.llm.temperature,
                    max_tokens=self.settings.llm.max_tokens,
                )
            )

        text = response.text.strip()
        if self.sentinel and self.sentinel in text:
            return self._refuse(
                question,
                AnswerStatus.REFUSED_BY_MODEL,
                "model reported the passages do not support an answer",
                rendered.used,
                timings,
            )

        citations, unknown = resolve_citations(text, rendered)
        claims = split_claims(text)
        uncited = [c.text for c in claims if c.uncited]

        verdicts: list[ClaimVerdict] = []
        supported_ratio = 1.0
        if self.verifier is not None:
            with timed(timings, "verification"):
                verdicts, supported_ratio = self.verifier.verify(
                    text, rendered, question
                )
            if (
                self.settings.citation.enforce
                and supported_ratio < self.settings.citation.min_supported_ratio
            ):
                return self._refuse(
                    question,
                    AnswerStatus.REFUSED_LOW_SUPPORT,
                    f"only {supported_ratio:.0%} of claims are supported by the "
                    f"retrieved passages (threshold "
                    f"{self.settings.citation.min_supported_ratio:.0%})",
                    rendered.used,
                    timings,
                    verdicts,
                )

        answer = Answer(
            question=question,
            text=text,
            status=AnswerStatus.ANSWERED,
            citations=citations,
            contexts=rendered.used,
            claim_verdicts=verdicts,
            prompt_version=self.prompt.id,
            model=f"{self.llm.name}:{self.llm.model}",
            timings_ms=timings,
            usage=response.usage,
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


def build_answerer(settings: Settings) -> Answerer:
    from ..index.builder import get_store
    from .verify import get_verifier

    return Answerer(settings, get_store(settings), verifier=get_verifier(settings))

"""Citation enforcement: does the answer's own evidence actually support it?

This is the system's central trust property. Retrieval and generation can both
succeed and still produce a confident sentence the cited passage never says --
and that failure is invisible to the user, because the citation marker looks
like proof. So every claim is checked against the passage it cites, and the
answer is refused outright when too few claims survive.

Three checks, cheapest first:

**Content-word coverage.** How much of the claim's substance appears in the
cited passage. Fast and catches wholesale invention, but on its own it is a
weak proxy for entailment: "X improves Y" and "X does not improve Y" overlap
almost completely.

**Numeric agreement.** Every number in a claim must appear in the cited
passage. In an academic corpus this is the highest-precision cheap check there
is -- a claim of "2.5 BLEU" against a passage saying "3.1 BLEU" has excellent
word overlap and is simply false.

**Negation parity.** A claim that negates where its passage does not (or the
reverse) is treated as unsupported however well the words match. This is the
specific failure coverage alone cannot see.

An LLM judge settles only what these cannot, so its cost tracks ambiguity
rather than answer length.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..config import Settings
from ..logging_utils import get_logger
from ..prompts import load_prompt
from ..providers import LLMRequest, get_llm
from ..schemas import ClaimVerdict, RetrievedChunk
from .citations import Claim, cited_chunks, split_claims
from .context import RenderedContext

log = get_logger(__name__)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_]*")
# Integers, decimals, percentages and scientific notation.
_NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?(?![\w])")
_NEGATION = {
    "not", "no", "never", "none", "neither", "nor", "without", "cannot",
    "cant", "doesnt", "dont", "isnt", "arent", "wasnt", "werent", "fails",
    "fail", "failed", "unable", "absent", "lacks", "lack", "nothing",
    "unsupported", "insufficient", "worse", "degrades", "decreases",
}
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "as", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "their", "our", "we", "they", "he", "she", "which", "who", "whom", "when",
    "where", "how", "why", "what", "can", "could", "should", "would", "will",
    "may", "might", "must", "have", "has", "had", "do", "does", "did", "also",
    "both", "each", "more", "most", "other", "such", "some", "any", "there",
    "here", "over", "under", "between", "about", "into", "through", "during",
}


def _content_words(text: str) -> set[str]:
    return {
        w.lower()
        for w in _WORD_RE.findall(text)
        if w.lower() not in _STOP and len(w) > 2
    }


# A leading "(6)" / "6." / "iii)" is list structure carried over from the
# source document, not a figure the claim asserts.
# The trailing separator must be a bracket, or a dot followed by whitespace.
# A bare `\.` would eat the "2." of "2.50 BLEU" and leave the figure 50,
# turning a correct claim into a fabricated one.
_ENUM_PREFIX_RE = re.compile(
    r"^\s*[\(\[]?\s*(?:\d{1,2}|[ivxIVX]{1,4}|[a-zA-Z])\s*(?:[\)\]]\s*|\.\s+)"
)
# "Figure 3", "Table 2", "Section 4.1", "Eq. 7", "S5" point at something; they
# do not claim a quantity, and their numbering routinely differs from the
# passage that is cited.
_REFERENCE_NUM_RE = re.compile(
    r"\b(?:figure|fig|table|tab|section|sec|equation|eq|appendix|app|supplementary|algorithm|alg|step|item|re)\s*\.?\s*[\dSivx]+(?:\.\d+)*",
    re.IGNORECASE,
)


def _numbers(text: str) -> set[str]:
    """Quantities the claim actually asserts.

    Enumeration markers and cross-references are stripped first: treating
    "(6) The FedAvg technique is used" as asserting the figure 6 produced a
    false "cites figures absent from the passage" rejection on a real answer.
    """
    text = _ENUM_PREFIX_RE.sub("", text)
    text = _REFERENCE_NUM_RE.sub(" ", text)
    out: set[str] = set()
    for raw in _NUM_RE.findall(text):
        token = raw.rstrip("%").lstrip("+")
        try:
            out.add(f"{float(token):g}")
        except ValueError:
            continue
    return out


_ANY_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _evidence_numbers(text: str) -> set[str]:
    """Every number the evidence contains, including ones embedded in an
    identifier. The claim side stays strict (only quantities it asserts);
    the evidence side must be inclusive, or "Qwen 3.6-35B" in an answer is
    rejected against a passage writing "Qwen3.6-35B" -- the digits are glued
    to letters there, so the strict pattern never sees 3.6 at all."""
    out = _numbers(text)
    for raw in _ANY_NUM_RE.findall(text):
        out.add(f"{float(raw):g}")
    return out


def _negations(text: str) -> set[str]:
    return {
        w.lower().replace("'", "")
        for w in _WORD_RE.findall(text)
        if w.lower().replace("'", "") in _NEGATION
    }


@dataclass
class LexicalJudgement:
    coverage: float
    numbers_ok: bool
    negation_ok: bool
    missing_numbers: set[str]

    @property
    def certain_fail(self) -> bool:
        """Failures no amount of paraphrase can excuse."""
        return not self.numbers_ok or not self.negation_ok


def judge_lexically(claim: str, evidence: str) -> LexicalJudgement:
    claim_words = _content_words(claim)
    evidence_words = _content_words(evidence)
    coverage = (
        len(claim_words & evidence_words) / len(claim_words) if claim_words else 0.0
    )

    claim_numbers = _numbers(claim)
    missing = claim_numbers - _evidence_numbers(evidence)

    claim_neg = bool(_negations(claim))
    evidence_neg = bool(_negations(evidence))
    # Only flag when the claim asserts a negation its evidence does not. The
    # reverse (evidence hedges, claim does not) is caught by coverage.
    negation_ok = not (claim_neg and not evidence_neg)

    return LexicalJudgement(
        coverage=coverage,
        numbers_ok=not missing,
        negation_ok=negation_ok,
        missing_numbers=missing,
    )


class LexicalVerifier:
    """Offline, deterministic, free. What CI runs."""

    name = "lexical"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.citation

    def _evidence_for(
        self, claim: Claim, rendered: RenderedContext
    ) -> tuple[str, list[str]]:
        chunks: list[RetrievedChunk] = cited_chunks(claim, rendered)
        if chunks:
            return "\n".join(c.text for c in chunks), [c.chunk_id for c in chunks]
        if self.cfg.require_citation_per_sentence:
            return "", []
        # Lenient mode: an uncited claim may still be grounded somewhere in
        # the retrieved set. That is a citation-quality problem, not a
        # hallucination, so it is scored rather than failed outright.
        return "\n".join(rc.text for rc in rendered.used), [
            rc.chunk_id for rc in rendered.used
        ]

    def verify(
        self, answer_text: str, rendered: RenderedContext, question: str
    ) -> tuple[list[ClaimVerdict], float]:
        claims = split_claims(answer_text)
        if not claims:
            return [], 0.0

        verdicts: list[ClaimVerdict] = []
        for claim in claims:
            evidence, chunk_ids = self._evidence_for(claim, rendered)
            if not evidence:
                verdicts.append(
                    ClaimVerdict(
                        claim=claim.text,
                        supported=False,
                        cited_chunk_ids=[],
                        support_score=0.0,
                        reason="no citation: nothing to verify this sentence against",
                    )
                )
                continue

            judged = judge_lexically(claim.text, evidence)
            if not judged.numbers_ok:
                reason = (
                    "cites figures absent from the passage: "
                    + ", ".join(sorted(judged.missing_numbers))
                )
                supported = False
            elif not judged.negation_ok:
                reason = "negates what the cited passage states"
                supported = False
            else:
                supported = judged.coverage >= self.cfg.lexical_threshold
                reason = (
                    f"{judged.coverage:.0%} of the claim's terms appear in the "
                    f"cited passage"
                    + ("" if supported else f" (below {self.cfg.lexical_threshold:.0%})")
                )

            verdicts.append(
                ClaimVerdict(
                    claim=claim.text,
                    supported=supported,
                    cited_chunk_ids=chunk_ids,
                    support_score=round(judged.coverage, 3),
                    reason=reason,
                )
            )

        ratio = sum(1 for v in verdicts if v.supported) / len(verdicts)
        return verdicts, ratio


class LLMVerifier:
    """Asks a model to judge entailment for every claim."""

    name = "llm"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.citation
        self.llm = get_llm(settings)
        self.prompt = load_prompt(
            "claim_check", settings.prompts.claim_check_version, str(settings.prompts.path)
        )

    def judge(
        self, claims: list[Claim], rendered: RenderedContext
    ) -> dict[int, tuple[bool, float, str]]:
        if not claims:
            return {}
        lines = [f"CLAIM {i}: {c.text}" for i, c in enumerate(claims, start=1)]
        system, user = self.prompt.render(
            context=rendered.text, claims="\n".join(lines)
        )
        response = self.llm.complete(
            LLMRequest(system=system, user=user, task="claim_check", temperature=0.0)
        )
        return self._parse(response.text, len(claims))

    @staticmethod
    def _parse(text: str, expected: int) -> dict[int, tuple[bool, float, str]]:
        """Tolerate a model that wraps JSON in prose or fences.

        A judge whose output cannot be parsed must not silently pass every
        claim -- that would turn enforcement off exactly when it matters.
        Unparseable verdicts are simply absent, and the caller keeps its
        lexical result for those claims.
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            log.warning("claim-check response contained no JSON object")
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            log.warning("claim-check JSON did not parse: %s", exc)
            return {}

        out: dict[int, tuple[bool, float, str]] = {}
        for i, verdict in enumerate(payload.get("verdicts", []), start=1):
            if not isinstance(verdict, dict):
                continue
            claim_id = int(verdict.get("claim_id", i))
            if not 1 <= claim_id <= expected:
                continue
            out[claim_id] = (
                bool(verdict.get("supported", False)),
                float(verdict.get("score", 0.0) or 0.0),
                str(verdict.get("reason", ""))[:300],
            )
        return out

    def verify(
        self, answer_text: str, rendered: RenderedContext, question: str
    ) -> tuple[list[ClaimVerdict], float]:
        claims = split_claims(answer_text)
        if not claims:
            return [], 0.0
        judged = self.judge(claims, rendered)
        verdicts: list[ClaimVerdict] = []
        for i, claim in enumerate(claims, start=1):
            supported, score, reason = judged.get(
                i, (False, 0.0, "judge returned no verdict for this claim")
            )
            verdicts.append(
                ClaimVerdict(
                    claim=claim.text,
                    supported=supported,
                    cited_chunk_ids=[rc.chunk_id for rc in cited_chunks(claim, rendered)],
                    support_score=round(score, 3),
                    reason=reason,
                )
            )
        ratio = sum(1 for v in verdicts if v.supported) / len(verdicts)
        return verdicts, ratio


class HybridVerifier:
    """Lexical first; the LLM judges only what lexical cannot settle.

    Claims that clearly pass coverage are accepted, and claims that fail the
    numeric or negation checks are rejected outright -- a model that talks
    itself into accepting "2.5 BLEU" against a passage saying "3.1 BLEU" is
    wrong, and deferring to it would weaken the guarantee. Everything in the
    uncertain middle is escalated, which is where an LLM actually adds
    something over word matching.
    """

    name = "hybrid"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.citation
        self.lexical = LexicalVerifier(settings)
        self._llm: LLMVerifier | None = None

    @property
    def llm(self) -> LLMVerifier:
        if self._llm is None:
            self._llm = LLMVerifier(self.settings)
        return self._llm

    def verify(
        self, answer_text: str, rendered: RenderedContext, question: str
    ) -> tuple[list[ClaimVerdict], float]:
        verdicts, _ = self.lexical.verify(answer_text, rendered, question)
        if not verdicts:
            return [], 0.0

        claims = split_claims(answer_text)
        threshold = self.cfg.lexical_threshold
        # Escalate only the ambiguous band: confidently-covered claims and
        # hard numeric/negation failures are already settled.
        uncertain = [
            i
            for i, v in enumerate(verdicts)
            if v.cited_chunk_ids
            and "figures absent" not in v.reason
            and "negates" not in v.reason
            and threshold * 0.5 <= v.support_score < min(1.0, threshold * 1.6)
        ]

        if uncertain:
            subset = [claims[i] for i in uncertain]
            try:
                judged = self.llm.judge(subset, rendered)
            except Exception as exc:  # noqa: BLE001 - provider failures vary
                log.warning("claim-check judge unavailable, keeping lexical: %s", exc)
                judged = {}
            for position, index in enumerate(uncertain, start=1):
                if position not in judged:
                    continue
                supported, score, reason = judged[position]
                verdicts[index].supported = supported
                verdicts[index].support_score = round(score, 3)
                verdicts[index].reason = f"judge: {reason}" if reason else "judge verdict"

        ratio = sum(1 for v in verdicts if v.supported) / len(verdicts)
        return verdicts, ratio


def get_verifier(settings: Settings):
    if not settings.citation.enforce:
        return None
    kind = settings.citation.verifier
    if kind == "lexical":
        return LexicalVerifier(settings)
    if kind == "llm":
        return LLMVerifier(settings)
    if kind == "hybrid":
        return HybridVerifier(settings)
    raise ValueError(f"unknown citation verifier: {kind}")

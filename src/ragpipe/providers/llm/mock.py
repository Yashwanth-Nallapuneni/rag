"""Deterministic offline LLM.

This is not a toy stub. It is an *extractive* model: it selects the best
supporting passage from the provided context by lexical overlap and answers
with sentences copied from it, cited with the matching [n] marker. When no
passage overlaps the question enough, it refuses.

That behaviour matters, because it means the citation-enforcement layer, the
refusal path, the API and the eval harness can all be exercised end to end --
and in CI -- with no API key, no cost and no run-to-run variance. Swapping in
a real provider changes one config line.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ..base import LLMRequest, LLMResponse

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "that",
    "this", "these", "those", "it", "its", "and", "or", "but", "if", "then",
    "than", "so", "such", "what", "which", "who", "whom", "how", "why",
    "when", "where", "does", "do", "did", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must", "have", "has", "had", "about",
}

# Matches a context block header like: [S2] (Attention Is All You Need | p. 4)
# The S prefix keeps these distinct from a paper's own "[4, 15]" references.
_BLOCK_RE = re.compile(r"^\s*\[S(\d+)\]\s*(?:\(([^)]*)\))?\s*$", re.MULTILINE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
# Anything from one of these headings onward is instruction, not context.
_TAIL_RE = re.compile(
    r"\n\s*(?:Question|QUESTION|Answer|ANSWER|CLAIM|Claim|Claims|CLAIMS"
    r"|Task|TASK|Instructions|INSTRUCTIONS)\b[^\n]*:",
)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-_.]*", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text.strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _parse_context_blocks(user_prompt: str) -> dict[int, str]:
    """Pull `[n] (locator)\\n body` blocks out of a rendered prompt."""
    matches = list(_BLOCK_RE.finditer(user_prompt))
    blocks: dict[int, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(user_prompt)
        body = user_prompt[start:end].strip()
        # Trim the instruction tail that follows the context blocks. Without
        # this the claim-checker reads the claims themselves as context and
        # marks every claim supported.
        body = _TAIL_RE.split(body)[0]
        if body:
            blocks[int(m.group(1))] = body.strip()
    return blocks


def _extract_question(user_prompt: str) -> str:
    m = re.findall(r"(?:Question|QUESTION)\s*:\s*(.+)", user_prompt)
    if m:
        return m[-1].strip()
    return user_prompt.strip().splitlines()[-1] if user_prompt.strip() else ""


class MockLLM:
    """Offline extractive provider satisfying the LLMProvider protocol."""

    name = "mock"

    def __init__(
        self,
        model: str = "mock-extractive-v1",
        min_overlap: float = 0.12,
        max_sentences: int = 3,
    ):
        self.model = model
        self.min_overlap = min_overlap
        self.max_sentences = max_sentences

    def complete(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        if request.task == "claim_check":
            text = self._claim_check(request)
        else:
            text = self._answer(request)
        elapsed = (time.perf_counter() - started) * 1000
        return LLMResponse(
            text=text,
            model=self.model,
            usage={
                "input_tokens": len(request.user.split()) + len(request.system.split()),
                "output_tokens": len(text.split()),
            },
            latency_ms=elapsed,
            finish_reason="stop",
        )

    # -- answer ---------------------------------------------------------
    def _answer(self, request: LLMRequest) -> str:
        blocks = _parse_context_blocks(request.user)
        question = _extract_question(request.user)
        q_tokens = _tokens(question)
        if not blocks or not q_tokens:
            return "INSUFFICIENT_CONTEXT"

        scored: list[tuple[float, int, str]] = []
        for marker, body in blocks.items():
            for sentence in _split_sentences(body):
                s_tokens = _tokens(sentence)
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens) / len(q_tokens)
                # Mild length normalisation so a huge sentence cannot win by
                # sheer surface area.
                density = len(q_tokens & s_tokens) / (len(s_tokens) ** 0.5)
                scored.append((overlap + 0.15 * density, marker, sentence))

        if not scored:
            return "INSUFFICIENT_CONTEXT"
        scored.sort(key=lambda t: (-t[0], t[1]))
        if scored[0][0] < self.min_overlap:
            return "INSUFFICIENT_CONTEXT"

        # Only keep sentences that clear the bar themselves -- otherwise the
        # answer pads itself with unrelated passages and cites them.
        cutoff = max(self.min_overlap, scored[0][0] * 0.5)
        chosen: list[tuple[int, str]] = []
        seen: set[str] = set()
        for _score, marker, sentence in scored[: self.max_sentences * 3]:
            if _score < cutoff:
                break
            key = sentence[:80]
            if key in seen:
                continue
            seen.add(key)
            chosen.append((marker, sentence))
            if len(chosen) >= self.max_sentences:
                break

        return " ".join(
            f"{s.rstrip()}{'' if s.rstrip().endswith(('.', '!', '?')) else '.'} [S{m}]"
            for m, s in chosen
        )

    # -- claim check ----------------------------------------------------
    def _claim_check(self, request: LLMRequest) -> str:
        """Judge each claim by lexical containment against the given context."""
        blocks = _parse_context_blocks(request.user)
        context_tokens = _tokens(" ".join(blocks.values())) if blocks else set()
        claims = re.findall(r"(?:CLAIM|Claim)\s*\d*\s*:\s*(.+)", request.user)
        verdicts = []
        for claim in claims:
            c_tokens = _tokens(claim)
            ratio = (
                len(c_tokens & context_tokens) / len(c_tokens) if c_tokens else 0.0
            )
            verdicts.append(
                {
                    "claim": claim.strip(),
                    "supported": bool(ratio >= 0.6),
                    "score": round(ratio, 3),
                    "reason": (
                        "lexically grounded in context"
                        if ratio >= 0.6
                        else "insufficient overlap with retrieved context"
                    ),
                }
            )
        return json.dumps({"verdicts": verdicts}, ensure_ascii=False)

    def health(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "ready": True, "offline": True}

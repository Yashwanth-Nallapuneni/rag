"""Parsing and validating the citation markers in a generated answer.

Three failure modes matter here and all are silent if unchecked:

1. A model cites `[S7]` when only five passages were supplied. The marker
   resolves to nothing, and an answer that looks cited is in fact ungrounded.
2. A sentence carries no marker at all, so there is nothing to verify it
   against.
3. A quoted passage carries the source paper's OWN reference markers --
   "[4, 15, 20, 26]" -- which must never be mistaken for citations to our
   passages. This is why markers are `[S<n>]`; see `context.py`.

All three are detected here and surfaced on the Answer rather than swallowed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas import Citation, RetrievedChunk
from .context import RenderedContext, citation_for

# Only our own [S<n>] markers. A bare [12] in quoted text is the source
# paper's bibliography reference and is deliberately ignored.
_MARKER_RE = re.compile(r"\[S(\d{1,3})\]")

# Match a sentence together with the markers that TRAIL it. Splitting on
# sentence boundaries alone attributes "...recurrence. [1] Next..." to the
# wrong sentence -- the [1] lands at the head of the next claim, so every
# claim is then verified against the passage belonging to its neighbour.
_ABBREV = r"(?<!\be\.g)(?<!\bi\.e)(?<!\bet\sal)(?<!\bcf)(?<!\bvs)(?<!\bFig)(?<!\bEq)(?<!\bSec)"
# A sentence terminator only counts when whitespace or end-of-text follows it.
# Without `(?=\s|$)` every dot inside an email address or dotted identifier is
# a sentence break: an author block like "Chenxi.Wu25 ... @student.xjtlu.edu.cn"
# shattered into ten bogus claims ("edu.", "xjtlu.", "Wang19}@student."), none
# of them carrying the citation marker, which dropped a perfectly good answer's
# support ratio to 23% and refused it.
#
# The lookarounds around the terminator refuse to break inside an ellipsis:
# quoted maths such as "(y1, . . . , yT)" would otherwise fragment one sentence
# into several, most of them a bare ".".
_CLAIM_RE = re.compile(
    rf"(.+?{_ABBREV}(?<!\.\s)[.!?](?!\s*[.!?])[\"')\]]?)((?:\s*\[S\d{{1,3}}\])*)(?=\s|$)",
    re.DOTALL,
)

# A claim must assert something. Fragments with no real word ("." or ", yT),
# (1).") are continuations of the previous sentence, not claims of their own.
_HAS_WORD_RE = re.compile(r"[A-Za-z]{2,}")


@dataclass
class Claim:
    """One sentence of an answer plus the markers it cites."""

    text: str
    markers: list[int]

    @property
    def uncited(self) -> bool:
        return not self.markers


def extract_markers(text: str) -> list[int]:
    """Markers in order of first appearance, de-duplicated."""
    seen: dict[int, None] = {}
    for m in _MARKER_RE.finditer(text):
        seen.setdefault(int(m.group(1)), None)
    return list(seen)


def strip_markers(text: str) -> str:
    """Remove our citation markers, leaving the source's own references
    intact -- they are part of the quoted sentence's meaning."""
    return re.sub(r"\s*\[S\d{1,3}\]", "", text).strip()


def split_claims(text: str) -> list[Claim]:
    """Split an answer into sentence-level claims with their markers.

    Sentence granularity is deliberate: it is the unit a reader would check,
    and the unit the faithfulness verifier can actually evaluate against a
    passage. A whole-answer verdict hides one unsupported sentence inside four
    good ones.
    """
    body = text.strip()
    if not body:
        return []

    claims: list[Claim] = []

    def _add(raw: str) -> None:
        prose = strip_markers(raw)
        markers = [int(m.group(1)) for m in _MARKER_RE.finditer(raw)]
        if prose and _HAS_WORD_RE.search(prose):
            claims.append(Claim(text=prose, markers=markers))
        elif claims:
            # Punctuation-only or symbol-only tail: fold it back into the
            # sentence it belongs to so it is not judged as its own claim.
            if prose:
                claims[-1].text = f"{claims[-1].text} {prose}".strip()
            claims[-1].markers.extend(markers)

    end = 0
    for match in _CLAIM_RE.finditer(body):
        _add(match.group(1) + match.group(2))
        end = match.end()

    # A trailing fragment with no terminal punctuation is still a claim.
    remainder = body[end:].strip()
    if remainder:
        _add(remainder)

    for claim in claims:
        claim.markers = sorted(set(claim.markers))
    return claims


def resolve_citations(
    answer_text: str, rendered: RenderedContext
) -> tuple[list[Citation], list[int]]:
    """Map markers to chunks. Returns (citations, markers that resolve to
    nothing)."""
    mapping = rendered.marker_to_chunk
    citations: list[Citation] = []
    unknown: list[int] = []
    for marker in extract_markers(answer_text):
        rc = mapping.get(marker)
        if rc is None:
            unknown.append(marker)
            continue
        citations.append(citation_for(marker, rc))
    return citations, unknown


def cited_chunks(
    claim: Claim, rendered: RenderedContext
) -> list[RetrievedChunk]:
    mapping = rendered.marker_to_chunk
    return [mapping[m] for m in claim.markers if m in mapping]

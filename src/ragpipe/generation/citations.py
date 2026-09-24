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

# Real models do not reliably emit ASCII brackets. Groq's gpt-oss-120b cited
# every passage as 【S1】 with CJK fullwidth brackets (U+3010/U+3011), which our
# ASCII-only pattern could not see -- so every sentence counted as uncited,
# support came out at 0%, and the pipeline refused three out of three
# answerable questions while the answers were in fact correct and properly
# attributed. Normalisation is deliberately narrow: only a bracket pair
# wrapping an S-marker is rewritten, so a fullwidth bracket appearing in
# quoted source text is left alone.
# On OpenRouter, the same model also cites as \u3010S1\u2020L1-L4\u3011 -- a ChatGPT-style
# "browsing" annotation with a dagger and a line range glued onto the marker.
# The original pattern required the closing bracket immediately after the
# digits, so this shape matched neither the ASCII nor the fullwidth regex,
# every sentence was uncited, and support came out at 0% -- same failure
# mode as the CJK-bracket bug, different suffix. The middle group is now
# "anything that is not a closing bracket", so both a bare \u3010S1\u3011 and a
# \u3010S1\u2020L1-L4\u3011 collapse to [S1].
_FULLWIDTH_MARKER_RE = re.compile(
    r"[\u3010\uff3b\u3014\ufe5d\u2045]\s*[Ss]\s*(\d{1,3})[^\u3011\uff3d\u3015\ufe5e\u2046]*[\u3011\uff3d\u3015\ufe5e\u2046]"
)
# Signals that a model tried to cite but in a shape we do not accept. Logged
# rather than silently swallowed, because the failure mode is a 100% refusal
# rate that looks like a retrieval problem.
_SUSPECT_MARKER_RE = re.compile(
    r"[\u3010\uff3b\u3014]\s*\d{1,3}\s*[\u3011\uff3d\u3015]|\(\s*[Ss]\d{1,3}\s*\)"
)


# Typographic characters gpt-oss-120b emits and every parser downstream
# assumes are ASCII. Each broke a real answer in the 40-pair pilot:
#   U+200B inside a marker, "[\u200bS2]"    -> citation invisible, 0% support
#   U+201D before a marker, '.\u201d [S1]'  -> two claims merged into one
#   U+2011 / U+202F in "Qwen\u202f3.6\u20113" -> numbers split oddly
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_TYPOGRAPHY = str.maketrans({
    **{c: " " for c in "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f"},
    **{c: "-" for c in "\u2010\u2011\u2012\u2013\u2212"},
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
})


def normalize_typography(text: str) -> str:
    """Map invisible and typographic characters to their ASCII equivalents.
    Deliberately leaves the em dash (U+2014) alone: it is punctuation, not a
    hyphen, and turning it into '-' would glue words together."""
    return _ZERO_WIDTH_RE.sub("", text).translate(_TYPOGRAPHY)


def normalize_citation_markers(text: str) -> str:
    """Normalise typography, then rewrite fullwidth/CJK bracket citation
    markers to ASCII [S<n>]."""
    return _FULLWIDTH_MARKER_RE.sub(lambda m: f"[S{m.group(1)}]", normalize_typography(text))


def has_suspect_markers(text: str) -> bool:
    """True when the text looks like it cites in an unsupported shape."""
    return bool(_SUSPECT_MARKER_RE.search(text))

# Match a sentence together with the markers that TRAIL it. Splitting on
# sentence boundaries alone attributes "...recurrence. [1] Next..." to the
# wrong sentence -- the [1] lands at the head of the next claim, so every
# claim is then verified against the passage belonging to its neighbour.
#
# "U.S." was a real false split: gpt-oss-120b answered "...from TikTok,
# Twitter/X, and Truth Social during the 2024 U.S. presidential election
# [S1][S2]" with no other terminal punctuation. The period in "U.S." is
# followed by whitespace, so it satisfied the terminator lookahead and the
# claim broke there -- the half with the real content lost its markers, the
# trailing half ("presidential election") kept them and passed trivially,
# and the true support ratio (one grounded claim) was reported as 50%.
# `(?<!\b[A-Za-z]\.[A-Za-z])` generalises the fix beyond one hardcoded
# abbreviation: it refuses to break after the second letter of any
# single-letter-dot-single-letter initialism ("U.S.", "U.K.", "e.g" already
# covered explicitly below for clarity/backcompat).
_ABBREV = (
    r"(?<!\be\.g)(?<!\bi\.e)(?<!\bet\sal)(?<!\bcf)(?<!\bvs)(?<!\bFig)(?<!\bEq)(?<!\bSec)"
    r"(?<!\b[A-Za-z]\.[A-Za-z])"
)
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
    text = normalize_citation_markers(text)
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
    body = normalize_citation_markers(text).strip()
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
    answer_text = normalize_citation_markers(answer_text)
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

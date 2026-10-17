"""Document-attribution check (D15). No LLM call, pure string logic.

Failure class: the question names one paper ("... does <Paper X> use for its
released code?") and the model answers from, and cites, a passage of a
*different* paper. The passage is faithful to its own paper, relevant to the
topic, and lexically close to the claim, so the relevance gate and the
faithfulness verifier both pass it. Only the pairing of question and cited
document is wrong, and that is what this module checks.

When does a question "name" a document? Deliberately conservative -- a false
trigger refuses an answerable question, which costs more than a missed
detection (the status quo). Text is normalised first (case, accents, dashes,
quotes and punctuation all collapse to single spaces, so "Paint-Anything"
equals "paint anything"). A document is named when the question contains, on
word boundaries:

  1. its full normalised title; or
  2. the title's main part, i.e. the text before the first colon (the usual
     "Short-Name: Long descriptive subtitle" shape), but only if that part is
     distinctive enough to be a name rather than a topic:
       - it has >= 2 tokens (a coined name or a multi-word phrase), or
       - it is a single token of >= 6 characters that occurs as a token in
         exactly one corpus title (so "Transformers" or "Survey" in many
         titles never counts).

Only if EXACTLY ONE distinct document is named does the check apply. If two
or more are named (a comparison question) or none (a general question), the
answer is left alone -- cross-paper questions legitimately cite several
documents.

Known limit: a 2-token main part that is also an everyday phrase would match
any question containing it. The "exactly one document" rule bounds the damage
and the offline replay (scripts/replay_attribution.py) measures it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..schemas import RetrievedChunk

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_MIN_SINGLE_TOKEN_LEN = 6


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def _contains_phrase(haystack: str, phrase: str) -> bool:
    """Whole-token containment on already-normalised strings."""
    return bool(phrase) and f" {phrase} " in f" {haystack} "


@dataclass(frozen=True)
class DocumentIndex:
    """doc_id -> (original title, full normalised title, usable main-part
    name or ''), built once from the store's chunks."""

    titles: dict[str, str]
    _full: dict[str, str]
    _names: dict[str, str]

    @classmethod
    def from_titles(cls, titles: dict[str, str]) -> DocumentIndex:
        full = {d: normalize(t) for d, t in titles.items()}
        token_doc_freq: dict[str, int] = {}
        for t in set(full.values()):
            for tok in set(t.split()):
                token_doc_freq[tok] = token_doc_freq.get(tok, 0) + 1
        names: dict[str, str] = {}
        for d, title in titles.items():
            if ":" not in title:
                continue
            head = normalize(title.split(":", 1)[0])
            toks = head.split()
            if len(toks) >= 2 or (
                len(toks) == 1
                and len(head) >= _MIN_SINGLE_TOKEN_LEN
                and token_doc_freq.get(head, 0) == 1
            ):
                names[d] = head
        return cls(titles=dict(titles), _full=full, _names=names)

    @classmethod
    def from_chunks(cls, chunks) -> DocumentIndex:
        titles: dict[str, str] = {}
        for c in chunks:
            titles.setdefault(c.doc_id, c.doc_title)
        return cls.from_titles(titles)

    def named_documents(self, question: str) -> list[str]:
        q = normalize(question)
        hits = []
        for d, full in self._full.items():
            if _contains_phrase(q, full) or _contains_phrase(q, self._names.get(d, "")):
                hits.append(d)
        return hits


def check_attribution(
    question: str,
    cited: list[RetrievedChunk],
    index: DocumentIndex,
) -> str | None:
    """Return a refusal reason if the question names exactly one document and
    any cited chunk comes from another; otherwise None."""
    if not cited:
        return None
    named = index.named_documents(question)
    if len(named) != 1:
        return None
    target = named[0]
    wrong: list[str] = []
    for rc in cited:
        if rc.chunk.doc_id != target and rc.chunk.doc_title not in wrong:
            wrong.append(rc.chunk.doc_title)
    if not wrong:
        return None
    return (
        f'the question asks about "{index.titles[target]}" but the answer cites '
        f"a different document: " + "; ".join(f'"{t}"' for t in wrong)
    )

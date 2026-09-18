"""Tokenizer tuned for BM25 over this corpus (arXiv-ish ML papers).

BM25's whole job in this pipeline is to catch exact-term and identifier
matches that the dense embedder blurs together (model names, arXiv ids,
metric acronyms, dataset names). That means the tokenizer is the single
highest-leverage decision here: get it wrong and BM25 degenerates into a
worse dense retriever.

Design choices
--------------
1. Lowercase, then split on whitespace/punctuation, but treat `-`, `.`,
   `_` as *token-internal* when they sit between alphanumerics. This is
   what keeps `bge-small-en-v1.5`, `cross-encoder`, `GPT-4o`, `cs.CL` and
   `1706.03762` intact as single tokens instead of being shredded into
   `bge small en v1 5` -- which would throw away exactly the precision
   BM25 exists to provide over the dense retriever.

2. Stopwords are removed with a small built-in English list (no nltk
   dependency, per spec). Stopwords are near-universally high document
   frequency, so BM25's IDF term already discounts them heavily -- but
   removing them outright still saves index size and avoids a stray
   stopword in a query diluting the term overlap signal.

3. No stemming/lemmatisation. Dense retrieval already generalises across
   morphological variants (`retrieve`/`retrieval`/`retrieving`) via
   embeddings; that is precisely what it is good at. BM25's reason to
   exist in a hybrid system is the opposite job: reward a query for
   matching the *exact* surface form of an identifier, model name or
   acronym. Stemming `BM25` or `bge-small-en-v1.5` down to a crude root
   would blur that distinction and duplicate what dense already covers,
   while making exact-match false-positives (stemmed collisions between
   unrelated words) more likely. So we keep surface forms as-is.

4. Compound sub-tokens: a hyphenated/dotted compound like `cross-encoder`
   or `bge-small-en-v1.5` is emitted *both* as the full compound token and
   as its alphanumeric sub-parts (`cross`, `encoder` / `bge`, `small`,
   `en`, `v1`, `5`). Tradeoff: this inflates document length and adds a
   little noise (a query for "small" now also matches inside
   `bge-small-en-v1.5`), but without it a query for the generic word
   "encoder" would never retrieve a chunk that only ever writes
   "cross-encoder" -- a strictly worse failure mode for a retriever whose
   job is recall of technical text. The full compound is kept alongside
   the sub-tokens (not replaced), so an exact query for `cross-encoder`
   still gets the stronger, more specific match via repeated term
   frequency on that exact token.
"""

from __future__ import annotations

import re

# Small, dependency-free English stopword list. Deliberately conservative:
# it strips function words but keeps anything that could plausibly be part
# of a technical phrase (e.g. "not", "no" are kept out on purpose since
# negation can matter in claims/QA text).
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the of in on at to for with by from as that this these those it
    its and or but if then than so such what which who whom how why when
    where does do did can could should would will shall may might must
    have has had having be been being was were are is am i we you he she
    they them his her their our your my me us not no nor also into over
    under between out up down off again further once here there all any
    both each few more most other some only own same too very s t can
    will just don now
    """.split()
)

# A token is a run of alphanumerics, optionally chained with -, ., or _
# to the next alphanumeric run: "bge-small-en-v1.5", "1706.03762",
# "cs.CL", "GPT-4o", "model_name". The lookahead requires the connector be
# followed by another alphanumeric so trailing punctuation ("BM25." at a
# sentence end) never gets swallowed into the token.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[\-._][a-z0-9]+)*")

# Splits a compound into its alphanumeric sub-parts, for the extra
# sub-tokens described in choice 4 above.
_SUBPART_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, extract identifier-preserving tokens, drop stopwords,
    and add sub-tokens for compounds. No stemming -- see module docstring."""
    if not text:
        return []
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        out.append(tok)
        if "-" in tok or "." in tok or "_" in tok:
            parts = _SUBPART_RE.findall(tok)
            if len(parts) > 1:
                out.extend(p for p in parts if p not in STOPWORDS and len(p) > 1)
    return out

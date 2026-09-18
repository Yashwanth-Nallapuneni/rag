"""Known-item retrieval diagnostic for tuning fusion and reranking.

WHAT THIS IS: a query is synthesised from a known target chunk, and we measure
how often each retrieval configuration puts that chunk back on top. It is
cheap, needs no labels, and is sensitive enough to tune fusion weights and to
prove the reranker is reordering in the right direction.

WHAT THIS IS NOT: an answer-quality benchmark, and not a defensible
"precision improved by X%" claim. Queries derived from a chunk's own text
share vocabulary with it, which structurally favours lexical matching -- so
BM25 and hybrid will look better here than they would on natural questions.
The three query families below make that bias explicit rather than hiding it
behind one aggregate number. The honest precision/faithfulness numbers come
from the human-verified golden dataset in the evaluation phase.

Query families, weakest bias last:
  verbatim_sentence -- a sentence copied from the target chunk (pure lexical)
  term_bag          -- its salient terms, shuffled (lexical, no phrase match)
  identifier        -- a real identifier from it (the exact-term case)
  abstract_to_body  -- a sentence from the paper's ABSTRACT, scored against
                       that paper's BODY passages with the abstract excluded.
                       No verbatim shortcut exists here.

MEASURED FINDING, recorded so nobody re-derives it: abstract_to_body does not
discriminate either, and BM25 scores a perfect 1.000 on it. Accepting any of a
paper's 10-25 body chunks makes the task document identification, and each
paper's distinctive jargon ("FL-Net", "dQwen3.5", a bespoke metric name) makes
that trivial for term matching. Discriminating between dense and sparse needs
natural questions labelled against the SPECIFIC passage that answers them --
which is the human-verified golden dataset, not a synthesised query.

So do NOT tune fusion weights on these numbers. What they do support:
  * BM25 is far stronger on exact identifiers (5x dense R@1), which is the
    documented reason for including it at all.
  * Reranking substantially improves a weak first pass and does nothing for
    an already-saturated one.
  * Fusion is never worse than its weaker component.

Reported per family and per configuration:
  recall@1/@5/@10 -- was the target chunk in the top n
  MRR@10          -- 1/rank of the target, averaged
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Settings
from ..schemas import Chunk, RetrievedChunk
from ..tokenization import count_tokens

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_.]{2,}")
# Something that looks like a technical identifier rather than prose: an
# internal digit, hyphen or dot, or an internal capital (CamelCase).
_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Za-z]+[-_.][A-Za-z0-9][A-Za-z0-9\-_.]*|[A-Za-z]+\d[A-Za-z0-9]*|[a-z]+[A-Z][A-Za-z]*)\b"
)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "which", "their", "these", "those", "have", "has", "had", "been", "being",
    "our", "its", "into", "than", "then", "also", "can", "may", "such", "each",
    "when", "where", "while", "does", "using", "used", "use", "both", "more",
    "most", "other", "over", "under", "between", "because", "however", "thus",
    "shown", "show", "results", "paper", "work", "method", "based", "given",
}


@dataclass
class BenchQuery:
    """A query plus the chunks that count as a correct hit.

    `accept_chunk_ids` is a set rather than a single id because the
    abstract_to_body family accepts any body passage of the right paper, and
    `exclude_chunk_ids` removes the chunks that would make it a trivial
    lexical hit (the abstract the query was lifted from).
    """

    query: str
    target_doc_id: str
    accept_chunk_ids: set[str]
    family: str
    exclude_chunk_ids: set[str] = field(default_factory=set)


@dataclass
class FamilyResult:
    family: str
    n: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    doc_recall_at_5: float
    mean_latency_ms: float

    def as_row(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "n": self.n,
            "recall@1": round(self.recall_at_1, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr@10": round(self.mrr_at_10, 4),
            "doc_recall@5": round(self.doc_recall_at_5, 4),
            "latency_ms": round(self.mean_latency_ms, 1),
        }


@dataclass
class ConfigResult:
    label: str
    config: dict[str, Any]
    families: list[FamilyResult] = field(default_factory=list)
    overall: FamilyResult | None = None


# --- query synthesis ------------------------------------------------------


def _salient_terms(text: str, limit: int) -> list[str]:
    """Longer, rarer-looking words first; they carry the most signal."""
    seen: dict[str, None] = {}
    for word in _WORD_RE.findall(text):
        lowered = word.lower()
        if lowered in _STOP or len(lowered) < 4:
            continue
        seen.setdefault(word, None)
    ranked = sorted(seen, key=lambda w: (-len(w), w))
    return ranked[:limit]


def verbatim_query(chunk: Chunk, rng: random.Random) -> str | None:
    """A sentence lifted from the chunk. Maximum lexical overlap: this is the
    exact-term case BM25 exists to handle."""
    sentences = [s.strip() for s in _SENT_RE.split(chunk.text) if 40 <= len(s.strip()) <= 300]
    return rng.choice(sentences) if sentences else None


def term_bag_query(chunk: Chunk, rng: random.Random) -> str | None:
    """Salient terms, shuffled. Keeps vocabulary overlap but destroys phrase
    order, so a retriever cannot win on exact phrase match alone."""
    terms = _salient_terms(chunk.text, 12)
    if len(terms) < 5:
        return None
    picked = rng.sample(terms, min(7, len(terms)))
    rng.shuffle(picked)
    return " ".join(picked)


def identifier_query(chunk: Chunk, rng: random.Random) -> str | None:
    """A real identifier from the chunk plus light natural framing -- the
    'user searches for an exact spec name' case from the spec."""
    candidates = [
        c for c in dict.fromkeys(_IDENTIFIER_RE.findall(chunk.text)) if len(c) >= 5
    ]
    if not candidates:
        return None
    ident = rng.choice(candidates[:10])
    return f"what does the paper say about {ident}"


QUERY_FAMILIES: dict[str, Callable[[Chunk, random.Random], str | None]] = {
    "verbatim_sentence": verbatim_query,
    "term_bag": term_bag_query,
    "identifier": identifier_query,
}


def _is_abstract(chunk: Chunk) -> bool:
    top = (chunk.section_path[0] if chunk.section_path else "").lower()
    return "abstract" in top


def build_abstract_queries(
    chunks: list[Chunk], limit: int = 50, seed: int = 20260917
) -> list[BenchQuery]:
    """The one family here with no lexical shortcut.

    The query is a sentence from a paper's abstract; a hit is any BODY
    passage of that same paper, with the abstract chunks themselves excluded
    so a verbatim match cannot score. Abstracts state claims in their own
    compressed wording, so finding the body passage behind one is a genuinely
    semantic task -- which is why dense retrieval should be competitive here
    even though it loses badly on the lexical families.
    """
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    rng = random.Random(seed + 7)
    queries: list[BenchQuery] = []
    for doc_id, doc_chunks in sorted(by_doc.items()):
        abstracts = [c for c in doc_chunks if _is_abstract(c)]
        body = [c for c in doc_chunks if not _is_abstract(c)]
        if not abstracts or len(body) < 3:
            continue
        sentences = [
            s.strip()
            for s in _SENT_RE.split(abstracts[0].text)
            if 60 <= len(s.strip()) <= 300
        ]
        if not sentences:
            continue
        queries.append(
            BenchQuery(
                query=rng.choice(sentences),
                target_doc_id=doc_id,
                accept_chunk_ids={c.chunk_id for c in body},
                family="abstract_to_body",
                exclude_chunk_ids={c.chunk_id for c in abstracts},
            )
        )
        if len(queries) >= limit:
            break
    return queries


def build_queries(
    chunks: list[Chunk],
    per_family: int = 60,
    seed: int = 20260917,
    min_tokens: int = 120,
) -> list[BenchQuery]:
    """Sample target chunks and synthesise one query per family from each.

    Short chunks are skipped: there is not enough text to build a query that
    is distinguishable from the chunk itself.
    """
    rng = random.Random(seed)
    pool = [c for c in chunks if count_tokens(c.text) >= min_tokens]
    rng.shuffle(pool)

    queries: list[BenchQuery] = []
    for family, builder in QUERY_FAMILIES.items():
        made = 0
        family_rng = random.Random(seed + hash(family) % 10_000)
        for chunk in pool:
            if made >= per_family:
                break
            query = builder(chunk, family_rng)
            if not query or len(query.split()) < 3:
                continue
            queries.append(
                BenchQuery(
                    query=query,
                    target_doc_id=chunk.doc_id,
                    accept_chunk_ids={chunk.chunk_id},
                    family=family,
                )
            )
            made += 1
    return queries


# --- scoring --------------------------------------------------------------


def _rank_of(results: list[RetrievedChunk], bq: BenchQuery) -> int | None:
    """Rank of the first acceptable hit, ignoring excluded chunks entirely
    (they neither count as hits nor consume a rank)."""
    rank = 0
    for rc in results:
        if rc.chunk_id in bq.exclude_chunk_ids:
            continue
        rank += 1
        if rc.chunk_id in bq.accept_chunk_ids:
            return rank
    return None


def score_family(
    family: str, records: list[tuple[int | None, bool, float]]
) -> FamilyResult:
    """records: (rank of target or None, doc hit within top 5, latency ms)."""
    n = len(records) or 1
    ranks = [r for r, _, _ in records]
    return FamilyResult(
        family=family,
        n=len(records),
        recall_at_1=sum(1 for r in ranks if r == 1) / n,
        recall_at_5=sum(1 for r in ranks if r and r <= 5) / n,
        recall_at_10=sum(1 for r in ranks if r and r <= 10) / n,
        mrr_at_10=sum(1.0 / r for r in ranks if r and r <= 10) / n,
        doc_recall_at_5=sum(1 for _, d, _ in records if d) / n,
        mean_latency_ms=sum(lat for _, _, lat in records) / n,
    )


def run_config(
    label: str,
    settings: Settings,
    retriever: Any,
    queries: list[BenchQuery],
    k: int = 10,
) -> ConfigResult:
    import time

    per_family: dict[str, list[tuple[int | None, bool, float]]] = {}
    for bq in queries:
        started = time.perf_counter()
        results = retriever.retrieve(bq.query, k=k)
        latency = (time.perf_counter() - started) * 1000
        rank = _rank_of(results, bq)
        eligible = [rc for rc in results if rc.chunk_id not in bq.exclude_chunk_ids]
        doc_hit = any(rc.chunk.doc_id == bq.target_doc_id for rc in eligible[:5])
        per_family.setdefault(bq.family, []).append((rank, doc_hit, latency))

    families = [score_family(f, recs) for f, recs in sorted(per_family.items())]
    pooled = [rec for recs in per_family.values() for rec in recs]
    return ConfigResult(
        label=label,
        config={
            "mode": settings.retrieval.mode,
            "fusion": settings.retrieval.fusion,
            "dense_weight": settings.retrieval.dense_weight,
            "sparse_weight": settings.retrieval.sparse_weight,
            "candidate_k": settings.retrieval.candidate_k,
            "rerank": settings.rerank.model if settings.rerank.enabled else None,
        },
        families=families,
        overall=score_family("ALL", pooled),
    )


def format_table(results: list[ConfigResult]) -> str:
    """A readable before/after table -- this is the artifact worth reading."""
    lines: list[str] = []
    header = f"{'configuration':<28} {'family':<19} {'R@1':>7} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'docR@5':>7} {'ms':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for res in results:
        for fam in res.families:
            row = fam.as_row()
            lines.append(
                f"{res.label:<28} {row['family']:<19} {row['recall@1']:>7.3f} "
                f"{row['recall@5']:>7.3f} {row['recall@10']:>7.3f} {row['mrr@10']:>7.3f} "
                f"{row['doc_recall@5']:>7.3f} {row['latency_ms']:>8.1f}"
            )
        if res.overall:
            row = res.overall.as_row()
            lines.append(
                f"{res.label:<28} {'ALL':<19} {row['recall@1']:>7.3f} "
                f"{row['recall@5']:>7.3f} {row['recall@10']:>7.3f} {row['mrr@10']:>7.3f} "
                f"{row['doc_recall@5']:>7.3f} {row['latency_ms']:>8.1f}"
            )
        lines.append("")
    return "\n".join(lines)

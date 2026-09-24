"""LLM pre-screen for the golden dataset review queue.

This is advisory ONLY. It writes a sidecar file (`eval/golden_prescreen.jsonl`
by default) that the review UI (`scripts/review_golden.py`) reads to show the
reviewer a flag and sort suspect pairs first. It never writes to the review
ledger (`ragpipe.eval.golden.record_decision` / `save_review_ledger`) and
never marks a pair verified. `load_verified()` is unaffected by anything in
this module -- that is a hard requirement, not a convenience.

Two families of checks:

  * ANSWERABLE pairs -- is the ground truth actually supported by the source
    chunk(s)? Is the question standalone (names the paper/method, not "the
    passage")? Is the ground truth complete? Plus deterministic near-duplicate
    and empty/short-answer checks.
  * UNANSWERABLE pairs -- off-domain probes are just confirmed off-topic
    (no LLM call needed). In-domain "uncovered detail" probes are checked
    against a BM25 search restricted to that paper's own chunks: if a chunk
    plausibly answers the question, that is flagged "suspect" with the
    offending chunk id and a quote, because the probe may not actually be
    unanswerable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rank_bm25 import BM25Okapi

from ..providers import LLMRequest, ProviderError
from ..providers.base import LLMProvider
from ..retrieval.tokenize import tokenize
from ..schemas import Chunk, QAPair

Verdict = Literal["ok", "suspect"]


# --------------------------------------------------------------------------
# sidecar record + I/O
# --------------------------------------------------------------------------


@dataclass
class PrescreenRecord:
    id: str
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    timestamp: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "checks": self.checks,
            "model": self.model,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PrescreenRecord:
        return cls(
            id=data["id"],
            verdict=data.get("verdict", "ok"),
            reasons=list(data.get("reasons", [])),
            checks=dict(data.get("checks", {})),
            model=data.get("model", ""),
            timestamp=data.get("timestamp", ""),
        )


def default_prescreen_path(dataset_path: str | Path) -> Path:
    """`eval/golden_dataset.jsonl` -> `eval/golden_prescreen.jsonl`, sibling file."""
    p = Path(dataset_path)
    return p.with_name(f"{p.stem.replace('_dataset', '')}_prescreen.jsonl")


def load_prescreen(path: str | Path) -> dict[str, PrescreenRecord]:
    """Every pair id ever pre-screened, keyed by id. Missing file -> empty.

    Never raises on a missing sidecar -- the review UI must work identically
    whether or not the pre-screen has ever been run.
    """
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, PrescreenRecord] = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = PrescreenRecord.from_json(json.loads(line))
                out[rec.id] = rec
    return out


def save_prescreen(records: dict[str, PrescreenRecord], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for pair_id in sorted(records):
            fh.write(json.dumps(records[pair_id].to_json(), ensure_ascii=False))
            fh.write("\n")


def append_prescreen(path: str | Path, record: PrescreenRecord) -> None:
    """Merge one record into the sidecar and persist immediately -- same
    crash-safety discipline as the review ledger's record_decision: a run
    that dies partway through keeps everything decided so far."""
    records = load_prescreen(path)
    records[record.id] = record
    save_prescreen(records, path)


# --------------------------------------------------------------------------
# deterministic checks (no LLM, no cost)
# --------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "that",
    "this", "these", "those", "it", "its", "and", "or", "but", "if", "then",
    "does", "do", "did", "what", "which", "who", "how", "why",
}


def _norm_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def find_near_duplicate_questions(pairs: list[QAPair]) -> dict[str, list[str]]:
    """Exact (post-normalisation) and near-duplicate (high token-overlap)
    question groups, keyed by one representative id -> list of all ids in
    the group. Near-duplicate uses Jaccard over content words so paraphrases
    of the same underlying question ("What is On-Demand Attention?" x2) are
    caught even when the wording differs slightly."""
    norm_by_id = {qa.id: _norm_question(qa.question) for qa in pairs}
    tokens_by_id = {
        qa.id: {
            w
            for w in re.findall(r"[a-z0-9]+", norm_by_id[qa.id])
            if w not in _STOPWORDS and len(w) > 1
        }
        for qa in pairs
    }

    ids = [qa.id for qa in pairs]
    groups: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for i, id_a in enumerate(ids):
        if id_a in assigned:
            continue
        group = [id_a]
        for id_b in ids[i + 1 :]:
            if id_b in assigned:
                continue
            if norm_by_id[id_a] == norm_by_id[id_b]:
                group.append(id_b)
                continue
            ta, tb = tokens_by_id[id_a], tokens_by_id[id_b]
            if not ta or not tb:
                continue
            jaccard = len(ta & tb) / len(ta | tb)
            if jaccard >= 0.75:
                group.append(id_b)
        if len(group) > 1:
            for gid in group:
                assigned.add(gid)
            groups[id_a] = group
    return groups


_STANDALONE_BAD_PATTERNS = [
    re.compile(r"\bthe passage\b", re.IGNORECASE),
    re.compile(r"\bthe text\b", re.IGNORECASE),
    re.compile(r"\bthe study\b", re.IGNORECASE),
    re.compile(r"\bthe paper\b", re.IGNORECASE),
    re.compile(r"\bthis section\b", re.IGNORECASE),
    re.compile(r"\baccording to the (passage|text|excerpt|document)\b", re.IGNORECASE),
]


def question_names_nothing(question: str) -> bool:
    """Heuristic standalone-ness check: true if the question both (a) uses a
    generic backreference like "the passage" and (b) never names a specific
    method/system/paper (a quoted phrase or a capitalised multi-word term
    outside the first word). This is deliberately permissive -- it only
    flags questions that fail BOTH tests, since many real questions use
    'this paper' while still naming a method elsewhere in the sentence."""
    has_backref = any(p.search(question) for p in _STANDALONE_BAD_PATTERNS)
    if not has_backref:
        return False
    # A quoted title ("...") or an acronym/proper-noun run (e.g. "On-Demand
    # Attention", "ODA", "BERT") counts as naming something concrete.
    if re.search(r'"[^"]{3,}"|“[^”]{3,}”', question):
        return False
    words = question.split()
    for w in words[1:]:
        bare = w.strip(",.?:;()")
        if len(bare) >= 2 and (bare.isupper() or (bare[0].isupper() and not bare.isupper() and bare.lower() not in _STOPWORDS)):
            return False
    return True


def answer_too_short(ground_truth: str) -> bool:
    # Only an empty answer is flagged. A length floor flagged 41 correct
    # answers -- "11.9 ms", "GRPO", "O(1)" -- because a numeric or factual
    # question's right answer is usually short.
    return not ground_truth or not ground_truth.strip()


_GENERIC_REFS = {
    "figure", "fig", "table", "section", "sec", "equation", "eq", "appendix",
    "algorithm", "theorem", "lemma", "step", "stage", "phase",
}


def question_names_something(question: str) -> bool:
    """True if the question text itself names a concrete paper, method or
    system: a quoted title, an acronym (ODA, GRPO), or a capitalised or
    digit-bearing term past the first word (Agile-WAM, dQwen3.5-9B).

    The LLM judge's STANDALONE verdict over-fired on 142/180 pairs, including
    questions quoting the full paper title; it overrides the judge only in
    that direction -- a question naming something IS standalone."""
    if re.search(r'"[^"]{3,}"|\u201c[^\u201d]{3,}\u201d|\'[^\']{8,}\'', question):
        return True
    for w in question.split()[1:]:
        bare = w.strip(",.?:;()'\"")
        # "Figure 4" / "Table 2" point INTO a paper without naming it.
        if len(bare) < 2 or bare.lower() in _STOPWORDS or bare.lower() in _GENERIC_REFS:
            continue
        if sum(c.isupper() for c in bare) >= 2 or bare[0].isupper() or (
            any(c.isdigit() for c in bare) and any(c.isalpha() for c in bare)
        ):
            return True
    return False


# --------------------------------------------------------------------------
# LLM judge prompts
# --------------------------------------------------------------------------

_SUPPORT_SYSTEM = (
    "You are a strict fact-checking judge for a retrieval-augmented QA "
    "evaluation set. You are given a source passage, a question, and a "
    "proposed ground-truth answer. Judge three things independently: "
    "(1) SUPPORTED: is the ground-truth answer fully and correctly supported "
    "by the passage (no fabrication, no partial/incomplete answer)? "
    "(2) STANDALONE: would a real user who has NOT seen the passage "
    "understand what paper, method, or system the question refers to, from "
    "the question text alone (it must name something concrete, not say "
    "'the passage' or 'the study')? "
    "(3) ANSWERABLE: does the passage actually contain enough information to "
    "answer the question at all? "
    "Reply with strict JSON only: "
    '{"supported": true|false, "standalone": true|false, "answerable": true|false, '
    '"reason": "one short concrete sentence a reviewer can act on"}'
)

_UNCOVERED_SYSTEM = (
    "You are a strict judge checking whether a set of passages from ONE "
    "research paper actually answers a specific question about that paper. "
    "The question was designed to be UNANSWERABLE from the paper (e.g. asking "
    "for licensing terms, a grant number, electricity cost, annotator pay, or "
    "a release date the paper is not expected to state). You are given up to "
    "5 candidate passages retrieved by keyword search from that paper. If ANY "
    "passage actually states information that answers the question, say so "
    "and quote the exact supporting sentence. Reply with strict JSON only: "
    '{"answered": true|false, "chunk_id": "id or empty string", '
    '"quote": "exact quoted sentence or empty string", '
    '"reason": "one short concrete sentence a reviewer can act on"}'
)


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------
# per-pair checks
# --------------------------------------------------------------------------


def _judge_answerable_pair(
    llm: LLMProvider, qa: QAPair, chunk: Chunk
) -> tuple[dict[str, Any], list[str]]:
    """LLM judge for one answerable pair against its expected chunk. Returns
    (checks dict, reasons list). Never raises on a provider error -- a failed
    call is recorded as an inconclusive check, not a crash."""
    prompt = (
        f"Passage ({chunk.locator()}):\n{chunk.text}\n\n"
        f"Question: {qa.question}\n"
        f"Proposed ground truth: {qa.ground_truth}\n\n"
        "Respond with the JSON object only."
    )
    try:
        resp = llm.complete(
            LLMRequest(system=_SUPPORT_SYSTEM, user=prompt, task="generic", temperature=0.0)
        )
    except ProviderError as exc:
        return {"llm_judge_error": str(exc)}, [f"LLM judge call failed: {exc}"]

    data = _extract_json(resp.text)
    if data is None:
        return (
            {"llm_judge_error": "unparseable response", "raw": resp.text[:300]},
            ["LLM judge returned an unparseable response; needs manual review."],
        )

    reasons: list[str] = []
    if data.get("supported") is False:
        reasons.append(f"Ground truth may not be supported by the source passage: {data.get('reason', '')}".strip())
    if data.get("standalone") is False and not question_names_something(qa.question):
        reasons.append(f"Question may not be standalone (names no paper/method): {data.get('reason', '')}".strip())
    if data.get("answerable") is False:
        reasons.append(f"Passage may not actually answer this question: {data.get('reason', '')}".strip())

    checks = {
        "llm_supported": data.get("supported"),
        "llm_standalone": data.get("standalone"),
        "llm_answerable": data.get("answerable"),
        "llm_reason": data.get("reason", ""),
        "_usage": dict(resp.usage) if resp.usage else {},
    }
    return checks, reasons


def _judge_uncovered_pair(
    llm: LLMProvider, qa: QAPair, candidates: list[Chunk]
) -> tuple[dict[str, Any], list[str]]:
    """LLM judge for one in-domain unanswerable ('plausible but uncovered')
    pair against the top BM25 candidates from its own paper."""
    if not candidates:
        return {"bm25_candidates": 0}, []

    blocks = "\n\n".join(
        f"[{c.chunk_id}] {c.text[:1200]}" for c in candidates
    )
    prompt = (
        f"Question about the paper: {qa.question}\n\n"
        f"Candidate passages from that paper (top BM25 matches):\n{blocks}\n\n"
        "Respond with the JSON object only."
    )
    try:
        resp = llm.complete(
            LLMRequest(system=_UNCOVERED_SYSTEM, user=prompt, task="generic", temperature=0.0)
        )
    except ProviderError as exc:
        return (
            {"bm25_candidates": len(candidates), "llm_judge_error": str(exc)},
            [f"LLM judge call failed: {exc}"],
        )

    data = _extract_json(resp.text)
    if data is None:
        return (
            {"bm25_candidates": len(candidates), "llm_judge_error": "unparseable response"},
            ["LLM judge returned an unparseable response; needs manual review."],
        )

    reasons: list[str] = []
    checks: dict[str, Any] = {
        "bm25_candidates": len(candidates),
        "llm_answered": data.get("answered"),
        "llm_chunk_id": data.get("chunk_id", ""),
        "llm_quote": data.get("quote", ""),
        "llm_reason": data.get("reason", ""),
        "_usage": dict(resp.usage) if resp.usage else {},
    }
    if data.get("answered") is True:
        chunk_id = data.get("chunk_id", "") or "(unspecified)"
        quote = data.get("quote", "")
        reasons.append(
            f"Paper appears to answer this after all -- chunk {chunk_id} may state: "
            f"“{quote}”" if quote else
            f"Paper appears to answer this after all -- see chunk {chunk_id}."
        )
    return checks, reasons


def _bm25_search_within_doc(question: str, doc_chunks: list[Chunk], top_n: int = 5) -> list[Chunk]:
    if not doc_chunks:
        return []
    tokenized = [tokenize(c.text) for c in doc_chunks]
    q_tokens = tokenize(question)
    if not q_tokens:
        return []
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(range(len(doc_chunks)), key=lambda i: scores[i], reverse=True)
    return [doc_chunks[i] for i in ranked[:top_n] if scores[i] > 0]


# --------------------------------------------------------------------------
# top-level driver
# --------------------------------------------------------------------------


def prescreen_pair(
    qa: QAPair,
    *,
    llm: LLMProvider,
    chunk_map: dict[str, Chunk],
    chunks_by_doc: dict[str, list[Chunk]],
    duplicate_groups: dict[str, set[str]],
    model_name: str,
) -> PrescreenRecord:
    """Run every applicable check for one pair and return its record.

    Never touches the review ledger. Pure function of its inputs plus one
    LLM call (at most) -- safe to call repeatedly / out of order / resumed.
    """
    reasons: list[str] = []
    checks: dict[str, Any] = {}

    if answer_too_short(qa.ground_truth):
        reasons.append("Ground truth is empty or too short to be a real answer.")
        checks["too_short"] = True

    if qa.id in duplicate_groups:
        group = sorted(duplicate_groups[qa.id])
        checks["duplicate_group"] = group
        reasons.append(f"Near-duplicate of {', '.join(g for g in group if g != qa.id)}.")

    if not qa.unanswerable:
        if question_names_nothing(qa.question):
            reasons.append(
                "Question uses a generic backreference ('the passage'/'the study') "
                "without naming a paper, method, or system -- likely not standalone."
            )
            checks["heuristic_standalone_fail"] = True

        chunk = chunk_map.get(qa.expected_chunk_ids[0]) if qa.expected_chunk_ids else None
        if chunk is None:
            reasons.append("No resolvable source chunk for this answerable pair.")
            checks["missing_chunk"] = True
        else:
            llm_checks, llm_reasons = _judge_answerable_pair(llm, qa, chunk)
            checks.update(llm_checks)
            reasons.extend(llm_reasons)
    else:
        if qa.doc_id is None:
            checks["off_domain"] = True
            # Off-domain probes need no LLM call -- domain membership is
            # settled by construction (real-world trivia questions), not by
            # a judge that could hallucinate an on-topic connection.
        else:
            doc_chunks = chunks_by_doc.get(qa.doc_id, [])
            candidates = _bm25_search_within_doc(qa.question, doc_chunks, top_n=5)
            llm_checks, llm_reasons = _judge_uncovered_pair(llm, qa, candidates)
            checks.update(llm_checks)
            reasons.extend(llm_reasons)

    verdict: Verdict = "suspect" if reasons else "ok"
    return PrescreenRecord(
        id=qa.id,
        verdict=verdict,
        reasons=reasons,
        checks=checks,
        model=model_name,
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def build_indices(
    pairs: list[QAPair], chunks: list[Chunk]
) -> tuple[dict[str, Chunk], dict[str, list[Chunk]], dict[str, set[str]]]:
    """Shared lookup structures for a prescreen run: chunk-by-id, chunks
    grouped by doc_id, and near-duplicate question groups (id -> full group,
    including itself, for O(1) lookup per pair)."""
    chunk_map = {c.chunk_id: c for c in chunks}
    chunks_by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        chunks_by_doc.setdefault(c.doc_id, []).append(c)
    raw_groups = find_near_duplicate_questions(pairs)
    duplicate_groups: dict[str, set[str]] = {}
    for group in raw_groups.values():
        gset = set(group)
        for gid in group:
            duplicate_groups[gid] = gset
    return chunk_map, chunks_by_doc, duplicate_groups

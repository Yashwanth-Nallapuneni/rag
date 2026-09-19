"""Golden evaluation dataset: drafting, storage, review ledger, validation.

The spec this serves is blunt: 50-200 QA pairs, *manually verified against
the corpus*, is what makes eval numbers mean anything. This module's job is
to get a human to that verification as fast as possible without ever letting
a pair count as verified that no human actually looked at.

Two files, two concerns, never merged:
  - the dataset itself (`QAPair` rows: question, ground truth, provenance)
  - the review ledger (who approved/rejected which pair id, and when)

Keeping them separate means re-running the drafter (to fix a heuristic bug,
add more pairs, whatever) can never accidentally reset or fabricate approval
state -- the ledger is keyed by pair id and untouched by drafting.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..config import Settings
from ..providers import LLMRequest, ProviderError, get_llm
from ..schemas import Chunk, QAPair
from ..tokenization import count_tokens

ReviewStatus = Literal["draft", "verified", "rejected"]

DEFAULT_CATEGORIES = ("factual", "multi_hop", "numeric", "definition", "comparison")

# Off-domain questions used for a slice of the unanswerable set: plausible
# questions nobody would expect this corpus (arXiv ML/CS papers) to answer.
_OFF_DOMAIN_QUESTIONS = [
    "What is the boiling point of liquid nitrogen at sea level?",
    "Who won the FIFA World Cup in 1998?",
    "What are the side effects of ibuprofen?",
    "What is the current population of Iceland?",
    "How do you prune a rose bush in early spring?",
    "What year did the Berlin Wall fall?",
    "What is the recommended daily intake of vitamin C?",
    "How does a four-stroke combustion engine work?",
    "What is the capital of Mongolia?",
    "How long should a soft-boiled egg be cooked?",
    "Who wrote the novel One Hundred Years of Solitude?",
    "What causes the northern lights?",
    "What is the tallest mountain in Africa?",
    "How is a sourdough starter maintained?",
    "What is the half-life of carbon-14?",
    "Which planet has the most moons?",
    "What are the rules of offside in ice hockey?",
    "How do you change a flat bicycle tire?",
    "What is the main ingredient in traditional guacamole?",
    "When was the Eiffel Tower completed?",
]

# In-domain but uncovered: a real paper's title with a detail papers of this
# kind rarely state. Rotated so the refusal set is not one template repeated
# -- a repeated probe measures one question many times, not refusal accuracy.
# The reviewer must still confirm each is truly absent from the paper.
_UNCOVERED_TEMPLATES = [
    'What dataset licensing terms does "{title}" use for its released code and data?',
    'What was the total electricity cost in US dollars of the experiments in "{title}"?',
    'Which funding agency grant number supported the work in "{title}"?',
    'How many human annotators were paid, and at what hourly rate, for "{title}"?',
    'On what date was the code for "{title}" first released publicly?',
]

_SECTION_SKIP_RE = re.compile(
    r"reference|bibliograph|acknowledg|author|affiliation|appendix",
    re.IGNORECASE,
)
_MIN_CHUNK_TOKENS = 150


# --------------------------------------------------------------------------
# dataset I/O
# --------------------------------------------------------------------------


def load_dataset(path: str | Path) -> list[QAPair]:
    """Read the golden dataset. Missing file -> empty list (nothing drafted yet)."""
    p = Path(path)
    if not p.exists():
        return []
    pairs: list[QAPair] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(QAPair.model_validate_json(line))
    return pairs


def save_dataset(pairs: list[QAPair], path: str | Path) -> None:
    """Write pairs as JSONL, sorted by id for a stable, diffable file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(pairs, key=lambda qa: qa.id)
    with p.open("w", encoding="utf-8") as fh:
        for qa in ordered:
            fh.write(qa.model_dump_json())
            fh.write("\n")


# --------------------------------------------------------------------------
# review ledger -- the ONLY source of truth for verification state
# --------------------------------------------------------------------------


@dataclass
class ReviewRecord:
    pair_id: str
    status: ReviewStatus = "draft"
    reviewer: str = ""
    decided_at: str = ""
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "status": self.status,
            "reviewer": self.reviewer,
            "decided_at": self.decided_at,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ReviewRecord":
        return cls(
            pair_id=data["pair_id"],
            status=data.get("status", "draft"),
            reviewer=data.get("reviewer", ""),
            decided_at=data.get("decided_at", ""),
            notes=data.get("notes", ""),
        )


def default_review_path(dataset_path: str | Path) -> Path:
    """`eval/golden_dataset.jsonl` -> `eval/golden_review.jsonl`, sibling file."""
    p = Path(dataset_path)
    return p.with_name(f"{p.stem.replace('_dataset', '')}_review.jsonl")


def load_review_ledger(path: str | Path) -> dict[str, ReviewRecord]:
    """Every pair id ever decided on, keyed by id. Missing file -> empty."""
    p = Path(path)
    if not p.exists():
        return {}
    ledger: dict[str, ReviewRecord] = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = ReviewRecord.from_json(json.loads(line))
                ledger[rec.pair_id] = rec
    return ledger


def save_review_ledger(ledger: dict[str, ReviewRecord], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for pair_id in sorted(ledger):
            fh.write(json.dumps(ledger[pair_id].to_json(), ensure_ascii=False))
            fh.write("\n")


def record_decision(
    ledger_path: str | Path,
    pair_id: str,
    status: ReviewStatus,
    *,
    reviewer: str = "",
    notes: str = "",
) -> ReviewRecord:
    """Append/update one decision and persist immediately.

    Called after every single reviewer action (never batched) so a crash or
    a closed tab loses at most the in-flight click, not a session's work.
    """
    ledger = load_review_ledger(ledger_path)
    rec = ReviewRecord(
        pair_id=pair_id,
        status=status,
        reviewer=reviewer,
        decided_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )
    ledger[pair_id] = rec
    save_review_ledger(ledger, ledger_path)
    return rec


def review_status(pair_id: str, ledger: dict[str, ReviewRecord]) -> ReviewStatus:
    rec = ledger.get(pair_id)
    return rec.status if rec else "draft"


def load_verified(
    dataset_path: str | Path, ledger_path: str | Path | None = None
) -> list[QAPair]:
    """Pairs a human has explicitly approved. This -- not the raw dataset --
    is what the eval harness must read. A pair absent from the ledger, or
    present with any status other than 'verified', is excluded: verification
    is opt-in, never assumed."""
    ledger_path = ledger_path or default_review_path(dataset_path)
    pairs = load_dataset(dataset_path)
    ledger = load_review_ledger(ledger_path)
    return [qa for qa in pairs if review_status(qa.id, ledger) == "verified"]


# --------------------------------------------------------------------------
# drafting
# --------------------------------------------------------------------------


def _eligible_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Drop thin, boilerplate, or reference/author-list chunks -- questions
    drafted from those are either trivial or unanswerable-by-accident."""
    out = []
    for c in chunks:
        if c.token_count and c.token_count < _MIN_CHUNK_TOKENS:
            continue
        if not c.token_count and count_tokens(c.text) < _MIN_CHUNK_TOKENS:
            continue
        section = c.section_label
        if section and _SECTION_SKIP_RE.search(section):
            continue
        out.append(c)
    return out


def _spread_sample(chunks: list[Chunk], n: int, rng: random.Random) -> list[Chunk]:
    """Sample across different documents (and within a doc, different
    sections) round-robin, so 150 questions aren't all about one intro."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    for doc_chunks in by_doc.values():
        rng.shuffle(doc_chunks)
        # prefer section diversity within a doc: keep at most one chunk per
        # section_label consecutively by sorting distinct sections first
        doc_chunks.sort(key=lambda c: c.section_label)

    doc_ids = list(by_doc)
    rng.shuffle(doc_ids)
    picked: list[Chunk] = []
    seen_sections: dict[str, set[str]] = {d: set() for d in doc_ids}
    cursors = {d: 0 for d in doc_ids}
    while len(picked) < n and any(cursors[d] < len(by_doc[d]) for d in doc_ids):
        for d in doc_ids:
            if len(picked) >= n:
                break
            pool = by_doc[d]
            i = cursors[d]
            while i < len(pool) and pool[i].section_label in seen_sections[d]:
                i += 1
            if i >= len(pool):
                cursors[d] = len(pool)
                continue
            picked.append(pool[i])
            seen_sections[d].add(pool[i].section_label)
            cursors[d] = i + 1
    return picked[:n]


_LLM_DRAFT_SYSTEM = (
    "You write evaluation questions for a retrieval-augmented QA system. "
    "Given ONE passage, write exactly one question of the requested category "
    "that is answerable using ONLY this passage, plus a concise ground-truth "
    "answer drawn only from it. The question must read as a real user's, "
    "standalone: never refer to 'the passage', 'the text' or 'the study' "
    "without naming it -- the user has not seen the passage. Name the "
    "method, system or paper the question is about instead. "
    "Reply with strict JSON: "
    '{"question": "...", "ground_truth": "..."}. No other text.'
)

_CATEGORY_HINT = {
    "factual": "a direct factual question about a specific claim in the passage",
    "multi_hop": "a question that requires connecting two details stated in the passage",
    "numeric": "a question whose answer is a number, percentage, or measurement in the passage",
    "definition": "a question asking what a term or method defined in the passage means",
    "comparison": "a question comparing two things (methods, values, conditions) mentioned in the passage",
}


def _llm_draft_pair(llm, chunk: Chunk, category: str) -> tuple[str, str] | None:
    hint = _CATEGORY_HINT.get(category, _CATEGORY_HINT["factual"])
    prompt = (
        f"Category: {category} ({hint})\n\n"
        f"Passage ({chunk.locator()}):\n{chunk.text}\n\n"
        "Respond with the JSON object only."
    )
    try:
        resp = llm.complete(
            LLMRequest(system=_LLM_DRAFT_SYSTEM, user=prompt, task="generic", temperature=0.3)
        )
    except ProviderError:
        return None
    match = re.search(r"\{.*\}", resp.text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    question = str(data.get("question", "")).strip()
    truth = str(data.get("ground_truth", "")).strip()
    if not question or not truth:
        return None
    return question, truth


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _heuristic_draft_pair(chunk: Chunk, category: str, rng: random.Random) -> tuple[str, str]:
    """Deterministic fallback so drafting works offline with the mock
    provider (no API key, no network, fully testable in CI).

    This is intentionally crude -- it turns one sentence of the passage into
    a question stem. Quality is lower than an LLM draft, so callers must mark
    these clearly in `notes` for the human reviewer.
    """
    sentences = [s.strip() for s in _SENT_RE.split(chunk.text) if len(s.strip()) > 25]
    sentence = rng.choice(sentences) if sentences else chunk.text[:200].strip()
    topic = chunk.section_label or chunk.doc_title
    numbers = re.findall(r"\b\d[\d,.]*%?\b", sentence)

    if category == "numeric" and numbers:
        question = f"According to \"{topic}\", what figure is reported in: “{sentence[:140]}…”?"
    elif category == "definition":
        question = f"Based on the section \"{topic}\", what does this passage say is being described here: “{sentence[:140]}…”?"
    elif category == "comparison":
        question = f"In \"{topic}\", what comparison or contrast is drawn in the statement: “{sentence[:140]}…”?"
    elif category == "multi_hop":
        question = f"Drawing on \"{topic}\", what does the passage establish when it states: “{sentence[:140]}…”?"
    else:
        question = f"According to \"{topic}\", what is stated in: “{sentence[:140]}…”?"

    return question, sentence


def _unanswerable_pair(
    idx: int,
    chunks: list[Chunk],
    rng: random.Random,
    off_domain: bool,
) -> QAPair:
    if off_domain or not chunks:
        question = _OFF_DOMAIN_QUESTIONS[idx % len(_OFF_DOMAIN_QUESTIONS)]
        truth = "The corpus does not cover this topic; no answer is supported."
        doc_id = None
    else:
        # Plausible in-domain question the corpus happens not to answer: ask
        # about a real paper's title/topic but for a detail no chunk covers.
        c = chunks[idx % len(chunks)]
        template = _UNCOVERED_TEMPLATES[idx % len(_UNCOVERED_TEMPLATES)]
        question = template.format(title=c.doc_title)
        truth = (
            "The corpus does not contain this information; no answer is "
            "supported by the indexed documents."
        )
        doc_id = c.doc_id
    return QAPair(
        id=f"qa-unans-{idx:04d}",
        question=question,
        ground_truth=truth,
        doc_id=doc_id,
        expected_chunk_ids=[],
        expected_sources=[],
        category="factual",
        unanswerable=True,
        notes="unanswerable: off-domain probe" if off_domain else "unanswerable: plausible but uncovered",
    )


def draft_candidates(
    settings: Settings,
    chunks: list[Chunk],
    n: int = 150,
    categories: tuple[str, ...] | None = None,
    llm: Any | None = None,
    unanswerable_ratio: float = 0.15,
    seed: int | None = None,
) -> list[QAPair]:
    """Draft `n` candidate QAPairs from real corpus chunks.

    Uses the configured LLM when available; falls back to a deterministic
    heuristic (clearly flagged in `notes`) so this works offline with the
    `mock` provider. Roughly `unanswerable_ratio` of the output is
    `unanswerable=True` (mixed in-domain-but-uncovered and off-domain), the
    rest spread across `categories` and sampled from different documents and
    sections.
    """
    # A caller passing categories=None (the CLI does, when no flag is given)
    # must get the defaults rather than a TypeError.
    categories = tuple(categories) if categories else DEFAULT_CATEGORIES
    rng = random.Random(seed)
    n_unans = round(n * unanswerable_ratio)
    n_ans = n - n_unans

    pool = _eligible_chunks(chunks)
    if not pool:
        pool = list(chunks)
    sampled = _spread_sample(pool, n_ans, rng)

    if llm is None and settings.llm.provider:
        try:
            llm = get_llm(settings)
        except ProviderError:
            llm = None
    use_llm = llm is not None and getattr(llm, "name", "") != "mock"

    pairs: list[QAPair] = []
    for i, chunk in enumerate(sampled):
        category = categories[i % len(categories)]
        drafted = _llm_draft_pair(llm, chunk, category) if use_llm else None
        note = ""
        if drafted is None:
            drafted = _heuristic_draft_pair(chunk, category, rng)
            note = "heuristic-drafted: lower quality, verify carefully"
        question, truth = drafted
        pairs.append(
            QAPair(
                id=f"qa-{chunk.chunk_id.replace('::', '-')}-{category}",
                question=question,
                ground_truth=truth,
                doc_id=chunk.doc_id,
                expected_chunk_ids=[chunk.chunk_id],
                expected_sources=[chunk.locator()],
                category=category,
                unanswerable=False,
                notes=note,
            )
        )

    # Distinct papers for the in-domain probes, a shuffled off-domain list:
    # indices below walk each without replacement.
    off_domain_count = n_unans // 2
    by_doc: dict[str, Chunk] = {}
    for c in sampled:
        by_doc.setdefault(c.doc_id, c)
    uncovered_pool = list(by_doc.values())
    rng.shuffle(uncovered_pool)
    off_order = list(range(len(_OFF_DOMAIN_QUESTIONS)))
    rng.shuffle(off_order)
    for i in range(n_unans):
        off = i < off_domain_count
        idx = off_order[i % len(off_order)] if off else i - off_domain_count
        qa = _unanswerable_pair(idx, uncovered_pool, rng, off_domain=off)
        pairs.append(qa.model_copy(update={"id": f"qa-unans-{i:04d}"}))

    rng.shuffle(pairs)
    return pairs


# --------------------------------------------------------------------------
# validation + stats
# --------------------------------------------------------------------------


def _norm_question(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


@dataclass
class ValidationReport:
    total: int
    duplicate_questions: list[list[str]] = field(default_factory=list)
    missing_ground_truth: list[str] = field(default_factory=list)
    unknown_chunk_ids: list[tuple[str, str]] = field(default_factory=list)
    unanswerable_with_expected_chunks: list[str] = field(default_factory=list)
    category_counts: dict[str, int] = field(default_factory=dict)
    answerable_count: int = 0
    unanswerable_count: int = 0
    unanswerable_ratio: float = 0.0
    in_spec_range: bool = False
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.duplicate_questions
            or self.missing_ground_truth
            or self.unknown_chunk_ids
            or self.unanswerable_with_expected_chunks
        ) and self.in_spec_range

    def summary(self) -> str:
        lines = [
            f"total={self.total} answerable={self.answerable_count} "
            f"unanswerable={self.unanswerable_count} "
            f"({self.unanswerable_ratio:.0%})",
            f"in spec range (50-200): {self.in_spec_range}",
            f"category counts: {dict(sorted(self.category_counts.items()))}",
            f"duplicate question groups: {len(self.duplicate_questions)}",
            f"missing ground_truth: {len(self.missing_ground_truth)}",
            f"unknown expected_chunk_ids: {len(self.unknown_chunk_ids)}",
            f"unanswerable pairs wrongly carrying expected chunks: "
            f"{len(self.unanswerable_with_expected_chunks)}",
        ]
        lines.extend(f"ISSUE: {i}" for i in self.issues)
        return "\n".join(lines)


def validate_dataset(pairs: list[QAPair], store: Any | None = None) -> ValidationReport:
    """Catch the mistakes a hurried human reviewer would miss: duplicate
    questions, empty answers, dangling chunk ids, unanswerable pairs that
    still point at evidence, skewed category/answerability balance, and the
    dataset falling outside the spec's 50-200 pair window."""
    total = len(pairs)
    report = ValidationReport(total=total)

    by_question: dict[str, list[str]] = {}
    for qa in pairs:
        by_question.setdefault(_norm_question(qa.question), []).append(qa.id)
    report.duplicate_questions = [ids for ids in by_question.values() if len(ids) > 1]

    report.missing_ground_truth = [
        qa.id for qa in pairs if not qa.ground_truth or not qa.ground_truth.strip()
    ]

    report.unanswerable_with_expected_chunks = [
        qa.id for qa in pairs if qa.unanswerable and qa.expected_chunk_ids
    ]

    known_ids: set[str] | None = None
    if store is not None:
        wanted = sorted({cid for qa in pairs for cid in qa.expected_chunk_ids})
        if wanted:
            found = {c.chunk_id for c in store.get(wanted)}
            known_ids = found
            for qa in pairs:
                for cid in qa.expected_chunk_ids:
                    if cid not in found:
                        report.unknown_chunk_ids.append((qa.id, cid))

    cat_counts = Counter(qa.category for qa in pairs)
    report.category_counts = dict(cat_counts)

    report.unanswerable_count = sum(1 for qa in pairs if qa.unanswerable)
    report.answerable_count = total - report.unanswerable_count
    report.unanswerable_ratio = report.unanswerable_count / total if total else 0.0
    report.in_spec_range = 50 <= total <= 200

    if not report.in_spec_range:
        report.issues.append(
            f"dataset has {total} pairs; spec wants 50-200"
        )
    if total and not (0.05 <= report.unanswerable_ratio <= 0.30):
        report.issues.append(
            f"unanswerable ratio {report.unanswerable_ratio:.0%} looks off "
            "(expected roughly 10-20%)"
        )
    if len(cat_counts) < 2 and total > 5:
        report.issues.append("questions are not spread across categories")
    if known_ids is None and any(qa.expected_chunk_ids for qa in pairs):
        report.issues.append(
            "no vector store passed -- expected_chunk_ids were not checked against the index"
        )

    return report


def dataset_stats(pairs: list[QAPair], ledger: dict[str, ReviewRecord] | None = None) -> dict[str, Any]:
    """Counts a reviewer or CLI cares about: by category, by document,
    verified vs draft, answerable vs unanswerable."""
    ledger = ledger or {}
    by_category = Counter(qa.category for qa in pairs)
    by_doc = Counter(qa.doc_id or "(none)" for qa in pairs)
    by_status = Counter(review_status(qa.id, ledger) for qa in pairs)
    unanswerable = sum(1 for qa in pairs if qa.unanswerable)
    return {
        "total": len(pairs),
        "by_category": dict(by_category),
        "by_document": dict(by_doc),
        "by_review_status": {
            "draft": by_status.get("draft", 0),
            "verified": by_status.get("verified", 0),
            "rejected": by_status.get("rejected", 0),
        },
        "answerable": len(pairs) - unanswerable,
        "unanswerable": unanswerable,
    }

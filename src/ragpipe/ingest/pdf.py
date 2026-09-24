"""PDF parser: multi-column-aware extraction into `Block`s.

Two failure modes make naive PDF extraction actively harmful to a RAG
pipeline rather than just imperfect:

1. arXiv papers are overwhelmingly two-column, and a plain top-to-bottom
   sort of `page.get_text("blocks")` interleaves the columns -- a sentence
   from the left column is followed by an unrelated sentence from the
   right. That garbage gets embedded and cited. So this module detects the
   column layout per page and orders text left-column-then-right-column,
   with full-width elements (titles, abstracts, wide figures/tables) kept
   as their own rows at the correct vertical position.
2. Running headers/footers and page numbers are boilerplate that pollutes
   every chunk they touch. They are identified by *repetition* across
   pages (never by blind margin cropping alone), so a one-off arXiv stamp
   on page 1 survives while a header repeated on every page is stripped
   and recorded in `ParsedDocument.dropped` for audit.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf

from ..config import IngestConfig
from ..schemas import SourceDocument, SourceType
from .base import Block, BlockKind, ParsedDocument, normalize_block_text, normalize_whitespace

# Bold bit in PyMuPDF span flags (bit 4 of the font-descriptor-style flags).
_BOLD_FLAG = 1 << 4

_PAGE_NUM_RE = re.compile(
    r"^\s*[-–—]?\s*(?:page\s*)?\d{1,4}\s*(?:of\s*\d{1,4})?\s*[-–—]?\s*$",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})[.\s]+(\S.*)$")
# IEEE-style section numbering: "I. INTRODUCTION" (roman, all-caps body) and
# "A. Dataset" (single letter, title-case body). Captured together because
# telling them apart needs the *rest* of the line, not just the prefix --
# both are a single uppercase-letter run followed by ". ".
_ALPHA_HEADING_RE = re.compile(r"^([A-Z]{1,6})\.\s+(\S.*)$")
_ROMAN_RE = re.compile(r"^[IVXLCDM]+$")
# "TABLE I" / "Fig. 3" style captions, numbered either with digits or with
# the roman numerals IEEE tables commonly use -- these must never be read as
# section headings even though "TABLE I" would otherwise match the roman
# heading shape above.
_CAPTION_RE = re.compile(
    r"^(Figure|Fig\.|Table|Algorithm|Listing)\s*(\d+|[IVXLCDM]+)\b", re.IGNORECASE
)
_DIGITS_RE = re.compile(r"\d+")
# Font-name fragments that signal a heading weight/style distinct from body
# text, independent of point size (see _font_signals_heading).
_HEADING_FONT_RE = re.compile(r"(medi|bold|bd|black|semibold)", re.IGNORECASE)

_STANDARD_SECTIONS = {
    "abstract", "introduction", "related work", "background",
    "method", "methods", "methodology", "experiments", "experiment",
    "results", "discussion", "conclusion", "conclusions", "limitations",
    "references", "bibliography", "acknowledgements", "acknowledgments",
    "appendix",
}
_REFERENCE_HEADINGS = {"references", "bibliography"}


class PDFParseError(Exception):
    """A PDF could not be opened or read; never let a bare pymupdf/fitz
    exception escape this module."""


def _normalize_furniture(text: str) -> str:
    """Fold a header/footer candidate to a repetition key: case- and
    digit-insensitive, so 'Page 3' and 'Page 4' collapse to one pattern."""
    return _DIGITS_RE.sub("#", text.strip().lower())


def _is_bold(span: dict[str, Any]) -> bool:
    return bool(span.get("flags", 0) & _BOLD_FLAG)


class _RawBlock:
    """One PyMuPDF text block plus the typographic stats we need for
    heading detection and column ordering."""

    __slots__ = ("page", "bbox", "text", "size", "bold", "n_lines", "font")

    def __init__(self, page: int, bbox: tuple[float, float, float, float], text: str,
                 size: float, bold: bool, n_lines: int, font: str = ""):
        self.page = page
        self.bbox = bbox
        self.text = text
        self.size = size
        self.bold = bold
        self.n_lines = n_lines
        self.font = font

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


def _extract_raw_blocks(page: pymupdf.Page, page_num: int) -> list[_RawBlock]:
    """Pull text blocks with bbox + dominant font size/weight from a page.

    Uses `get_text("dict")` rather than `get_text("blocks", sort=True)`
    because reading order here is decided ourselves from geometry, and the
    dict form is what exposes per-span font size and bold flags needed for
    heading detection.
    """
    out: list[_RawBlock] = []
    d = page.get_text("dict")
    for b in d.get("blocks", []):
        lines = b.get("lines")
        if not lines:
            continue  # image or other non-text block
        line_texts: list[str] = []
        sizes: list[float] = []
        bold_chars = 0
        total_chars = 0
        font_chars: Counter[str] = Counter()
        for line in lines:
            spans = line.get("spans", [])
            line_text = "".join(s.get("text", "") for s in spans)
            if line_text.strip() == "":
                continue
            line_texts.append(line_text)
            for s in spans:
                n = len(s.get("text", ""))
                sizes.append(s.get("size", 0.0))
                total_chars += n
                font_chars[s.get("font", "")] += n
                if _is_bold(s):
                    bold_chars += n
        text = "\n".join(line_texts).strip()
        if not text:
            continue
        size = max(sizes) if sizes else 0.0
        bold = total_chars > 0 and (bold_chars / total_chars) > 0.6
        font = font_chars.most_common(1)[0][0] if font_chars else ""
        out.append(_RawBlock(page_num, tuple(b["bbox"]), text, size, bold, len(line_texts), font))
    return out


def _is_page_number(text: str) -> bool:
    return bool(_PAGE_NUM_RE.match(text.strip()))


# A running head can sit below the geometric margin band -- journal styles
# often place it a centimetre or two into the page. Treating the first and
# last block of a page as furniture-eligible catches those, and is safe
# because repetition across pages remains the actual test: a first block
# that is real content differs on every page and is never dropped.
_ORDINAL_FURNITURE_MAX_CHARS = 100


def _furniture_zone(
    blk: _RawBlock,
    page_blocks: list[_RawBlock],
    top_y: float,
    bot_y: float,
) -> str | None:
    """Return "header"/"footer" if this block could be running furniture."""
    if blk.y1 <= top_y:
        return "header"
    if blk.y0 >= bot_y:
        return "footer"
    if len(blk.text.strip()) <= _ORDINAL_FURNITURE_MAX_CHARS and page_blocks:
        if blk.bbox == page_blocks[0].bbox:
            return "header"
        if blk.bbox == page_blocks[-1].bbox:
            return "footer"
    return None


def _collect_furniture(
    pages: list[list[_RawBlock]],
    page_height: float,
    margin_ratio: float,
    min_repeat_ratio: float,
) -> tuple[set[int], dict[str, list[str]]]:
    """Identify header/footer/page-number blocks by repetition across pages.

    Returns the set of `id()`-keyed... (blocks are not hashable by identity
    reliably across a list, so instead we return a set of (page, bbox, text)
    keys to drop) and the audit trail for `ParsedDocument.dropped`.
    """
    top_y = page_height * margin_ratio
    bot_y = page_height * (1 - margin_ratio)
    n_pages = len(pages)

    header_pattern_pages: dict[str, set[int]] = {}
    footer_pattern_pages: dict[str, set[int]] = {}
    header_samples: dict[str, str] = {}
    footer_samples: dict[str, str] = {}

    all_pages = {blk.page for blocks in pages for blk in blocks}
    to_drop: set[tuple[int, tuple[float, float, float, float]]] = set()
    dropped: dict[str, list[str]] = {"headers": [], "footers": [], "page_numbers": []}

    for blocks in pages:
        ordered = sorted(blocks, key=lambda b: (b.y0, b.x0))
        for blk in blocks:
            zone = _furniture_zone(blk, ordered, top_y, bot_y)
            if zone is None:
                continue
            in_header, in_footer = zone == "header", zone == "footer"
            if _is_page_number(blk.text):
                to_drop.add((blk.page, blk.bbox))
                dropped["page_numbers"].append(blk.text.strip())
                continue
            key = _normalize_furniture(blk.text)
            if not key:
                continue
            if in_header:
                header_pattern_pages.setdefault(key, set()).add(blk.page)
                header_samples.setdefault(key, blk.text.strip())
            else:
                footer_pattern_pages.setdefault(key, set()).add(blk.page)
                footer_samples.setdefault(key, blk.text.strip())

    def _repeats(pgs: set[int]) -> bool:
        """Is this line running furniture rather than content?

        Plain repetition across all pages is the common case. The parity
        checks handle alternating recto/verso running heads -- a journal
        style that puts the authors on even pages and the title on odd
        ones, so each head appears on only ~half the pages and slips under
        a whole-document threshold. Requiring 3+ occurrences keeps this
        from firing on a two-page document by accident.
        """
        if len(pgs) / n_pages >= min_repeat_ratio:
            return True
        if len(pgs) < 3:
            return False
        for parity in (0, 1):
            same = {p for p in all_pages if p % 2 == parity}
            if not same or not pgs <= same:
                continue
            if len(pgs) / len(same) >= min_repeat_ratio:
                return True
        return False

    repeated_header_keys = {k for k, pgs in header_pattern_pages.items() if _repeats(pgs)}
    repeated_footer_keys = {k for k, pgs in footer_pattern_pages.items() if _repeats(pgs)}

    for blocks in pages:
        ordered = sorted(blocks, key=lambda b: (b.y0, b.x0))
        for blk in blocks:
            zone = _furniture_zone(blk, ordered, top_y, bot_y)
            if zone is None or _is_page_number(blk.text):
                continue
            in_header, in_footer = zone == "header", zone == "footer"
            key = _normalize_furniture(blk.text)
            if in_header and key in repeated_header_keys:
                to_drop.add((blk.page, blk.bbox))
            elif in_footer and key in repeated_footer_keys:
                to_drop.add((blk.page, blk.bbox))

    dropped["headers"] = sorted({header_samples[k] for k in repeated_header_keys})
    dropped["footers"] = sorted({footer_samples[k] for k in repeated_footer_keys})
    dropped["page_numbers"] = sorted(set(dropped["page_numbers"]))
    return to_drop, dropped


def _detect_two_column(blocks: list[_RawBlock], page_width: float) -> bool:
    """Bimodal test: do narrow blocks cluster left/right of page centre?"""
    center = page_width / 2
    margin = page_width * 0.04
    narrow = [b for b in blocks if (b.x1 - b.x0) < 0.55 * page_width]
    left = sum(1 for b in narrow if (b.x0 + b.x1) / 2 < center - margin)
    right = sum(1 for b in narrow if (b.x0 + b.x1) / 2 > center + margin)
    return left >= 2 and right >= 2


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _order_page_blocks(blocks: list[_RawBlock], page_width: float) -> list[_RawBlock]:
    """Column-aware reading order: full-width rows stay in vertical position;
    two-column bands are read left-column-complete then right-column.

    A block counts as full width either by measured width, or -- the case a
    plain width cutoff misses -- by being a short, centred line (a wrapped
    title, an author list) that lines up with neither column's left edge.
    """
    if not blocks:
        return []
    center = page_width / 2
    margin = page_width * 0.04
    two_col = _detect_two_column(blocks, page_width)
    if not two_col:
        return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))

    wide_cut = 0.6 * page_width
    clearly_left = [b for b in blocks if (b.x1 - b.x0) < wide_cut and (b.x0 + b.x1) / 2 < center - margin]
    clearly_right = [b for b in blocks if (b.x1 - b.x0) < wide_cut and (b.x0 + b.x1) / 2 > center + margin]
    left_x0 = _median([b.x0 for b in clearly_left])
    right_x0 = _median([b.x0 for b in clearly_right])
    # Wide enough to absorb a heading's extra left indent/kerning relative
    # to the body-text column edge (e.g. a numbered section heading can sit
    # ~4% of the page width right of where that column's paragraphs start)
    # without also swallowing a genuinely full-width row.
    col_tol = page_width * 0.05

    def classify(blk: _RawBlock) -> int | None:
        """Returns 0 (left), 1 (right), or None (full-width row)."""
        if (blk.x1 - blk.x0) >= wide_cut:
            return None
        if left_x0 is not None and abs(blk.x0 - left_x0) <= col_tol:
            return 0
        if right_x0 is not None and abs(blk.x0 - right_x0) <= col_tol:
            return 1
        # Doesn't align with either column's left edge -- either a
        # standalone element centred on the *page* (a wrapped title/author
        # line, whose centre sits within `margin` of the page centre), or a
        # section heading centred within its own *column* instead of being
        # left-aligned like body paragraphs (its centre sits well to one
        # side of the page centre even though it doesn't reach a column's
        # left edge). Only the former is truly a full-width row.
        mid = (blk.x0 + blk.x1) / 2
        if mid < center - margin:
            return 0
        if mid > center + margin:
            return 1
        return None

    ordered: list[_RawBlock] = []
    left_pending: list[_RawBlock] = []
    right_pending: list[_RawBlock] = []

    def flush() -> None:
        ordered.extend(sorted(left_pending, key=lambda b: b.y0))
        ordered.extend(sorted(right_pending, key=lambda b: b.y0))
        left_pending.clear()
        right_pending.clear()

    for blk in sorted(blocks, key=lambda b: b.y0):
        col = classify(blk)
        if col is None:
            flush()
            ordered.append(blk)
        elif col == 0:
            left_pending.append(blk)
        else:
            right_pending.append(blk)
    flush()
    return ordered


_WORD_RE = re.compile(r"[A-Za-z]{3,}")

# Symbols that show up in figure/table/equation fragments lifted from a page
# ("1 ≤ 2", "x ∈ R", a plot axis) but never in a real section title. A bare
# "=" / "<" / ">" is only rejected when it sits on its own between spaces --
# that is the shape of an inline (in)equality, not of a hyphen or dash used
# inside ordinary title punctuation.
_MATH_SYMBOL_RE = re.compile(r"[≤≥±×÷≈≠∈∑∏∫√∞→←↔∀∃∇∂⊂⊆∪∩]")
_BARE_MATH_OP_RE = re.compile(r"(?:^|\s)[=<>](?:\s|$)")
# Minor words that stay lowercase in correctly-cased Title Case ("Related
# Work and Future Directions"), so they must not count against a heading
# candidate when checking whether its content words are capitalised.
_TITLE_CASE_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "with",
    "from", "by", "as", "is", "are", "into", "via", "vs", "at", "using",
}


def _upper_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase; 0 if there are none."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _looks_like_numbered_heading_rest(rest: str) -> bool:
    """Guard for the text after a "12.3 " / "4 " numeric prefix.

    Plenty of non-heading text matches "^\\d+(\\.\\d+)* " too: a table row
    ("13 relations"), a plot axis tick plus its label, or a fragment of
    inline math ("1 ≤2 sup"). What distinguishes a real numbered section
    title from those is: it starts with a capital letter (never lowercase
    prose), it never contains a mathematical/relational symbol, it is
    mostly words rather than digits/symbols, and its content words are
    either ALL CAPS (IEEE style) or Title Case -- never a run of lowercase
    words the way ordinary sentence fragments read.
    """
    rest = rest.strip()
    if not rest or not rest[0].isupper():
        return False
    if _MATH_SYMBOL_RE.search(rest) or _BARE_MATH_OP_RE.search(rest):
        return False
    compact = rest.replace(" ", "")
    letters = sum(1 for c in compact if c.isalpha())
    if not compact or letters / len(compact) < 0.5:
        return False  # mostly digits/symbols, e.g. a stray numeric/plot label
    if rest.isupper():
        return True  # IEEE/ACM-style ALL CAPS numbered heading
    significant = [
        w for w in rest.split()
        if len(re.sub(r"[^A-Za-z]", "", w)) >= 4 and w.lower() not in _TITLE_CASE_STOPWORDS
    ]
    if not significant:
        return True  # short heading, no long content word to judge case by
    capitalised = sum(1 for w in significant if w[0].isupper())
    return capitalised / len(significant) >= 0.6


def _looks_like_section_rest(rest: str) -> bool:
    """Guard for the text following an "I. " / "A. " style prefix.

    This is what keeps the roman/letter heading regex from firing on
    reference-list entries and author-initial lists, which have exactly the
    same "X. " shape ("S. Hirche, "Whom to Trust..."", "A. Clements, D. O.
    Joly, ..."): those always carry a comma very early and/or run long,
    neither of which a real section title does.
    """
    rest = rest.strip()
    if not rest or len(rest) > 70:
        return False
    if "," in rest:
        return False
    words = rest.split()
    if not (1 <= len(words) <= 10):
        return False
    return words[0][:1].isupper()


def _font_signals_heading(font: str, body_font: str) -> bool:
    """A span's font can mark it as a heading even at body-or-smaller size.

    Some IEEE-template PDFs render section headings in a *smaller* point
    size than body text but in a distinct medium/bold font (e.g. body is
    10pt "NimbusRomNo9L-Regu", headings are 9pt "NimbusRomNo9L-Medi") --
    a pure size-ratio test can never see these. Any font name that differs
    from the modal body font and carries a weight/emphasis marker is taken
    as an independent heading signal.
    """
    if not font or not body_font or font == body_font:
        return False
    return bool(_HEADING_FONT_RE.search(font))


def _heading_level(
    text: str, size: float, body_size: float, bold: bool,
    font: str = "", body_font: str = "",
) -> int | None:
    # A figure/table axis label or numeric cell ("42.9", "0 +2") is not a
    # heading just because it renders large and bold -- require an actual
    # word so numeric chart furniture doesn't get promoted to a section.
    if not _WORD_RE.search(text):
        return None
    m = _NUMBERED_HEADING_RE.match(text)
    if m:
        prefix, rest = m.group(1), m.group(2)
        parts = prefix.split(".")
        depth = len(parts)
        # A paper does not have a section "87", nor does it nest five levels
        # deep -- a leading number that large/deep is virtually always a
        # figure/table/equation label or a stray numeric value, never a real
        # section number, so reject outright rather than clamp.
        if depth > 4 or int(parts[0]) > 30:
            return None
        if not _looks_like_numbered_heading_rest(rest):
            return None
        return depth

    m_alpha = _ALPHA_HEADING_RE.match(text)
    if m_alpha:
        prefix, rest = m_alpha.group(1), m_alpha.group(2)
        if _looks_like_section_rest(rest):
            is_roman = bool(_ROMAN_RE.match(prefix))
            # IEEE-style top section: roman numeral prefix, rest rendered in
            # (near-)all caps, e.g. "I. INTRODUCTION" / "IV. ALGORITHMS FOR ...".
            if is_roman and _upper_ratio(rest) >= 0.8:
                return 1
            # Lettered subsection: single letter prefix, title-cased rest,
            # e.g. "A. Dataset" / "B. Protocol and Metrics". Excludes the
            # roman+all-caps case above so "D. CONCLUSIONS" (unlikely but
            # possible) still resolves to a top-level section.
            if len(prefix) == 1 and _upper_ratio(rest) < 0.8:
                return 2

    normalized = text.strip().rstrip(":").lower()
    if normalized in _STANDARD_SECTIONS:
        return 1
    if body_size > 0 and size >= body_size * 1.15 and bold:
        ratio = size / body_size
        if ratio >= 1.4:
            return 1
        if ratio >= 1.25:
            return 2
        return 3
    # Distinct heading font (see _font_signals_heading) stands in for the
    # size-ratio test above when the heading is set at body size or smaller.
    # Kept deliberately narrow: short, no comma, and not much smaller than
    # body text -- table headers/legend labels ("Parameter", "Select Window
    # Length") use the same medium-weight font at 6-8pt against a 10pt body,
    # noticeably smaller than a real heading's 9pt, which is what separates
    # them from a genuine section title in this font.
    if (
        _font_signals_heading(font, body_font)
        and len(text) <= 60
        and "," not in text
        and body_size > 0
        and size >= body_size * 0.85
    ):
        return 2
    return None


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    if "\n" in stripped and len(stripped.splitlines()) > 2:
        return False
    if stripped.endswith((".", ",", ";")):
        # numbered/standard headings never end mid-sentence like prose.
        return False
    return True


def _modal_body_size(pages: list[list[_RawBlock]]) -> float:
    counts: Counter[float] = Counter()
    for blocks in pages:
        for b in blocks:
            counts[round(b.size, 1)] += len(b.text)
    if not counts:
        return 10.0
    return counts.most_common(1)[0][0]


def _is_content_free_fragment(text: str) -> bool:
    """A chart axis tick, plot-legend entry, or stray equation piece pulled
    out of an embedded figure ("40", "10 0", "i=1", "ND", "56.4%", "CoT+").

    These get extracted as their own text block like any real sentence, but
    carry no citable content -- embedding and retrieving them just hands a
    user a nonsense quote. Conservative on purpose: real prose almost always
    has a word of 3+ letters, so requiring both "no such word" and "short"
    means it is far more likely to leave a stray label in than to eat a
    genuine (if terse) sentence.
    """
    stripped = text.strip()
    return len(stripped) < 20 and not _WORD_RE.search(stripped)


def _starts_new_heading(text: str) -> bool:
    """True if `text` opens its own numbered/roman/lettered heading.

    Used to stop the heading-merge step (below) from gluing two distinct,
    independently-numbered headings together just because they land at the
    same font size with a small gap -- e.g. "II. BACKGROUND..." followed
    immediately by its own first subsection "A. QLSTM...". A genuine
    wrapped second line of one heading never itself starts with a new
    "<numeral>. " prefix, so this only blocks the cross-heading case.
    """
    return bool(_NUMBERED_HEADING_RE.match(text) or _ALPHA_HEADING_RE.match(text))


def _normalize_match_text(s: str) -> str:
    """Fold to a whitespace- and case-insensitive comparison key, trimming
    the trailing punctuation a heading candidate carries but a title
    substring wouldn't ("...Assessment" vs "...Assessment:")."""
    return re.sub(r"\s+", " ", s.strip()).strip(":.,").lower()


def _is_title_continuation(candidate: str, title: str) -> bool:
    """True if `candidate` is a wrapped second/third line of the paper's
    own title, mis-split into its own block.

    A long arXiv title routinely wraps across 2-3 large-font lines before
    the author list; PyMuPDF sees the same visual gap between those lines
    as it does between a heading and its body text, so each wrapped line
    looks exactly like a heading (large, bold, short) even though it is
    not one. Left alone, it becomes a nonsensical `section_path[0]` in
    every citation built from that block ("... | and Risk Assessment | p.
    7"). The manifest/PDF-metadata title (`_resolve_title`) is taken as
    complete and correct, so any genuine wrapped fragment must appear
    verbatim inside it; a real section heading essentially never will.
    """
    cand = _normalize_match_text(candidate)
    full = _normalize_match_text(title)
    if len(cand) < 4:
        return False  # too short to be a meaningful title-substring test
    if cand in _STANDARD_SECTIONS:
        return False  # never swallow "abstract"/"introduction" etc.
    return cand in full


def _modal_body_font(pages: list[list[_RawBlock]]) -> str:
    counts: Counter[str] = Counter()
    for blocks in pages:
        for b in blocks:
            if b.font:
                counts[b.font] += len(b.text)
    return counts.most_common(1)[0][0] if counts else ""


class PDFParser:
    """Parses arXiv-style PDFs into reading-order `Block`s.

    Constructed with the resolved `IngestConfig` so header/footer stripping,
    margin ratios and reference-section handling follow project config
    rather than being hardcoded.
    """

    name = "pdf"

    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
        self._manifest_cache: dict[str, dict[str, Any]] | None = None

    def supports(self, source: str) -> bool:
        return source.lower().endswith(".pdf")

    # -- manifest lookup --------------------------------------------------

    def _manifest(self) -> dict[str, dict[str, Any]]:
        """Lazily load+cache data/processed/corpus_manifest.json, keyed by
        arxiv_id (== filename stem for this corpus)."""
        if self._manifest_cache is not None:
            return self._manifest_cache
        cache: dict[str, dict[str, Any]] = {}
        candidates = [
            Path("data/processed/corpus_manifest.json"),
            Path(__file__).resolve().parents[3] / "data" / "processed" / "corpus_manifest.json",
        ]
        for path in candidates:
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    for paper in data.get("papers", []):
                        aid = paper.get("arxiv_id")
                        if aid:
                            cache[aid] = paper
                except (json.JSONDecodeError, OSError):
                    pass
                break
        self._manifest_cache = cache
        return cache

    def _resolve_title(
        self, stem: str, pdf_meta_title: str | None, page1_blocks: list[_RawBlock]
    ) -> tuple[str, dict[str, Any]]:
        manifest = self._manifest()
        entry = manifest.get(stem)
        metadata: dict[str, Any] = {}
        if entry:
            if entry.get("arxiv_id"):
                metadata["arxiv_id"] = entry["arxiv_id"]
            if entry.get("authors"):
                metadata["authors"] = entry["authors"]
            if entry.get("categories"):
                metadata["categories"] = entry["categories"]
            if entry.get("title"):
                return entry["title"].strip(), metadata

        if pdf_meta_title:
            t = pdf_meta_title.strip()
            junk = not t or len(t) < 3 or t.lower().endswith((".pdf", ".doc", ".docx")) or t.isdigit()
            if not junk:
                return t, metadata

        if page1_blocks:
            best = max(page1_blocks, key=lambda b: b.size)
            candidate = normalize_whitespace(best.text)
            if candidate:
                return candidate, metadata

        return stem, metadata

    # -- main entry point ---------------------------------------------------

    def parse(self, source: str) -> ParsedDocument:
        path = Path(source)
        doc_id = SourceDocument.make_doc_id(str(path))

        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:  # pragma: no cover - pymupdf raises many types
            raise PDFParseError(f"cannot open PDF {path}: {exc}") from exc

        try:
            if doc.is_encrypted and not doc.authenticate(""):
                document = SourceDocument(
                    doc_id=doc_id, title=path.stem, source_type=SourceType.PDF,
                    source_path=str(path), page_count=doc.page_count,
                )
                return ParsedDocument(
                    document=document,
                    warnings=[f"encrypted PDF, could not decrypt: {path.name}"],
                )

            page_count = doc.page_count
            if page_count == 0:
                document = SourceDocument(
                    doc_id=doc_id, title=path.stem, source_type=SourceType.PDF,
                    source_path=str(path), page_count=0,
                )
                return ParsedDocument(document=document, warnings=["PDF has zero pages"])

            try:
                pages_raw: list[list[_RawBlock]] = [
                    _extract_raw_blocks(doc[i], i + 1) for i in range(page_count)
                ]
            except Exception as exc:  # pragma: no cover
                raise PDFParseError(f"failed extracting text from {path}: {exc}") from exc

            total_chars = sum(len(b.text) for blocks in pages_raw for b in blocks)
            warnings: list[str] = []

            page1_blocks = pages_raw[0] if pages_raw else []
            pdf_meta_title = (doc.metadata or {}).get("title")
            title, meta_from_manifest = self._resolve_title(path.stem, pdf_meta_title, page1_blocks)

            if total_chars == 0:
                document = SourceDocument(
                    doc_id=doc_id, title=title, source_type=SourceType.PDF,
                    source_path=str(path), page_count=page_count,
                    metadata=meta_from_manifest,
                )
                return ParsedDocument(
                    document=document,
                    warnings=["no extractable text (likely an image-only scan)"],
                )

            page_width = doc[0].rect.width
            page_height = doc[0].rect.height

            drop_keys: set[tuple[int, tuple[float, float, float, float]]] = set()
            dropped: dict[str, list[str]] = {"headers": [], "footers": [], "page_numbers": []}
            if self.cfg.strip_pdf_headers_footers:
                drop_keys, dropped = _collect_furniture(
                    pages_raw, page_height, self.cfg.header_footer_margin_ratio,
                    self.cfg.min_repeat_ratio,
                )
            dropped["fragments"] = []

            body_size = _modal_body_size(pages_raw)
            body_font = _modal_body_font(pages_raw)

            blocks: list[Block] = []
            section_stack: list[tuple[int, str]] = []
            in_references = False
            order = 0
            # True until the first "real" heading (Abstract, Introduction,
            # ...) is seen on page 1. While true, any heading-shaped block
            # that is actually a wrapped continuation of the title is
            # demoted back to plain text instead of becoming a bogus
            # section (see `_is_title_continuation`).
            title_phase = True

            for page_no, raw_blocks in enumerate(pages_raw, start=1):
                kept = [b for b in raw_blocks if (b.page, b.bbox) not in drop_keys]
                ordered = _order_page_blocks(kept, page_width)

                # Merge obviously-continued blocks split across extraction
                # blocks: either a body paragraph wrapped mid-sentence, or a
                # heading (e.g. a wrapped title) split across two lines.
                def is_heading_candidate(b: _RawBlock) -> bool:
                    return _looks_like_heading(b.text) and _heading_level(
                        b.text, b.size, body_size, b.bold, b.font, body_font
                    ) is not None

                merged: list[_RawBlock] = []
                merged_is_heading: list[bool] = []
                for blk in ordered:
                    cur_is_heading = is_heading_candidate(blk)
                    # Gap must be small AND non-negative: column-aware ordering
                    # flushes the whole left column before the right one, so a
                    # right-column block's y0 can sit well *above* the previous
                    # (left-column) block's y1 -- a large negative "gap" that
                    # used to satisfy `< threshold` and wrongly glue unrelated
                    # headings from different columns together.
                    gap = (blk.y0 - merged[-1].y1) if merged else None
                    same_column = merged and abs(blk.x0 - merged[-1].x0) < 0.05 * page_width
                    close_gap = bool(
                        merged and gap is not None and 0 <= gap < 0.03 * page_height * 4 and same_column
                    )
                    same_size = abs(merged[-1].size - blk.size) < 0.6 if merged else False
                    merge_as_heading = (
                        merged
                        and merged_is_heading[-1]
                        and cur_is_heading
                        and same_size
                        and close_gap
                        and not _starts_new_heading(blk.text)
                    )
                    merge_as_paragraph = (
                        merged
                        and not cur_is_heading
                        and not merged_is_heading[-1]
                        and same_size
                        and not merged[-1].text.rstrip().endswith((".", "!", "?", ":"))
                        and close_gap
                    )
                    if merge_as_heading or merge_as_paragraph:
                        prev = merged[-1]
                        sep = " " if merge_as_heading else "\n"
                        merged[-1] = _RawBlock(
                            prev.page, prev.bbox, prev.text + sep + blk.text,
                            max(prev.size, blk.size), prev.bold, prev.n_lines + blk.n_lines,
                            prev.font,
                        )
                    else:
                        merged.append(blk)
                        merged_is_heading.append(cur_is_heading)

                for blk in merged:
                    raw_text = blk.text
                    is_heading = False
                    level: int | None = None
                    if _looks_like_heading(raw_text):
                        level = _heading_level(
                            raw_text, blk.size, body_size, blk.bold, blk.font, body_font
                        )
                        is_heading = level is not None

                    if is_heading and page_no == 1 and title_phase:
                        if _is_title_continuation(raw_text, title):
                            is_heading = False
                            level = None
                        else:
                            title_phase = False

                    kind: BlockKind = "paragraph"
                    if is_heading:
                        kind = "heading"
                    elif in_references:
                        kind = "footnote"
                    elif _CAPTION_RE.match(raw_text):
                        kind = "caption"

                    text = normalize_block_text(raw_text, kind)
                    if not text:
                        continue

                    if kind not in ("heading", "caption", "table") and _is_content_free_fragment(text):
                        dropped["fragments"].append(text)
                        continue

                    if is_heading:
                        heading_title = text.rstrip(":")
                        while section_stack and section_stack[-1][0] >= level:
                            section_stack.pop()
                        section_path = [t for _, t in section_stack]
                        section_stack.append((level, heading_title))
                        if heading_title.lower() in _REFERENCE_HEADINGS or (
                            (m := _NUMBERED_HEADING_RE.match(heading_title))
                            and m.group(2).strip().lower() in _REFERENCE_HEADINGS
                        ):
                            in_references = True
                        elif heading_title.lower().lstrip("0123456789. ") in _REFERENCE_HEADINGS:
                            in_references = True
                    else:
                        section_path = [t for _, t in section_stack]

                    if in_references and self.cfg.drop_references_section and not is_heading:
                        continue
                    if in_references and self.cfg.drop_references_section and is_heading and (
                        heading_title.lower() in _REFERENCE_HEADINGS
                    ):
                        # The References heading itself is also noise once
                        # its whole section is being dropped.
                        continue

                    blocks.append(
                        Block(
                            text=text,
                            kind=kind,
                            page=page_no,
                            section_path=section_path,
                            heading_level=level if is_heading else None,
                            order=order,
                        )
                    )
                    order += 1

            # char_start/char_end must match how ParsedDocument.text joins.
            pos = 0
            for b in blocks:
                b.char_start = pos
                b.char_end = pos + len(b.text)
                pos = b.char_end + 2  # "\n\n" separator

            full_text = "\n\n".join(b.text for b in blocks)
            if len(full_text) < self.cfg.min_chars_per_doc:
                warnings.append(
                    f"extracted text below min_chars_per_doc "
                    f"({len(full_text)} < {self.cfg.min_chars_per_doc})"
                )

            document = SourceDocument(
                doc_id=doc_id,
                title=title,
                source_type=SourceType.PDF,
                source_path=str(path),
                text=full_text,
                page_count=page_count,
                metadata=meta_from_manifest,
            )
            return ParsedDocument(document=document, blocks=blocks, dropped=dropped, warnings=warnings)
        finally:
            doc.close()

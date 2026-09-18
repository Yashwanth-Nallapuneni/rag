"""Token-window chunking with sentence-aligned overlap.

Two decisions here do most of the work:

**Sentences are the atomic unit, not characters or raw blocks.** Packing whole
sentences means a chunk boundary never lands mid-sentence, so the overlap
region is always readable prose rather than two half-sentences.

**Overlap is carried back as whole trailing sentences.** The spec's rationale
is that a claim sliced across a boundary loses the context that supports it
and stops being retrievable. Repeating the last ~100 tokens of chunk N at the
start of chunk N+1 means any sentence near a boundary appears intact, with its
neighbours, in at least one chunk.

Each sentence keeps the page and heading stack of the block it came from, so a
chunk's citation locator is derived from real provenance rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import ChunkingConfig
from ..ingest.base import Block, ParsedDocument
from ..schemas import Chunk
from ..tokenization import get_encoder

# Split on sentence-ending punctuation followed by whitespace and a capital or
# opening bracket. Guarded against the abbreviations and citation markers that
# litter academic prose -- splitting on "et al." or "Fig. 3" would shred
# sentences and make the overlap meaningless.
_ABBREV = (
    r"(?<!\be\.g)(?<!\bi\.e)(?<!\bet\sal)(?<!\bcf)(?<!\bvs)(?<!\bFig)(?<!\bfig)"
    r"(?<!\bEq)(?<!\beq)(?<!\bSec)(?<!\bsec)(?<!\bTab)(?<!\bRef)(?<!\bApp)"
    r"(?<!\bNo)(?<!\bpp)(?<!\bal)(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bProf)"
    r"(?<!\bApprox)(?<!\bResp)(?<!\bw\.r\.t)(?<!\bs\.t)"
)
_SENT_SPLIT = re.compile(rf"{_ABBREV}(?<=[.!?])[\"')\]]?\s+(?=[A-Z(\[\"'])")

# Blocks that must never be split: breaking them mid-way produces fragments
# that are worse than useless for both retrieval and display.
_ATOMIC_KINDS = {"code", "table", "formula"}


@dataclass
class _Unit:
    """One sentence (or atomic block) with the provenance it inherited."""

    text: str
    tokens: int
    page: int | None
    section_path: list[str]
    char_start: int | None
    char_end: int | None
    kind: str


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _hard_split(text: str, tokens: list[int], size: int, encoder) -> list[str]:
    """Last resort for a single sentence larger than a whole chunk."""
    out: list[str] = []
    for i in range(0, len(tokens), size):
        try:
            out.append(encoder.decode(tokens[i : i + size]))
        except NotImplementedError:
            step = size * 4
            out = [text[j : j + step] for j in range(0, len(text), step)]
            break
    return [s for s in out if s.strip()]


def _blocks_to_units(blocks: list[Block], cfg: ChunkingConfig) -> list[_Unit]:
    encoder = get_encoder(cfg.tokenizer)
    units: list[_Unit] = []
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        pieces = [text] if block.kind in _ATOMIC_KINDS else split_sentences(text)
        cursor = block.char_start
        for piece in pieces:
            tokens = encoder.encode(piece)
            span = len(piece)
            start = cursor
            cursor = (cursor + span + 1) if cursor is not None else None
            if len(tokens) > cfg.chunk_size:
                # Oversized single unit: split it rather than blow the budget.
                for frag in _hard_split(piece, tokens, cfg.chunk_size, encoder):
                    units.append(
                        _Unit(
                            frag,
                            len(encoder.encode(frag)),
                            block.page,
                            list(block.section_path),
                            start,
                            start + len(frag) if start is not None else None,
                            block.kind,
                        )
                    )
                continue
            units.append(
                _Unit(
                    piece,
                    len(tokens),
                    block.page,
                    list(block.section_path),
                    start,
                    (start + span) if start is not None else None,
                    block.kind,
                )
            )
    return units


def _top_section(path: list[str]) -> str:
    return path[0] if path else ""


def _emit(
    units: list[_Unit],
    parsed: ParsedDocument,
    index: int,
    cfg: ChunkingConfig,
) -> Chunk | None:
    if not units:
        return None
    text = " ".join(u.text for u in units).strip()
    if not text:
        return None
    encoder = get_encoder(cfg.tokenizer)
    pages = [u.page for u in units if u.page is not None]
    starts = [u.char_start for u in units if u.char_start is not None]
    ends = [u.char_end for u in units if u.char_end is not None]
    doc = parsed.document
    # The first unit's heading stack names the chunk: it is where the chunk
    # begins, which is what a reader following the citation will look for.
    section_path = next((u.section_path for u in units if u.section_path), [])
    return Chunk(
        chunk_id=Chunk.make_chunk_id(doc.doc_id, index),
        doc_id=doc.doc_id,
        doc_title=doc.title,
        chunk_index=index,
        text=text,
        token_count=len(encoder.encode(text)),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        section_path=list(section_path),
        char_start=min(starts) if starts else None,
        char_end=max(ends) if ends else None,
        source_type=doc.source_type,
        source_path=doc.source_path,
        source_uri=doc.source_uri,
        metadata=dict(doc.metadata),
    )


def chunk_document(parsed: ParsedDocument, cfg: ChunkingConfig) -> list[Chunk]:
    """Pack a parsed document into overlapping token windows."""
    units = _blocks_to_units(parsed.body_blocks(), cfg)
    if not units:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    n = len(units)

    while start < n:
        total = 0
        end = start
        while end < n:
            unit = units[end]
            if total + unit.tokens > cfg.chunk_size and end > start:
                break
            # heading_aware never merges across sections; token_window only
            # breaks early once the chunk is substantial enough to stand alone.
            if end > start:
                crossed = _top_section(unit.section_path) != _top_section(
                    units[start].section_path
                )
                if crossed and (
                    cfg.strategy == "heading_aware"
                    or (cfg.respect_headings and total >= cfg.chunk_size * 0.5)
                ):
                    break
            total += unit.tokens
            end += 1

        window = units[start:end]
        chunk = _emit(window, parsed, index, cfg)
        # Keep an undersized tail only if it is the document's single chunk;
        # otherwise it is a sliver whose content already appears in the
        # previous chunk's overlap region.
        if chunk and (chunk.token_count >= cfg.min_chunk_tokens or not chunks):
            chunks.append(chunk)
            index += 1

        if end >= n:
            break

        # Carry back whole trailing sentences worth ~chunk_overlap tokens.
        carried = 0
        back = end
        while back > start + 1 and carried < cfg.chunk_overlap:
            carried += units[back - 1].tokens
            back -= 1
        # Guarantee forward progress: never restart at or before this window's
        # own start, or the loop stalls and emits the same chunk forever.
        start = max(back, start + 1)

    return chunks


def chunk_documents(
    parsed_docs: list[ParsedDocument], cfg: ChunkingConfig
) -> list[Chunk]:
    out: list[Chunk] = []
    for parsed in parsed_docs:
        out.extend(chunk_document(parsed, cfg))
    return out

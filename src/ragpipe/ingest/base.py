"""The intermediate representation between parsing and chunking.

Parsers do not emit plain text. They emit an ordered list of `Block`s, each
carrying the page it came from and the heading stack it sits under. The
chunker then packs blocks into token windows and inherits their provenance.

That indirection is the whole reason a citation can say "p. 4, section 3.2
Attention" instead of just naming a file: page and section are attached at
parse time, where the information actually exists, rather than reconstructed
later from a flat string where it has already been lost.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ..schemas import SourceDocument

BlockKind = Literal[
    "heading", "paragraph", "list", "table", "caption", "code", "formula", "footnote"
]


class Block(BaseModel):
    """One contiguous span of text with its provenance."""

    text: str
    kind: BlockKind = "paragraph"
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    heading_level: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    order: int = 0

    @property
    def is_heading(self) -> bool:
        return self.kind == "heading"


class ParsedDocument(BaseModel):
    """A parsed source: its metadata plus its blocks in reading order."""

    document: SourceDocument
    blocks: list[Block] = Field(default_factory=list)
    # What the parser threw away, kept for inspection: stripped running
    # headers/footers, dropped page numbers, skipped pages. Silent removal is
    # how ingestion bugs hide, so the evidence is retained.
    dropped: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    @property
    def page_count(self) -> int:
        pages = {b.page for b in self.blocks if b.page is not None}
        return len(pages)

    def body_blocks(self) -> list[Block]:
        """Blocks a chunk should be built from -- headings carry structure but
        are already represented in every block's section_path."""
        return [b for b in self.blocks if not b.is_heading and b.text.strip()]


@runtime_checkable
class Parser(Protocol):
    """Every source type implements this."""

    name: str

    def supports(self, source: str) -> bool: ...

    def parse(self, source: str) -> ParsedDocument: ...


# Kinds whose line structure is meaningful: collapsing their newlines turns a
# code listing or a table into one unreadable line.
LINE_PRESERVING_KINDS = {"code", "table", "list", "formula"}


def normalize_preserving_lines(text: str) -> str:
    """Tidy each line but keep the line breaks.

    Used for code, tables and lists, where `normalize_whitespace`'s newline
    collapsing would destroy the structure a reader needs to make sense of the
    cited passage.
    """
    import re

    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).rstrip() for ln in text.splitlines()]
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def normalize_code(text: str) -> str:
    """Only trim trailing whitespace and blank runs.

    Leading whitespace is left alone because in Python -- and in any
    indentation-sensitive snippet -- it is semantics, not formatting.
    """
    import re

    lines = [ln.replace("\u00a0", " ").rstrip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")


def normalize_block_text(text: str, kind: str) -> str:
    """Normalise according to whether the block's line structure matters."""
    if kind == "code":
        return normalize_code(text)
    if kind in LINE_PRESERVING_KINDS:
        return normalize_preserving_lines(text)
    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace but keep paragraph structure readable.

    PDF extraction is full of hard-wrapped lines and stray hyphenation;
    normalising here keeps chunk token counts honest and stops the same
    sentence from looking different to BM25 and to the embedder.
    """
    import re

    # Join words broken across a line by hyphenation ("atten-\ntion").
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # Single newlines inside a paragraph become spaces; blank lines survive.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

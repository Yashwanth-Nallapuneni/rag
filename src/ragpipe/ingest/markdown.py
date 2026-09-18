"""Markdown parser.

Markdown carries no page numbers, so provenance here comes entirely from the
heading stack (`section_path`) and character offsets. The tricky part is
never treating fenced-code content as structure: a `#` or `---` inside a
``` block is text, not a heading, and getting that wrong shreds the section
hierarchy of any technical document (code blocks are exactly where `#`
comments and `---` separators show up).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..config import IngestConfig
from ..schemas import SourceDocument, SourceType
from .base import Block, ParsedDocument, normalize_block_text, normalize_whitespace

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT_UNDERLINE_RE = re.compile(r"^(=+|-+)\s*$")
_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_INDENTED_CODE_RE = re.compile(r"^(?: {4}|\t)")

# Inline syntax stripped from block text so embedder/BM25 see clean prose.
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_BOLD_ITALIC_RE = re.compile(r"(\*\*\*|___)(.+?)\1")
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC_RE = re.compile(r"(\*|_)(.+?)\1")


def _strip_inline(text: str) -> str:
    """Remove Markdown inline markup, keeping the underlying prose intact."""
    text = _IMAGE_RE.sub("", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _BOLD_ITALIC_RE.sub(r"\2", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _ITALIC_RE.sub(r"\2", text)
    return text


def _split_front_matter(text: str) -> tuple[dict[str, Any], str, int]:
    """Pull a leading `---`/`---` YAML block out. Returns (metadata, rest, offset)."""
    if not text.startswith("---"):
        return {}, text, 0
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return {}, text, 0
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            raw = "\n".join(lines[1:i])
            try:
                meta = yaml.safe_load(raw) or {}
            except yaml.YAMLError:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            rest_start = sum(len(l) + 1 for l in lines[: i + 1])
            return meta, text[rest_start:], rest_start
    return {}, text, 0


class _Line:
    __slots__ = ("text", "start", "end")

    def __init__(self, text: str, start: int, end: int):
        self.text = text
        self.start = start
        self.end = end


def _iter_lines(text: str, base_offset: int) -> list[_Line]:
    lines: list[_Line] = []
    pos = base_offset
    for raw in text.split("\n"):
        lines.append(_Line(raw, pos, pos + len(raw)))
        pos += len(raw) + 1
    return lines


class MarkdownParser:
    """Parses .md/.markdown/.mdx files into heading-scoped `Block`s."""

    name = "markdown"

    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg

    def supports(self, source: str) -> bool:
        return Path(source).suffix.lower() in {".md", ".markdown", ".mdx"}

    def parse(self, source: str) -> ParsedDocument:
        path = Path(source)
        raw = path.read_text(encoding="utf-8")
        metadata, body, offset = _split_front_matter(raw)

        lines = _iter_lines(body, offset)
        blocks: list[Block] = []
        stack: list[tuple[int, str]] = []  # (level, heading text)
        order = 0
        warnings: list[str] = []

        i = 0
        n = len(lines)
        para_buf: list[_Line] = []
        first_h1: str | None = None

        def flush_paragraph() -> None:
            nonlocal order
            if not para_buf:
                return
            text = "\n".join(l.text for l in para_buf)
            kind = "list" if _LIST_ITEM_RE.match(para_buf[0].text.strip("\t ") or "") else "paragraph"
            # A list group is one where every non-blank line looks like an item
            # or a continuation; treat as list if the first line does.
            cleaned = normalize_block_text(_strip_inline(text), kind)
            if cleaned:
                blocks.append(
                    Block(
                        text=cleaned,
                        kind=kind,
                        page=None,
                        section_path=[h for _, h in stack],
                        heading_level=None,
                        char_start=para_buf[0].start,
                        char_end=para_buf[-1].end,
                        order=order,
                    )
                )
                order += 1
            para_buf.clear()

        while i < n:
            line = lines[i]
            stripped = line.text.strip()

            # Fenced code block: consume verbatim until the matching fence.
            fence_m = _FENCE_RE.match(line.text)
            if fence_m:
                flush_paragraph()
                fence_char = fence_m.group(2)[0]
                fence_len = len(fence_m.group(2))
                code_lines = [line]
                j = i + 1
                while j < n:
                    close_m = _FENCE_RE.match(lines[j].text)
                    if (
                        close_m
                        and close_m.group(2)[0] == fence_char
                        and len(close_m.group(2)) >= fence_len
                        and not close_m.group(3).strip()
                    ):
                        code_lines.append(lines[j])
                        j += 1
                        break
                    code_lines.append(lines[j])
                    j += 1
                else:
                    warnings.append("Unterminated fenced code block")
                text = "\n".join(l.text for l in code_lines)
                blocks.append(
                    Block(
                        text=text,
                        kind="code",
                        page=None,
                        section_path=[h for _, h in stack],
                        heading_level=None,
                        char_start=code_lines[0].start,
                        char_end=code_lines[-1].end,
                        order=order,
                    )
                )
                order += 1
                i = j
                continue

            # Indented code block (4 spaces / tab), only outside a paragraph.
            if _INDENTED_CODE_RE.match(line.text) and not para_buf and stripped:
                flush_paragraph()
                code_lines = [line]
                j = i + 1
                while j < n and (_INDENTED_CODE_RE.match(lines[j].text) or not lines[j].text.strip()):
                    if not lines[j].text.strip() and j + 1 < n and not _INDENTED_CODE_RE.match(lines[j + 1].text):
                        break
                    code_lines.append(lines[j])
                    j += 1
                while code_lines and not code_lines[-1].text.strip():
                    code_lines.pop()
                text = "\n".join(re.sub(r"^(    |\t)", "", l.text) for l in code_lines)
                blocks.append(
                    Block(
                        text=text,
                        kind="code",
                        page=None,
                        section_path=[h for _, h in stack],
                        heading_level=None,
                        char_start=code_lines[0].start,
                        char_end=code_lines[-1].end,
                        order=order,
                    )
                )
                order += 1
                i = j
                continue

            # Blank line: paragraph boundary.
            if not stripped:
                flush_paragraph()
                i += 1
                continue

            # ATX heading.
            atx_m = _ATX_RE.match(line.text)
            if atx_m:
                flush_paragraph()
                level = len(atx_m.group(1))
                heading_text = normalize_whitespace(_strip_inline(atx_m.group(2)))
                if level == 1 and first_h1 is None:
                    first_h1 = heading_text
                stack = [(lv, h) for lv, h in stack if lv < level]
                blocks.append(
                    Block(
                        text=heading_text,
                        kind="heading",
                        page=None,
                        section_path=[h for _, h in stack],
                        heading_level=level,
                        char_start=line.start,
                        char_end=line.end,
                        order=order,
                    )
                )
                order += 1
                stack.append((level, heading_text))
                i += 1
                continue

            # Setext heading: current line is text, next non-blank line is
            # all `=` (level 1) or all `-` (level 2), and current para is
            # otherwise a single line (not a table separator context).
            if (
                i + 1 < n
                and not para_buf
                and _SETEXT_UNDERLINE_RE.match(lines[i + 1].text.strip())
                and stripped
                and not _LIST_ITEM_RE.match(stripped)
            ):
                underline = lines[i + 1].text.strip()
                level = 1 if underline[0] == "=" else 2
                heading_text = normalize_whitespace(_strip_inline(stripped))
                if level == 1 and first_h1 is None:
                    first_h1 = heading_text
                stack = [(lv, h) for lv, h in stack if lv < level]
                blocks.append(
                    Block(
                        text=heading_text,
                        kind="heading",
                        page=None,
                        section_path=[h for _, h in stack],
                        heading_level=level,
                        char_start=line.start,
                        char_end=lines[i + 1].end,
                        order=order,
                    )
                )
                order += 1
                stack.append((level, heading_text))
                i += 2
                continue

            # Table: a header row followed by a separator row of dashes/pipes.
            if (
                "|" in stripped
                and not para_buf
                and i + 1 < n
                and _TABLE_SEP_RE.match(lines[i + 1].text)
            ):
                flush_paragraph()
                table_lines = [line, lines[i + 1]]
                j = i + 2
                while j < n and lines[j].text.strip() and "|" in lines[j].text:
                    table_lines.append(lines[j])
                    j += 1
                text = "\n".join(l.text.strip() for l in table_lines)
                blocks.append(
                    Block(
                        text=normalize_block_text(text, "table"),
                        kind="table",
                        page=None,
                        section_path=[h for _, h in stack],
                        heading_level=None,
                        char_start=table_lines[0].start,
                        char_end=table_lines[-1].end,
                        order=order,
                    )
                )
                order += 1
                i = j
                continue

            # Otherwise: accumulate into the current paragraph/list buffer.
            para_buf.append(line)
            i += 1

        flush_paragraph()

        title = metadata.get("title") or first_h1 or path.stem
        full_text = normalize_whitespace(_strip_inline(body))
        char_count = sum(len(b.text) for b in blocks) or len(full_text)
        if char_count < self.cfg.min_chars_per_doc:
            warnings.append(
                f"Document has only {char_count} chars of content "
                f"(< min_chars_per_doc={self.cfg.min_chars_per_doc})"
            )

        document = SourceDocument(
            doc_id=SourceDocument.make_doc_id(str(path.resolve())),
            title=str(title),
            source_type=SourceType.MARKDOWN,
            source_path=str(path),
            text=full_text,
            metadata=metadata,
        )
        return ParsedDocument(document=document, blocks=blocks, warnings=warnings)

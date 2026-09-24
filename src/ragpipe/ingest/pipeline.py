"""Source routing and the parse-then-chunk pipeline.

Parsers are imported lazily and selected by source shape, so adding a format
means adding a parser and one routing entry -- nothing downstream changes,
because everything speaks `ParsedDocument` and then `Chunk`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..chunking.chunker import chunk_document
from ..config import Settings
from ..logging_utils import get_logger
from ..schemas import Chunk
from .base import ParsedDocument, Parser

log = get_logger(__name__)

PDF_SUFFIXES = {".pdf"}
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx"}
HTML_SUFFIXES = {".html", ".htm", ".xhtml"}
TEXT_SUFFIXES = {".txt"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | MARKDOWN_SUFFIXES | HTML_SUFFIXES | TEXT_SUFFIXES


class UnsupportedSourceError(ValueError):
    pass


def get_parser(source: str, settings: Settings) -> Parser:
    cfg = settings.ingest
    lowered = source.lower()

    if lowered.startswith(("http://", "https://")):
        from .html import HTMLParser

        return HTMLParser(cfg)

    suffix = Path(source).suffix.lower()
    if suffix in PDF_SUFFIXES:
        from .pdf import PDFParser

        return PDFParser(cfg)
    if suffix in MARKDOWN_SUFFIXES or suffix in TEXT_SUFFIXES:
        from .markdown import MarkdownParser

        return MarkdownParser(cfg)
    if suffix in HTML_SUFFIXES:
        from .html import HTMLParser

        return HTMLParser(cfg)

    raise UnsupportedSourceError(
        f"no parser for {source!r}; supported: {sorted(SUPPORTED_SUFFIXES)} or an http(s) URL"
    )


def discover_sources(root: Path) -> list[str]:
    """Every parseable file under `root`, in a stable order."""
    if root.is_file():
        return [str(root)]
    return [
        str(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def parse_source(source: str, settings: Settings) -> ParsedDocument:
    parsed = get_parser(source, settings).parse(source)
    for warning in parsed.warnings:
        log.warning("%s: %s", source, warning)
    return parsed


def parse_sources(
    sources: Iterable[str], settings: Settings, *, skip_failures: bool = True
) -> tuple[list[ParsedDocument], list[tuple[str, str]]]:
    """Parse many sources. Failures are collected, not fatal: one malformed PDF
    should not abandon a 40-document corpus."""
    parsed: list[ParsedDocument] = []
    failures: list[tuple[str, str]] = []
    for source in sources:
        try:
            parsed.append(parse_source(source, settings))
        except Exception as exc:  # noqa: BLE001 - parser backends vary
            if not skip_failures:
                raise
            log.error("failed to parse %s: %s", source, exc)
            failures.append((source, repr(exc)))
    return parsed, failures


def build_chunks(
    parsed_docs: Iterable[ParsedDocument], settings: Settings
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for parsed in parsed_docs:
        produced = chunk_document(parsed, settings.chunking)
        if not produced:
            log.warning("no chunks produced for %s", parsed.document.title)
        chunks.extend(produced)
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> Path:
    """Persist chunks as JSONL so indexing, BM25 and eval all read the same
    artifact instead of re-parsing PDFs independently and drifting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(chunk.model_dump_json() + "\n")
    return path


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as fh:
        return [Chunk.model_validate_json(line) for line in fh if line.strip()]


def ingest_corpus(
    settings: Settings,
    sources: list[str] | None = None,
    *,
    write: bool = True,
) -> tuple[list[Chunk], dict]:
    """Parse and chunk the whole corpus, returning chunks plus a report."""
    srcs = sources or discover_sources(settings.corpus.raw_path)
    parsed, failures = parse_sources(srcs, settings)
    chunks = build_chunks(parsed, settings)

    token_counts = [c.token_count for c in chunks] or [0]
    report = {
        "config_fingerprint": settings.fingerprint(),
        "sources": len(srcs),
        "parsed": len(parsed),
        "failed": failures,
        "documents": len({c.doc_id for c in chunks}),
        "chunks": len(chunks),
        "chunk_tokens": {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": round(sum(token_counts) / len(token_counts), 1),
        },
        "chunks_with_page": sum(1 for c in chunks if c.page_start is not None),
        "chunks_with_section": sum(1 for c in chunks if c.section_path),
        "chunking": {
            "size": settings.chunking.chunk_size,
            "overlap": settings.chunking.chunk_overlap,
            "strategy": settings.chunking.strategy,
        },
    }

    if write and chunks:
        out = settings.corpus.processed_path / "chunks.jsonl"
        write_chunks(chunks, out)
        report["output"] = str(out.relative_to(settings.project_root))
        (settings.corpus.processed_path / "ingest_report.json").write_text(
            json.dumps(report, indent=2)
        )
    return chunks, report

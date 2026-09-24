"""HTML / web page parser.

The same code path handles a local .html file and a live URL: the only
difference is how the raw bytes are obtained and which `SourceType` gets
stamped on the result. Chrome (nav, ads, related-links rails) outweighs
real content on most pages, so it is stripped before extraction rather than
filtered out later where it would already be mixed into paragraph text.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from ..config import IngestConfig
from ..schemas import SourceDocument, SourceType
from .base import Block, ParsedDocument, normalize_block_text, normalize_whitespace

_USER_AGENT = "ragpipe-ingest/0.1 (+https://github.com/ragpipe; contact: ingest@ragpipe.local)"

_CHROME_TAGS = {"script", "style", "noscript", "nav", "header", "footer",
                "aside", "form", "iframe", "svg"}
_CHROME_HINTS = (
    "nav", "menu", "sidebar", "banner", "cookie", "advert", "breadcrumb",
    "footer", "social", "related", "comment",
)
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class HTMLFetchError(RuntimeError):
    """Raised when a URL cannot be fetched as a parseable document."""


def _looks_like_chrome(tag: Tag) -> bool:
    haystack = " ".join(
        [
            *(tag.get("class") or []),
            tag.get("id") or "",
            tag.get("role") or "",
        ]
    ).lower()
    return any(hint in haystack for hint in _CHROME_HINTS)


def _strip_chrome(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(list(_CHROME_TAGS)):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": "navigation"}):
        tag.decompose()
    # A second pass: class/id hints, but never inside <main>/<article> since
    # those are the content root we are about to prefer.
    for tag in soup.find_all(True):
        if tag.name in {"main", "article", "html", "body"}:
            continue
        if not tag.parent:
            continue
        if _looks_like_chrome(tag):
            tag.decompose()


def _pick_root(soup: BeautifulSoup) -> Tag:
    for selector in ("main", "article", "[role=main]"):
        found = soup.select_one(selector)
        if found is not None:
            return found
    body = soup.body or soup
    candidates = body.find_all("div", recursive=True)
    if not candidates:
        return body
    densest = max(candidates, key=lambda d: len(d.get_text(strip=True)), default=None)
    if densest is not None and len(densest.get_text(strip=True)) > len(body.get_text(strip=True)) * 0.3:
        return densest
    return body


def _clean_text(node: Tag) -> str:
    return normalize_whitespace(node.get_text(" ", strip=True))


def _flatten_table(table: Tag) -> str:
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [normalize_whitespace(c.get_text(" ", strip=True)) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _is_pure_punct(text: str) -> bool:
    return bool(text) and not re.search(r"[A-Za-z0-9]", text)


class HTMLParser:
    """Parses local .html/.htm files and http(s) URLs into `Block`s."""

    name = "html"

    def __init__(self, cfg: IngestConfig, timeout_s: float = 30.0):
        self.cfg = cfg
        self.timeout_s = timeout_s

    def supports(self, source: str) -> bool:
        if source.startswith(("http://", "https://")):
            return True
        return Path(source).suffix.lower() in {".html", ".htm"}

    def parse(self, source: str) -> ParsedDocument:
        is_url = source.startswith(("http://", "https://"))
        if is_url:
            html, source_uri = self._fetch(source), source
            source_path = None
        else:
            path = Path(source)
            html = path.read_text(encoding="utf-8", errors="replace")
            source_uri = None
            source_path = str(path)

        soup = BeautifulSoup(html, "lxml")
        metadata = self._extract_metadata(soup)
        title = self._extract_title(soup, source)

        _strip_chrome(soup)
        root = _pick_root(soup)

        blocks, warnings = self._walk(root)

        full_text = "\n\n".join(b.text for b in blocks)
        if len(full_text) < self.cfg.min_chars_per_doc:
            warnings.append(
                f"Document has only {len(full_text)} chars of content "
                f"(< min_chars_per_doc={self.cfg.min_chars_per_doc})"
            )

        doc_id_source = source_uri or str(Path(source_path).resolve()) if source_path else source_uri
        document = SourceDocument(
            doc_id=SourceDocument.make_doc_id(doc_id_source or source),
            title=title,
            source_type=SourceType.WEB if is_url else SourceType.HTML,
            source_path=source_path,
            source_uri=source_uri,
            text=full_text,
            metadata=metadata,
        )
        return ParsedDocument(document=document, blocks=blocks, warnings=warnings)

    # -- fetching ---------------------------------------------------------

    def _fetch(self, url: str) -> str:
        try:
            resp = httpx.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=self.timeout_s,
                follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            raise HTMLFetchError(f"Failed to fetch {url}: {exc}") from exc
        if resp.status_code != 200:
            raise HTMLFetchError(
                f"Failed to fetch {url}: HTTP {resp.status_code}"
            )
        return resp.text

    # -- metadata / title ---------------------------------------------------

    def _extract_metadata(self, soup: BeautifulSoup) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        desc = soup.find("meta", attrs={"name": "description"})
        if desc and desc.get("content"):
            meta["description"] = desc["content"].strip()
        for tag in soup.find_all("meta", attrs={"property": re.compile(r"^og:")}):
            if tag.get("content"):
                key = tag["property"][3:]
                meta[f"og_{key}"] = tag["content"].strip()
        return meta

    def _extract_title(self, soup: BeautifulSoup, source: str) -> str:
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return normalize_whitespace(h1.get_text(" ", strip=True))
        if soup.title and soup.title.get_text(strip=True):
            raw = normalize_whitespace(soup.title.get_text(" ", strip=True))
            return re.sub(r"\s*[|–\-]\s*[^|–\-]+$", "", raw) or raw
        if source.startswith(("http://", "https://")):
            path = urlparse(source).path.strip("/")
            return path.rsplit("/", 1)[-1] if path else urlparse(source).netloc
        return Path(source).stem

    # -- tree walk ------------------------------------------------------

    def _walk(self, root: Tag) -> tuple[list[Block], list[str]]:
        blocks: list[Block] = []
        warnings: list[str] = []
        stack: list[tuple[int, str]] = []
        order = 0

        def add(text: str, kind: str, heading_level: int | None = None) -> None:
            nonlocal order
            cleaned = normalize_block_text(text, kind)
            if not cleaned or _is_pure_punct(cleaned):
                return
            blocks.append(
                Block(
                    text=cleaned,
                    kind=kind,  # type: ignore[arg-type]
                    page=None,
                    section_path=[h for _, h in stack],
                    heading_level=heading_level,
                    char_start=None,
                    char_end=None,
                    order=order,
                )
            )
            order += 1

        def visit(node: Tag) -> None:
            for child in node.children:
                if isinstance(child, NavigableString):
                    continue
                if not isinstance(child, Tag):
                    continue
                name = child.name

                if name in _HEADING_TAGS:
                    level = int(name[1])
                    heading_text = normalize_whitespace(child.get_text(" ", strip=True))
                    if heading_text:
                        stack[:] = [(lv, h) for lv, h in stack if lv < level]
                        add(heading_text, "heading", level)
                        stack.append((level, heading_text))
                    continue

                if name == "table":
                    flat = _flatten_table(child)
                    add(flat, "table")
                    continue

                if name in ("pre", "code"):
                    text = child.get_text("\n", strip=False).strip("\n")
                    add(text, "code")
                    continue

                if name in ("ul", "ol"):
                    items = [
                        normalize_whitespace(li.get_text(" ", strip=True))
                        for li in child.find_all("li", recursive=False)
                    ]
                    items = [i for i in items if i]
                    add("\n".join(f"- {i}" for i in items), "list")
                    continue

                if name == "figcaption":
                    add(_clean_text(child), "caption")
                    continue

                if name in ("p", "blockquote"):
                    add(_clean_text(child), "paragraph")
                    continue

                # Structural containers: keep walking so document order and
                # the heading stack stay correct instead of flattening early.
                if name in (
                    "div", "section", "main", "article", "body", "span",
                    "ul", "li", "figure",
                ):
                    visit(child)
                    continue

                # Leaf-ish tags we don't special-case (a, strong, em, etc.)
                # are covered by their containing <p>; skip re-emitting them
                # at this level to avoid duplicate/orphan text.
                if not child.find(list(_HEADING_TAGS) + ["p", "table", "ul", "ol", "pre", "blockquote"]):
                    continue
                visit(child)

        visit(root)
        return blocks, warnings

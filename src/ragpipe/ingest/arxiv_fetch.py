"""Fetch a corpus of arXiv papers (metadata + PDFs) for ingestion testing.

Real arXiv PDFs have exactly the furniture the ingestion phase needs to
strip -- running headers, footers, page numbers, multi-column bodies,
references sections -- so this hits the live arXiv API rather than any
synthetic fixture. The API's stdlib-only Atom XML keeps this dependency-free
(httpx is already a project dependency for HTTP; XML parsing uses
`xml.etree.ElementTree`).

arXiv's usage policy asks for roughly one request per three seconds and a
descriptive User-Agent; both are enforced here rather than left to the
caller, since getting the IP rate-limited would cost the whole corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from ..config import PROJECT_ROOT

ARXIV_API_URL = "http://export.arxiv.org/api/query"
USER_AGENT = "ragpipe-corpus-fetcher/1.0 (https://github.com/ragpipe; contact via repo issues)"

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"

_MIN_PDF_BYTES = 20_000  # smaller is almost always a withdrawal notice
_MAX_ATTEMPTS = 4


class ArxivPaper(BaseModel):
    """One entry from the arXiv Atom feed, normalised for our manifest."""

    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    abstract: str
    primary_category: str
    categories: list[str]
    published: date
    updated: date
    pdf_url: str
    doi: str | None = None
    journal_ref: str | None = None


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_id_version(raw_id: str) -> tuple[str, int]:
    """arXiv entry ids look like '.../abs/2401.12345v2' -- split id from version."""
    tail = raw_id.rsplit("/", 1)[-1]
    m = re.match(r"(.+?)(?:v(\d+))?$", tail)
    assert m is not None
    return m.group(1), int(m.group(2) or 1)


def _parse_entry(entry: ET.Element) -> ArxivPaper:
    def find_text(tag: str, ns: str = _ATOM_NS) -> str:
        el = entry.find(f"{ns}{tag}")
        return el.text or "" if el is not None else ""

    raw_id = find_text("id")
    arxiv_id, version = _split_id_version(raw_id)

    authors = [
        _normalize_ws(a.findtext(f"{_ATOM_NS}name") or "")
        for a in entry.findall(f"{_ATOM_NS}author")
    ]
    authors = [a for a in authors if a]

    categories = [
        c.get("term", "") for c in entry.findall(f"{_ATOM_NS}category") if c.get("term")
    ]
    primary_el = entry.find(f"{_ARXIV_NS}primary_category")
    primary_category = (
        primary_el.get("term", "") if primary_el is not None else (categories[0] if categories else "")
    )

    pdf_url = ""
    for link in entry.findall(f"{_ATOM_NS}link"):
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            pdf_url = link.get("href", "")
            break
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}v{version}"

    published_raw = find_text("published")
    updated_raw = find_text("updated")
    published = datetime.fromisoformat(published_raw.replace("Z", "+00:00")).date()
    updated = datetime.fromisoformat(updated_raw.replace("Z", "+00:00")).date()

    return ArxivPaper(
        arxiv_id=arxiv_id,
        version=version,
        title=_normalize_ws(find_text("title")),
        authors=authors,
        abstract=_normalize_ws(find_text("summary")),
        primary_category=primary_category,
        categories=categories,
        published=published,
        updated=updated,
        pdf_url=pdf_url,
        doi=find_text("doi", ns=_ARXIV_NS) or None,
        journal_ref=find_text("journal_ref", ns=_ARXIV_NS) or None,
    )


def _sleep(delay_s: float) -> None:
    if delay_s > 0:
        time.sleep(delay_s)


def _request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    max_attempts: int = _MAX_ATTEMPTS,
) -> httpx.Response:
    """GET with exponential backoff on timeouts/5xx/429, honouring Retry-After."""
    backoff = 2.0
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.request(method, url, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                if attempt == max_attempts:
                    resp.raise_for_status()
                time.sleep(wait)
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            if attempt == max_attempts:
                raise
            time.sleep(backoff)
            backoff *= 2
    raise last_exc or RuntimeError("unreachable")


def search_arxiv(
    categories: list[str],
    max_results: int,
    start: int = 0,
    query_extra: str | None = None,
    *,
    delay_s: float = 3.0,
    client: httpx.Client | None = None,
) -> list[ArxivPaper]:
    """Query the arXiv Atom API, sorted by submission date descending."""
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    query = f"({cat_query})" if len(categories) > 1 else cat_query
    if query_extra:
        query = f"{query} AND {query_extra}"

    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    owns_client = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0)
    try:
        _sleep(delay_s)
        resp = _request_with_retry(client, "GET", ARXIV_API_URL, params=params)
    finally:
        if owns_client:
            client.close()

    root = ET.fromstring(resp.text)
    return [_parse_entry(entry) for entry in root.findall(f"{_ATOM_NS}entry")]


def _is_valid_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < _MIN_PDF_BYTES:
        return False
    with path.open("rb") as f:
        return f.read(5) == b"%PDF-"


def download_pdf(
    paper: ArxivPaper,
    dest_dir: Path,
    *,
    delay_s: float = 3.0,
    client: httpx.Client | None = None,
) -> Path:
    """Download `<arxiv_id>.pdf`, skipping if a valid copy already exists.

    Raises on failure after retries; the orchestrator decides how to record
    that (skipped vs. failed) since only it knows the size/magic reasons.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{paper.arxiv_id}.pdf"
    if _is_valid_pdf(dest):
        return dest

    owns_client = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True)
    try:
        _sleep(delay_s)
        resp = _request_with_retry(client, "GET", paper.pdf_url)
        tmp = dest.with_suffix(".pdf.part")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)
    finally:
        if owns_client:
            client.close()
    return dest


def fetch_corpus(
    categories: list[str],
    target_count: int,
    dest_dir: Path,
    manifest_path: Path,
    delay_s: float = 3.0,
    dry_run: bool = False,
    progress: Any = None,
) -> dict[str, Any]:
    """Search, download and write the manifest. Idempotent: re-running only
    fills gaps -- existing valid PDFs are never re-fetched.

    `progress`, if given, is called as `progress(event: str, **kw)` for a
    simple CLI to render per-paper output without this function knowing
    about argparse or stdout formatting.
    """

    def emit(event: str, **kw: Any) -> None:
        if progress is not None:
            progress(event, **kw)

    dest_dir.mkdir(parents=True, exist_ok=True)

    papers: list[ArxivPaper] = []
    seen_ids: set[str] = set()
    start = 0
    page_size = min(max(target_count * 2, 20), 100)
    client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True)

    try:
        while len(papers) < target_count and start < target_count * 5 + 200:
            batch = search_arxiv(categories, page_size, start=start, delay_s=delay_s, client=client)
            if not batch:
                break
            for p in batch:
                if p.arxiv_id not in seen_ids:
                    seen_ids.add(p.arxiv_id)
                    papers.append(p)
            start += page_size
            if len(batch) < page_size:
                break
        papers = papers[:target_count]

        if dry_run:
            for p in papers:
                emit("dry_run", paper=p)
            return {
                "corpus": "arxiv-ml",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "categories": categories,
                "count": len(papers),
                "papers": [],
                "failed": [],
                "skipped": [],
                "dry_run": True,
            }

        manifest_papers: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        for paper in papers:
            dest = dest_dir / f"{paper.arxiv_id}.pdf"
            if _is_valid_pdf(dest):
                emit("cached", paper=paper)
                data = dest.read_bytes()
                manifest_papers.append(_manifest_entry(paper, dest, data))
                continue

            try:
                path = download_pdf(paper, dest_dir, delay_s=delay_s, client=client)
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
                emit("failed", paper=paper, reason=str(exc))
                failed.append({"arxiv_id": paper.arxiv_id, "title": paper.title, "reason": str(exc)})
                continue

            size = path.stat().st_size
            if size < _MIN_PDF_BYTES or not _is_valid_pdf(path):
                reason = f"pdf too small or invalid ({size} bytes) -- likely a withdrawal notice"
                path.unlink(missing_ok=True)
                emit("skipped", paper=paper, reason=reason)
                skipped.append({"arxiv_id": paper.arxiv_id, "title": paper.title, "reason": reason})
                continue

            data = path.read_bytes()
            emit("downloaded", paper=paper, bytes=size)
            manifest_papers.append(_manifest_entry(paper, path, data))
    finally:
        client.close()

    manifest = {
        "corpus": "arxiv-ml",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "categories": categories,
        "count": len(manifest_papers),
        "papers": manifest_papers,
        "failed": failed,
        "skipped": skipped,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _manifest_entry(paper: ArxivPaper, path: Path, data: bytes) -> dict[str, Any]:
    try:
        rel_path = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel_path = path.name
    entry = paper.model_dump(mode="json")
    entry["pdf_path"] = rel_path
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    entry["bytes"] = len(data)
    return entry


def restore_from_manifest(
    manifest_path: Path,
    dest_dir: Path,
    *,
    delay_s: float = 3.0,
    client: httpx.Client | None = None,
) -> dict[str, list[str]]:
    """Download exactly the papers a versioned manifest lists, pinned to the
    recorded PDF version, and check each file's sha256.

    `fetch_corpus` searches arXiv for the newest papers, so running it on a
    fresh machine (a CI cache miss) builds a *different* corpus -- and every
    golden-set question would then be asked of papers it was never written
    for. Reproducing the corpus means restoring it, not re-searching.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True)
    result: dict[str, list[str]] = {"ok": [], "mismatch": [], "failed": []}
    try:
        for paper in manifest["papers"]:
            dest = dest_dir / f"{paper['arxiv_id']}.pdf"
            try:
                if not _is_valid_pdf(dest):
                    _sleep(delay_s)
                    resp = _request_with_retry(client, "GET", paper["pdf_url"])
                    tmp = dest.with_suffix(".pdf.part")
                    tmp.write_bytes(resp.content)
                    tmp.replace(dest)
            except Exception as exc:  # recorded, and the caller fails the run
                result["failed"].append(paper["arxiv_id"])
                continue
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            expected = paper.get("sha256")
            if expected and digest != expected:
                result["mismatch"].append(paper["arxiv_id"])
            else:
                result["ok"].append(paper["arxiv_id"])
    finally:
        if owns_client:
            client.close()
    return result

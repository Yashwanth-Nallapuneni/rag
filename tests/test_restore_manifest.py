"""Corpus restore must reproduce the manifest exactly, never re-search."""

from __future__ import annotations

import hashlib
import json

import httpx

from ragpipe.ingest.arxiv_fetch import restore_from_manifest

PDF = b"%PDF-1.7\n" + b"x" * 20000


def _manifest(tmp_path, sha):
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps({"papers": [
        {"arxiv_id": "2609.00001", "pdf_url": "https://arxiv.org/pdf/2609.00001v1", "sha256": sha}
    ]}))
    return m


def _client(body=PDF):
    return httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=body)))


def test_restores_pinned_paper_and_checks_hash(tmp_path):
    m = _manifest(tmp_path, hashlib.sha256(PDF).hexdigest())
    r = restore_from_manifest(m, tmp_path / "raw", delay_s=0, client=_client())
    assert r == {"ok": ["2609.00001"], "mismatch": [], "failed": []}


def test_altered_pdf_is_reported_not_accepted(tmp_path):
    m = _manifest(tmp_path, "0" * 64)
    r = restore_from_manifest(m, tmp_path / "raw", delay_s=0, client=_client())
    assert r["mismatch"] == ["2609.00001"] and not r["ok"]

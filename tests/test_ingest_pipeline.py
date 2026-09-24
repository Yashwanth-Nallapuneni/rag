from __future__ import annotations

from pathlib import Path

import pytest

from ragpipe.ingest.pipeline import (
    UnsupportedSourceError,
    build_chunks,
    discover_sources,
    get_parser,
    parse_sources,
    read_chunks,
    write_chunks,
)

CORPUS = Path(__file__).resolve().parents[1] / "data" / "raw"


def test_routes_by_source_shape(settings):
    assert get_parser("a.pdf", settings).name
    assert get_parser("a.md", settings).name
    assert get_parser("a.html", settings).name
    assert get_parser("https://example.com/x", settings).name


def test_rejects_unknown_source(settings):
    with pytest.raises(UnsupportedSourceError):
        get_parser("archive.tar.gz", settings)


def test_parse_failures_are_collected_not_fatal(settings, tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"this is definitely not a pdf")
    good = tmp_path / "good.md"
    good.write_text("# Title\n\nSome real prose about self-attention mechanisms.\n")
    parsed, failures = parse_sources([str(bad), str(good)], settings)
    assert len(parsed) == 1, "one bad document must not abandon the corpus"
    assert len(failures) == 1


def test_discover_sources_filters_and_sorts(tmp_path):
    (tmp_path / "b.md").write_text("# B")
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "skip.zip").write_bytes(b"x")
    found = [Path(p).name for p in discover_sources(tmp_path)]
    assert found == ["a.pdf", "b.md"]


def test_chunks_roundtrip_through_jsonl(settings, tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(
        "# Title\n\n## Section One\n\n"
        + " ".join(f"Sentence {i} about retrieval augmented generation." for i in range(60))
    )
    parsed, _ = parse_sources([str(md)], settings)
    chunks = build_chunks(parsed, settings)
    assert chunks
    path = write_chunks(chunks, tmp_path / "chunks.jsonl")
    restored = read_chunks(path)
    assert [c.chunk_id for c in restored] == [c.chunk_id for c in chunks]
    assert restored[0].section_path == chunks[0].section_path


@pytest.mark.integration
@pytest.mark.skipif(not list(CORPUS.glob("*.pdf")), reason="corpus not fetched")
def test_real_corpus_ingests_with_provenance(settings):
    """Every chunk from a PDF must be citable: title, page and text."""
    sample = sorted(CORPUS.glob("*.pdf"))[:3]
    parsed, failures = parse_sources([str(p) for p in sample], settings)
    assert not failures
    chunks = build_chunks(parsed, settings)
    assert len(chunks) > 20
    for c in chunks:
        assert c.text.strip()
        assert c.doc_title
        assert c.page_start is not None, f"{c.chunk_id} has no page for its citation"
        assert c.token_count <= settings.chunking.chunk_size * 1.15


@pytest.mark.integration
@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data/processed/chunks.jsonl").exists(),
    reason="run `make ingest` first",
)
def test_real_corpus_overlap_and_budget():
    """End-to-end guarantee on the actual corpus: adjacent chunks share text,
    and chunk sizes stay inside the spec's 500-800 token window."""
    import collections
    import re

    root = Path(__file__).resolve().parents[1]
    chunks = read_chunks(root / "data/processed/chunks.jsonl")
    assert len(chunks) > 500

    def shingles(text: str, k: int = 8) -> set[tuple[str, ...]]:
        w = re.findall(r"\w+", text.lower())
        return {tuple(w[i : i + k]) for i in range(max(0, len(w) - k + 1))}

    by_doc: dict[str, list] = collections.defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)

    pairs = overlapping = 0
    for docs in by_doc.values():
        docs.sort(key=lambda c: c.chunk_index)
        for a, b in zip(docs, docs[1:], strict=False):
            pairs += 1
            if shingles(a.text) & shingles(b.text):
                overlapping += 1

    # The remainder are windows filled by a single long sentence or atomic
    # block, where there is genuinely nothing to carry over.
    assert overlapping / pairs > 0.95, f"only {overlapping}/{pairs} pairs overlap"
    assert all(c.page_start is not None for c in chunks)
    assert sum(1 for c in chunks if c.token_count > 700) == 0

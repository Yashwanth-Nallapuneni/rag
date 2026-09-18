from __future__ import annotations

import re
from pathlib import Path

import pytest

from ragpipe.ingest.pdf import PDFParser
from ragpipe.schemas import SourceType

CORPUS = Path(__file__).resolve().parents[1] / "data" / "raw"
PDFS = sorted(CORPUS.glob("*.pdf"))

pytestmark = pytest.mark.skipif(not PDFS, reason="corpus not fetched")


@pytest.fixture(scope="module")
def parsed_sample():
    from ragpipe.config import load_settings

    parser = PDFParser(load_settings().ingest)
    return [parser.parse(str(p)) for p in PDFS[:4]]


def test_documents_have_titles_and_type(parsed_sample):
    for doc in parsed_sample:
        assert doc.document.title.strip()
        assert doc.document.source_type == SourceType.PDF
        assert len(doc.document.title) > 5, "a title that short is probably junk"


def test_pages_are_one_indexed_and_in_range(parsed_sample):
    """An off-by-one here is a user-visible wrong citation."""
    for doc in parsed_sample:
        pages = [b.page for b in doc.blocks if b.page is not None]
        assert pages, "no page numbers attached"
        assert min(pages) >= 1
        assert max(pages) <= (doc.document.page_count or max(pages))


def test_blocks_are_in_page_order(parsed_sample):
    for doc in parsed_sample:
        pages = [b.page for b in doc.blocks if b.page is not None]
        assert pages == sorted(pages), "blocks are not in reading order across pages"


def test_heading_hierarchy_is_populated(parsed_sample):
    """At least some standard paper sections must be found, or the section
    half of every citation is empty."""
    found = 0
    for doc in parsed_sample:
        headings = " ".join(b.text.lower() for b in doc.blocks if b.is_heading)
        if re.search(r"introduction|abstract|conclusion|method|experiment|related work", headings):
            found += 1
    assert found >= 2, "heading detection failed across most of the sample"


def test_most_body_blocks_carry_a_section(parsed_sample):
    for doc in parsed_sample:
        body = doc.body_blocks()
        assert body
        with_section = sum(1 for b in body if b.section_path)
        assert with_section / len(body) > 0.5, "over half the blocks have no section"


def test_running_furniture_is_stripped(parsed_sample):
    """Page numbers and running headers must not survive into chunk text."""
    for doc in parsed_sample:
        for block in doc.body_blocks():
            stripped = block.text.strip()
            assert not re.fullmatch(r"[-–—\s]*\d{1,3}[-–—\s]*", stripped), (
                f"bare page number survived: {stripped!r}"
            )


def test_dropped_furniture_is_recorded(parsed_sample):
    """Removals must be auditable -- silent stripping is how ingestion bugs
    hide."""
    assert any(doc.dropped for doc in parsed_sample)


def test_reading_order_produces_continuous_prose(parsed_sample):
    """The two-column failure mode: alternating columns yields text with an
    abnormal rate of sentences that never terminate. Check that a healthy
    fraction of body text ends in real sentence punctuation."""
    for doc in parsed_sample:
        prose = [b for b in doc.body_blocks() if b.kind == "paragraph" and len(b.text) > 200]
        if len(prose) < 5:
            continue
        terminated = sum(1 for b in prose if b.text.rstrip().endswith((".", "!", "?", '"')))
        assert terminated / len(prose) > 0.5, (
            f"{doc.document.title}: most paragraphs do not end in sentence "
            "punctuation, which is what interleaved columns look like"
        )


def test_no_empty_or_whitespace_blocks(parsed_sample):
    for doc in parsed_sample:
        assert all(b.text.strip() for b in doc.blocks)


def test_hyphenation_is_rejoined(parsed_sample):
    """PDF line-wrap hyphens must be healed or BM25 sees broken tokens."""
    for doc in parsed_sample:
        text = doc.text
        # A hyphen followed immediately by a space then a lowercase letter is
        # the signature of an unhealed line break.
        broken = len(re.findall(r"[a-z]- [a-z]", text))
        assert broken < 30, f"{broken} unhealed hyphenations in {doc.document.title}"


def test_corrupt_pdf_fails_cleanly(settings, tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4\nnot really a pdf at all")
    parser = PDFParser(settings.ingest)
    try:
        result = parser.parse(str(bad))
    except Exception as exc:
        assert "pdf" in repr(exc).lower() or "parse" in repr(exc).lower()
    else:
        assert result.warnings, "a corrupt PDF must warn or raise, not pass silently"


@pytest.mark.slow
def test_whole_corpus_parses():
    from ragpipe.config import load_settings

    parser = PDFParser(load_settings().ingest)
    failures = []
    for path in PDFS:
        try:
            doc = parser.parse(str(path))
            if not doc.body_blocks():
                failures.append((path.name, "no body blocks"))
        except Exception as exc:  # noqa: BLE001
            failures.append((path.name, repr(exc)))
    assert not failures, f"{len(failures)}/{len(PDFS)} failed: {failures[:5]}"


@pytest.mark.skipif(
    not (CORPUS / "2609.20641.pdf").exists(), reason="specific paper not in corpus"
)
def test_alternating_running_heads_are_stripped(settings):
    """Regression: journal recto/verso styles put the authors on even pages
    and the title on odd ones, so each running head appears on only ~48% of
    pages and slipped under the whole-document repetition threshold. The
    author head then showed up as the section label on 38 chunks."""
    from ragpipe.config import load_settings

    doc = PDFParser(load_settings().ingest).parse(str(CORPUS / "2609.20641.pdf"))
    headings = " ".join(b.text for b in doc.blocks if b.is_heading)
    assert "SCHWENCKE" not in headings.upper(), "running head became a section heading"
    assert any(doc.dropped.get(k) for k in ("headers", "footers"))


@pytest.mark.skipif(not PDFS, reason="corpus not fetched")
def test_no_section_label_is_an_author_running_head(settings):
    """Guards the whole corpus against furniture leaking into citations."""
    from ragpipe.config import load_settings

    parser = PDFParser(load_settings().ingest)
    offenders = []
    for path in PDFS[:12]:
        doc = parser.parse(str(path))
        for block in doc.blocks:
            if block.is_heading and re.search(r"\b[A-Z]\.\s+[A-Z]{2,},", block.text):
                offenders.append((path.name, block.text))
    assert not offenders, f"author running heads detected as headings: {offenders[:3]}"

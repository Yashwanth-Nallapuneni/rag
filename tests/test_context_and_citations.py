from __future__ import annotations

import pytest

from ragpipe.generation.citations import (
    extract_markers,
    resolve_citations,
    split_claims,
    strip_markers,
)
from ragpipe.generation.context import BLOCK_HEADER_RE, render_context
from ragpipe.schemas import Chunk, RetrievedChunk, SourceType


def _rc(idx: int, text: str, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"doc::{idx:05d}",
            doc_id="doc",
            doc_title="Attention Is All You Need",
            chunk_index=idx,
            text=text,
            page_start=page,
            section_path=["3 Model Architecture"],
            source_type=SourceType.PDF,
        ),
        score=1.0 / idx,
        rank=idx,
    )


CHUNKS = [_rc(1, "Self-attention replaces recurrence."), _rc(2, "BERT uses masking.", 2)]


# --- context rendering ----------------------------------------------------


def test_rendered_headers_match_the_parser():
    r = render_context(CHUNKS, max_tokens=1000)
    found = [int(m.group(1)) for m in BLOCK_HEADER_RE.finditer(r.text)]
    assert found == [1, 2], "the writer and its regex have drifted apart"


def test_markers_are_one_based_and_ordered():
    r = render_context(CHUNKS, max_tokens=1000)
    assert r.marker_to_chunk[1].chunk_id == "doc::00001"
    assert r.marker_to_chunk[2].chunk_id == "doc::00002"


def test_locator_appears_in_the_header():
    r = render_context(CHUNKS, max_tokens=1000)
    assert "Attention Is All You Need" in r.text and "p. 1" in r.text


def test_locators_can_be_suppressed():
    r = render_context(CHUNKS, max_tokens=1000, include_locators=False)
    assert "[S1]" in r.text and "p. 1" not in r.text


def test_budget_drops_rather_than_truncates():
    """A half passage the model cites as whole is worse than one fewer
    passage."""
    r = render_context([_rc(i, "word " * 200) for i in range(1, 6)], max_tokens=500)
    assert r.used and r.dropped
    assert len(r.used) + len(r.dropped) == 5
    assert r.tokens <= 500


def test_single_oversized_chunk_is_truncated_not_dropped():
    r = render_context([_rc(1, "word " * 5000)], max_tokens=300)
    assert len(r.used) == 1 and not r.dropped


def test_empty_input():
    r = render_context([], max_tokens=500)
    assert r.text == "" and r.used == []


# --- citation parsing -----------------------------------------------------


def test_source_reference_markers_are_not_citations():
    """The central regression: academic text is full of "[4, 15, 20]" and a
    bare [n] scheme would read those as citations to passages that may not
    even exist."""
    text = "Generative retrieval in search [4, 15, 20, 26] guides generation [30]. [S1]"
    assert extract_markers(text) == [1]
    claims = split_claims(text)
    assert len(claims) == 1 and claims[0].markers == [1]
    assert "[4, 15, 20, 26]" in claims[0].text, "source refs must survive in the text"


def test_strip_markers_keeps_source_references():
    assert strip_markers("Uses attention [12, 15]. [S3]") == "Uses attention [12, 15]."


def test_trailing_marker_attaches_to_its_own_sentence():
    """Attributing a marker to the next sentence verifies every claim against
    its neighbour's passage."""
    claims = split_claims("A is true. [S1] B is false. [S2]")
    assert [(c.text, c.markers) for c in claims] == [
        ("A is true.", [1]),
        ("B is false.", [2]),
    ]


def test_inline_marker_also_works():
    claims = split_claims("A is true [S1]. B is false [S2].")
    assert [c.markers for c in claims] == [[1], [2]]


def test_ellipsis_does_not_fragment_a_sentence():
    claims = split_claims("A SID is y = (y1, . . . ,yT), (1). [S3]")
    assert len(claims) == 1 and claims[0].markers == [3]


def test_academic_abbreviations_do_not_split():
    claims = split_claims("Per Vaswani et al. we use 8 heads [S1]. See Fig. 3 [S2].")
    assert len(claims) == 2


def test_uncited_sentence_is_flagged():
    claims = split_claims("This is unsupported. This is cited. [S1]")
    assert claims[0].uncited and not claims[1].uncited


def test_multiple_markers_on_one_claim():
    claims = split_claims("Both agree. [S1][S2]")
    assert claims[0].markers == [1, 2]


def test_resolve_reports_unknown_markers():
    """A marker pointing at a passage that was never supplied means the
    answer only looks grounded."""
    r = render_context(CHUNKS, max_tokens=1000)
    citations, unknown = resolve_citations("Claim one. [S1] Claim two. [S9]", r)
    assert [c.marker for c in citations] == [1]
    assert unknown == [9]


def test_resolved_citation_carries_full_provenance():
    r = render_context(CHUNKS, max_tokens=1000)
    citations, _ = resolve_citations("Grounded. [S2]", r)
    c = citations[0]
    assert c.chunk_id == "doc::00002"
    assert c.page_start == 2
    assert c.section_path == ["3 Model Architecture"]
    assert "p. 2" in c.locator


def test_email_addresses_do_not_fragment_claims():
    """Regression: dots inside "@student.xjtlu.edu.cn" were read as sentence
    terminators, shattering an author block into ten bogus uncited claims and
    dropping a good answer's support ratio to 23%."""
    text = (
        "Authors are {Chenxi.Wu25, Zimu.Wang19}@student.xjtlu.edu.cn and they "
        "report results. [S1]"
    )
    claims = split_claims(text)
    assert len(claims) == 1
    assert claims[0].markers == [1]
    assert "edu.cn" in claims[0].text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("See https://example.com/path.html for details. [S1]", 1),
        ("Accuracy is 91.5% overall. [S1]", 1),
        ("It uses v1.5 weights. [S1]", 1),
        ("Filed under cs.CL last year. [S1]", 1),
        ("A is true. [S1] B is false. [S2]", 2),
    ],
)
def test_dotted_tokens_are_not_sentence_boundaries(text, expected):
    assert len(split_claims(text)) == expected

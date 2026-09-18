from __future__ import annotations

import pytest

from ragpipe.chunking.chunker import chunk_document, split_sentences
from ragpipe.config import ChunkingConfig
from ragpipe.ingest.base import Block, ParsedDocument
from ragpipe.schemas import SourceDocument, SourceType
from ragpipe.tokenization import count_tokens


def _doc(blocks: list[Block]) -> ParsedDocument:
    return ParsedDocument(
        document=SourceDocument(
            doc_id="d1",
            title="Attention Is All You Need",
            source_type=SourceType.PDF,
            source_path="data/raw/1706.03762.pdf",
        ),
        blocks=blocks,
    )


def _prose(n_sentences: int, page: int = 1, section: list[str] | None = None) -> Block:
    text = " ".join(
        f"Sentence number {i} explains the self-attention mechanism in detail "
        f"and adds enough words to consume a realistic number of tokens."
        for i in range(n_sentences)
    )
    return Block(text=text, page=page, section_path=section or ["3 Model Architecture"])


CFG = ChunkingConfig(chunk_size=200, chunk_overlap=50, min_chunk_tokens=20)


def test_respects_token_budget():
    chunks = chunk_document(_doc([_prose(60)]), CFG)
    assert chunks
    # A small tolerance: the final token count is measured on the joined text,
    # which can differ slightly from the sum of its parts.
    assert all(c.token_count <= CFG.chunk_size * 1.1 for c in chunks)


def test_adjacent_chunks_actually_overlap():
    """The overlap is the point of the design; assert it exists, not just that
    the parameter was passed in."""
    chunks = chunk_document(_doc([_prose(80)]), CFG)
    assert len(chunks) >= 3
    for a, b in zip(chunks, chunks[1:]):
        shared = set(split_sentences(a.text)) & set(split_sentences(b.text))
        assert shared, f"chunks {a.chunk_index}/{b.chunk_index} share no sentence"


def test_overlap_is_roughly_the_configured_size():
    chunks = chunk_document(_doc([_prose(80)]), CFG)
    a, b = chunks[0], chunks[1]
    shared = [s for s in split_sentences(a.text) if s in set(split_sentences(b.text))]
    overlap_tokens = count_tokens(" ".join(shared))
    assert 0 < overlap_tokens <= CFG.chunk_overlap * 2


def test_no_sentence_is_split_across_chunks():
    chunks = chunk_document(_doc([_prose(60)]), CFG)
    for c in chunks:
        for sentence in split_sentences(c.text):
            assert sentence.endswith((".", "!", "?")) or len(sentence) > 10


def test_every_chunk_keeps_provenance():
    chunks = chunk_document(
        _doc([_prose(40, page=3, section=["3 Model", "3.2 Attention"])]), CFG
    )
    for c in chunks:
        assert c.page_start == 3
        assert c.section_path == ["3 Model", "3.2 Attention"]
        assert c.doc_title == "Attention Is All You Need"
        assert c.source_path
        assert "p. 3" in c.locator()


def test_page_range_spans_multiple_pages():
    chunks = chunk_document(
        _doc([_prose(4, page=1), _prose(4, page=2)]),
        ChunkingConfig(chunk_size=800, chunk_overlap=100, min_chunk_tokens=10),
    )
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert "pp. 1-2" in chunks[0].locator()


def test_chunk_indices_are_sequential():
    chunks = chunk_document(_doc([_prose(80)]), CFG)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_headings_are_not_chunked_as_content():
    doc = _doc([Block(text="3 Model Architecture", kind="heading", page=1), _prose(5)])
    chunks = chunk_document(doc, CFG)
    assert not any(c.text.startswith("3 Model Architecture") for c in chunks)


def test_heading_aware_never_merges_sections():
    doc = _doc(
        [
            _prose(3, section=["1 Introduction"]),
            _prose(3, section=["2 Related Work"]),
        ]
    )
    cfg = ChunkingConfig(
        strategy="heading_aware", chunk_size=800, chunk_overlap=50, min_chunk_tokens=10
    )
    chunks = chunk_document(doc, cfg)
    tops = [c.section_path[0] for c in chunks]
    assert "1 Introduction" in tops and "2 Related Work" in tops
    for c in chunks:
        assert "Introduction" not in c.text or "Related" not in c.text


def test_oversized_single_sentence_is_split_not_dropped():
    giant = Block(text="word " * 3000, page=1)
    chunks = chunk_document(_doc([giant]), CFG)
    assert len(chunks) > 1
    assert all(c.token_count <= CFG.chunk_size * 1.2 for c in chunks)


def test_atomic_blocks_are_not_sentence_split():
    code = Block(text="def f():\n    return 1. Then more. And more.", kind="code", page=1)
    chunks = chunk_document(_doc([code]), CFG)
    assert len(chunks) == 1
    assert "def f():" in chunks[0].text


def test_empty_document_yields_nothing():
    assert chunk_document(_doc([]), CFG) == []
    assert chunk_document(_doc([Block(text="   ", page=1)]), CFG) == []


def test_tiny_document_still_yields_one_chunk():
    """Below min_chunk_tokens, but dropping it would lose the document."""
    chunks = chunk_document(_doc([Block(text="One short line.", page=1)]), CFG)
    assert len(chunks) == 1


def test_terminates_on_pathological_input():
    """Regression guard: an overlap that consumes the whole window used to
    restart the loop at the same index and never terminate."""
    cfg = ChunkingConfig(chunk_size=210, chunk_overlap=200, min_chunk_tokens=1)
    chunks = chunk_document(_doc([_prose(60)]), cfg)
    assert 0 < len(chunks) < 500


def test_sentence_splitter_survives_academic_abbreviations():
    text = (
        "We follow Vaswani et al. and use 8 heads. See Fig. 3 for details. "
        "Results improve by 2 BLEU, i.e. a modest gain."
    )
    assert len(split_sentences(text)) == 3


@pytest.mark.parametrize("size,overlap", [(500, 100), (650, 100), (800, 100)])
def test_spec_parameter_range_works(size, overlap):
    cfg = ChunkingConfig(chunk_size=size, chunk_overlap=overlap)
    chunks = chunk_document(_doc([_prose(200)]), cfg)
    assert chunks and all(c.token_count <= size * 1.1 for c in chunks)

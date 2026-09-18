from __future__ import annotations

from ragpipe.schemas import Answer, AnswerStatus, Chunk, SourceType


def _chunk(**kw) -> Chunk:
    base = dict(
        chunk_id="doc1::00003",
        doc_id="doc1",
        doc_title="Attention Is All You Need",
        chunk_index=3,
        text="The Transformer uses self-attention.",
        token_count=7,
        page_start=4,
        section_path=["3 Model Architecture", "3.2 Attention"],
        source_type=SourceType.PDF,
    )
    base.update(kw)
    return Chunk(**base)


def test_locator_is_human_readable():
    assert _chunk().locator() == (
        "Attention Is All You Need | 3 Model Architecture > 3.2 Attention | p. 4"
    )
    assert _chunk(page_start=4, page_end=5).locator().endswith("pp. 4-5")
    assert _chunk(page_start=None, section_path=[]).locator() == "Attention Is All You Need"


def test_store_metadata_roundtrip_preserves_provenance():
    original = _chunk(metadata={"arxiv_id": "1706.03762"})
    md = original.to_store_metadata()
    # Chroma only accepts flat primitives.
    assert all(isinstance(v, (str, int, float, bool)) for v in md.values())
    restored = Chunk.from_store(original.chunk_id, original.text, md)
    assert restored.section_path == original.section_path
    assert restored.page_start == original.page_start
    assert restored.doc_title == original.doc_title
    assert restored.metadata["arxiv_id"] == "1706.03762"


def test_store_metadata_omits_none_values():
    md = _chunk(page_start=None, page_end=None, source_path=None).to_store_metadata()
    assert "page_start" not in md and "source_path" not in md


def test_chunk_id_is_sortable():
    ids = [Chunk.make_chunk_id("d", i) for i in (2, 10, 1)]
    assert sorted(ids) == ["d::00001", "d::00002", "d::00010"]


def test_answer_refused_property():
    assert not Answer(question="q", text="a").refused
    assert Answer(question="q", text="", status=AnswerStatus.REFUSED_LOW_SUPPORT).refused

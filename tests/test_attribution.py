"""D15 document-attribution check and D16 splitter abbreviations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ragpipe.generation.answerer import Answerer
from ragpipe.generation.attribution import DocumentIndex, check_attribution, normalize
from ragpipe.generation.citations import split_claims
from ragpipe.schemas import AnswerStatus, Chunk, RetrievedChunk

PAINT = "Paint-Anything: Unified Any-Color Control for Image Generation and Editing"
OTHER = "Reproducible Widgets for Federated Learning"
THIRD = "Attention Is All You Need In Retrieval"
GENERIC = "Language Models: A Survey of Methods"


def _chunk(doc_id: str, title: str, i: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::{i:05d}", doc_id=doc_id, doc_title=title,
        chunk_index=i, text=text, page_start=1,
    )


def _rc(chunk: Chunk, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=1.0, rank=rank)


CHUNKS = [
    _chunk("paint", PAINT, 1, "Paint-Anything colors images. Code is released under MIT."),
    _chunk("other", OTHER, 25, "We follow the licenses and usage terms of the underlying datasets."),
    _chunk("third", THIRD, 1, "Attention is useful for retrieval."),
    _chunk("gen", GENERIC, 1, "A survey of language models."),
]
INDEX = DocumentIndex.from_chunks(CHUNKS)
Q_PAINT = f'What dataset licensing terms does "{PAINT}" use for its released code and data?'


def test_normalize_collapses_case_dashes_quotes():
    assert normalize("Paint–Anything: “X”") == "paint anything x"


def test_full_title_names_one_document():
    assert INDEX.named_documents(Q_PAINT) == ["paint"]


def test_main_part_before_colon_names_document():
    assert INDEX.named_documents("What licence does paint anything use?") == ["paint"]
    assert INDEX.named_documents("What licence does PAINT-ANYTHING use?") == ["paint"]


def test_misattributed_citation_is_flagged():
    reason = check_attribution(Q_PAINT, [_rc(CHUNKS[1])], INDEX)
    assert reason and PAINT in reason and OTHER in reason


def test_any_wrong_citation_among_correct_ones_is_flagged():
    assert check_attribution(Q_PAINT, [_rc(CHUNKS[0]), _rc(CHUNKS[1])], INDEX)


def test_correct_document_passes():
    assert check_attribution(Q_PAINT, [_rc(CHUNKS[0])], INDEX) is None


def test_no_citations_is_a_noop():
    assert check_attribution(Q_PAINT, [], INDEX) is None


def test_two_named_documents_is_unaffected():
    q = f"Compare Paint-Anything with {OTHER} on licensing."
    assert len(INDEX.named_documents(q)) == 2
    assert check_attribution(q, [_rc(CHUNKS[1])], INDEX) is None


def test_no_named_document_is_unaffected():
    q = "What licensing terms do image editing papers use?"
    assert INDEX.named_documents(q) == []
    assert check_attribution(q, [_rc(CHUNKS[1])], INDEX) is None


def test_generic_words_in_a_title_do_not_trigger():
    # "attention" / "retrieval" / "survey" / "language models" are title words,
    # not a title. Single-token heads also need >= 6 chars and uniqueness.
    for q in (
        "How does attention work in retrieval?",
        "Give me a survey of methods.",
        "What is image generation and editing?",
    ):
        assert INDEX.named_documents(q) == [], q


def test_short_or_shared_single_token_head_is_not_a_name():
    idx = DocumentIndex.from_titles({
        "a": "Zebra: A Study of Stripes", "b": "Zebra Crossing Safety",
        "c": "Foo: Bar Baz", "d": "Quantumlattice: Something Else",
    })
    assert idx.named_documents("Tell me about zebra.") == []  # token in 2 titles
    assert idx.named_documents("Tell me about zebra stripes.") == []
    assert idx.named_documents("What is foo?") == []  # < 6 chars
    assert idx.named_documents("What is quantumlattice?") == ["d"]


def test_disabled_by_config_default_is_true(settings):
    assert settings.citation.enforce_document_attribution is True


# -- through the graph, with no LLM and no model downloads ----------------


class _StubStore:
    name = "stub"

    def iter_chunks(self):
        return CHUNKS


class _StubRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question, where=None):
        return [_rc(c, i + 1) for i, c in enumerate(self.chunks)]


def _answerer(settings, retrieved, response_text):
    settings.citation.min_relevance_score = None
    a = Answerer(settings, _StubStore(), retriever=_StubRetriever(retrieved), verifier=None)
    a.llm = SimpleNamespace(
        name="stub", model="stub",
        complete=lambda req: SimpleNamespace(text=response_text, usage={}),
    )
    return a


def test_graph_refuses_paint_anything_misattribution(settings):
    a = _answerer(settings, [CHUNKS[1], CHUNKS[0]], "It follows the underlying dataset licenses [S1].")
    ans = a.answer(Q_PAINT)
    assert ans.status == AnswerStatus.REFUSED_MISATTRIBUTED
    assert ans.refused and PAINT in ans.refusal_reason and OTHER in ans.refusal_reason
    assert not ans.citations


def test_graph_passes_correct_attribution(settings):
    a = _answerer(settings, [CHUNKS[1], CHUNKS[0]], "The code is released under MIT [S2].")
    assert a.answer(Q_PAINT).status == AnswerStatus.ANSWERED


def test_graph_check_can_be_disabled(settings):
    settings.citation.enforce_document_attribution = False
    a = _answerer(settings, [CHUNKS[1], CHUNKS[0]], "Dataset licenses are followed [S1].")
    assert a.answer(Q_PAINT).status == AnswerStatus.ANSWERED


def test_flag_changes_the_config_fingerprint(settings):
    before = settings.fingerprint()
    settings.citation.enforce_document_attribution = False
    assert settings.fingerprint() != before


# -- D16 splitter ----------------------------------------------------------


@pytest.mark.parametrize("text", [
    "The student is trained with a weighted L1 loss (Eqs. 3, 4) during fine-tuning [S1].",
    "Method follows Smith et al. 2020 and works well [S1].",
    "Results improve (see Fig. 2 and Figs. 3-4) over the baseline [S1].",
    "Details are in Sec. 4 and Secs. 5-6, Tab. 2, Ref. 7, Refs. 8 [S1].",
    "Gains are approx. 3 points, cf. the baseline, vs. prior work [S1].",
    "This is listed as No. 3 in the table [S1].",
    "Techniques such as e.g. dropout, i.e. regularisation, help [S1].",
])
def test_abbreviations_do_not_split(text):
    claims = split_claims(text)
    assert len(claims) == 1 and claims[0].markers == [1]


def test_real_sentence_end_after_figure_number_still_splits():
    claims = split_claims("The gain is shown in Fig. 3. The model converges faster [S2].")
    assert [c.markers for c in claims] == [[], [2]]


def test_sentence_ending_in_no_still_splits():
    claims = split_claims("The answer is No. The paper never says [S1].")
    assert len(claims) == 2

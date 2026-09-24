"""Sparse (BM25) retrieval.

Complements `DenseRetriever`: where dense is strong on paraphrase and weak
on exact identifiers (model names, arXiv ids, metric acronyms), BM25 is the
reverse. Phase 4's fusion combines both, so this module keeps the same
per-stage-score discipline as `dense.py` and additionally has to solve a
problem dense retrieval doesn't have: BM25 scores are unbounded and
corpus-dependent, so they must be normalised before they can be fused with
dense's [0, 1] cosine similarities.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from ..config import BM25Config, Settings
from ..ingest.pipeline import read_chunks
from ..schemas import Chunk, RetrievedChunk
from .tokenize import tokenize


def _corpus_hash(chunks: list[Chunk]) -> str:
    """Fingerprint of the exact chunk set an index was built from.

    Hashing chunk_id + text (not just count or ids) means an edited chunk
    body invalidates the cache too, not just an added/removed chunk.
    """
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(c.text.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


class BM25Retriever:
    """BM25Okapi over the full chunk corpus, tokenized with `tokenize()`."""

    name = "sparse"

    def __init__(self, settings: Settings, chunks: list[Chunk] | None = None):
        self.settings = settings
        if chunks is None:
            chunks_path = settings.corpus.processed_path / "chunks.jsonl"
            if not chunks_path.exists():
                raise FileNotFoundError(
                    f"no chunks at {chunks_path} -- run `ragpipe ingest` first"
                )
            chunks = read_chunks(chunks_path)
        self.chunks: list[Chunk] = chunks
        self.cfg: BM25Config = settings.retrieval.bm25
        self._corpus_hash = _corpus_hash(chunks)

        self._tokenized: list[list[str]] = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(
            self._tokenized, k1=self.cfg.k1, b=self.cfg.b
        )

    # -- retrieval --------------------------------------------------
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        k = k or self.settings.retrieval.candidate_k
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunks:
            return []

        raw_scores = self._bm25.get_scores(q_tokens)

        # A query whose terms never occur in the corpus scores every doc
        # 0.0 -- that is not "top-k junk", it is "no match", so return [].
        if not any(s > 0 for s in raw_scores):
            return []

        candidates = [
            (i, score) for i, score in enumerate(raw_scores) if score > 0
        ]
        if where:
            candidates = [
                (i, s) for i, s in candidates if _matches(self.chunks[i], where)
            ]
        if not candidates:
            return []

        candidates.sort(key=lambda t: t[1], reverse=True)
        candidates = candidates[:k]

        # Min-max normalise the raw BM25 scores over *this* candidate set
        # into [0, 1] so fusion (Phase 4) can combine them with dense's
        # cosine similarities without BM25's unbounded scale dominating a
        # weighted sum. Known weakness: this is per-query, so the top hit
        # is always driven to 1.0 regardless of how strong the absolute
        # match is -- a corpus-wide, or running, calibration would be
        # needed to compare match quality *across* queries. The raw value
        # survives unchanged in `sparse_score` so nothing is lost.
        raw = [s for _, s in candidates]
        lo, hi = min(raw), max(raw)
        span = hi - lo

        out: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(candidates, start=1):
            norm = (score - lo) / span if span > 0 else 1.0
            out.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=norm,
                    sparse_score=score,
                    sparse_rank=rank,
                    rank=rank,
                    retriever="sparse",
                )
            )
        return out

    # -- persistence --------------------------------------------------
    def save(self, path: str | Path | None = None) -> Path:
        path = Path(path) if path is not None else self.cfg.path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "corpus_hash": self._corpus_hash,
            "chunk_ids": [c.chunk_id for c in self.chunks],
            "tokenized": self._tokenized,
            "k1": self.cfg.k1,
            "b": self.cfg.b,
        }
        with path.open("wb") as fh:
            pickle.dump(state, fh)
        return path

    @classmethod
    def load(
        cls, path: str | Path, settings: Settings, chunks: list[Chunk]
    ) -> BM25Retriever:
        """Load a saved index if it still matches `chunks`, else raise.

        Callers should prefer `build_or_load`, which falls back to a fresh
        build automatically -- this raises so a direct caller can't
        silently end up serving a stale index.
        """
        path = Path(path)
        with path.open("rb") as fh:
            state = pickle.load(fh)
        if state["corpus_hash"] != _corpus_hash(chunks):
            raise ValueError("saved BM25 index is stale for the given chunks")

        cfg = settings.retrieval.bm25
        retriever = cls.__new__(cls)
        retriever.settings = settings
        retriever.chunks = chunks
        retriever.cfg = cfg
        retriever._corpus_hash = state["corpus_hash"]
        retriever._tokenized = state["tokenized"]
        retriever._bm25 = BM25Okapi(
            retriever._tokenized, k1=cfg.k1, b=cfg.b
        )
        return retriever

    @classmethod
    def build_or_load(
        cls, settings: Settings, chunks: list[Chunk] | None = None
    ) -> BM25Retriever:
        """Load the on-disk index if it matches the current chunks, else
        (re)build from scratch and persist. This is the seam `get_retriever`
        should use -- a stale sparse index next to a fresh dense one is a
        confusing bug class, so we validate rather than trust the cache."""
        if chunks is None:
            chunks_path = settings.corpus.processed_path / "chunks.jsonl"
            if not chunks_path.exists():
                raise FileNotFoundError(
                    f"no chunks at {chunks_path} -- run `ragpipe ingest` first"
                )
            chunks = read_chunks(chunks_path)

        path = settings.retrieval.bm25.path
        if path.exists():
            try:
                return cls.load(path, settings, chunks)
            except (ValueError, EOFError, pickle.PickleError, KeyError):
                pass  # stale or corrupt -- fall through to a fresh build

        retriever = cls(settings, chunks)
        retriever.save(path)
        return retriever

    # -- introspection --------------------------------------------------
    def stats(self) -> dict[str, Any]:
        vocab = {t for doc in self._tokenized for t in doc}
        lengths = [len(doc) for doc in self._tokenized] or [0]
        return {
            "documents": len(self.chunks),
            "vocabulary_size": len(vocab),
            "avg_doc_length": sum(lengths) / len(lengths),
            "k1": self.cfg.k1,
            "b": self.cfg.b,
        }


def _matches(chunk: Chunk, where: dict[str, Any]) -> bool:
    """Equality-only filter on chunk metadata fields (post-scoring).

    Supported: direct `Chunk` attributes (doc_id, doc_title, source_type,
    ...) and keys under `chunk.metadata`, each checked with `==` against
    the given value. Not supported: ranges, `$in`-style operators, or
    nested/compound conditions -- if `where` needs any of those, filter
    the returned list yourself.
    """
    for key, value in where.items():
        if hasattr(chunk, key):
            if getattr(chunk, key) != value:
                return False
        elif chunk.metadata.get(key) != value:
            return False
    return True

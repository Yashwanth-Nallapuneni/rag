"""ChromaDB implementation of the `VectorStore` protocol.

Chroma stores vectors, documents and metadata separately and hands them back
unordered relative to the ids you queried with, and its `query()` returns a
distance whose meaning depends on the collection's configured space -- not a
similarity. Both of those are easy to get subtly wrong, so this module is the
one place that translates Chroma's shape into the contract `base.py`
promises: similarity in [0, 1] (higher better), and `get()` in request order.
"""

from __future__ import annotations

from typing import Any

import chromadb

from ..config import VectorStoreConfig
from ..schemas import Chunk
from .base import VectorStoreError

_DEFAULT_MAX_BATCH = 2000


class ChromaStore:
    """Persistent Chroma-backed vector store."""

    name = "chroma"

    def __init__(self, cfg: VectorStoreConfig, dimension: int):
        self.cfg = cfg
        self.dimension = dimension
        self._client = chromadb.PersistentClient(path=str(cfg.store_path))
        self._collection = self._get_or_create()

    def _get_or_create(self):
        # No `embedding_function` here on purpose: Chroma defaults to its own
        # ONNX MiniLM embedder when one isn't supplied, which would silently
        # embed queries/documents with a different model than the one this
        # pipeline is configured to use -- vectors from two models in the same
        # space are meaningless neighbours, and nothing would error.
        return self._client.get_or_create_collection(
            name=self.cfg.collection,
            metadata={"hnsw:space": self.cfg.distance},
        )

    def _max_batch_size(self) -> int:
        getter = getattr(self._client, "get_max_batch_size", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:  # noqa: BLE001 - fall back below
                pass
        return _DEFAULT_MAX_BATCH

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"chunk/embedding count mismatch: {len(chunks)} chunks vs "
                f"{len(embeddings)} embeddings"
            )
        for i, vec in enumerate(embeddings):
            if len(vec) != self.dimension:
                raise VectorStoreError(
                    f"embedding {i} has dimension {len(vec)}, expected "
                    f"{self.dimension}"
                )
        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.to_store_metadata() for c in chunks]

        batch_size = self._max_batch_size()
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        return len(chunks)

    def query(
        self,
        embedding: list[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        if not ids:
            return []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        hits: list[tuple[Chunk, float]] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
            hits.append((Chunk.from_store(cid, doc or "", meta or {}), self._to_similarity(dist)))
        return hits

    def _to_similarity(self, distance: float) -> float:
        """Map Chroma's raw distance to similarity in [0, 1], per space.

        - cosine: Chroma returns 1 - cosine_similarity, so similarity is
          simply 1 - distance; clamp because float error can push it a hair
          outside [0, 1] (e.g. -1e-9 or 1.0000001).
        - l2: Chroma returns *squared* Euclidean distance, unbounded above,
          so there is no exact linear map back to [0, 1]. `1 / (1 + distance)`
          is a monotonic decreasing map that lands in (0, 1] with 0 distance
          -> 1.0, which is all score fusion needs (relative order preserved).
        - ip: Chroma returns raw (negative) inner product as "distance" --
          more negative is more similar. Negate it, then squash through the
          same 1/(1+x) monotonic map so it lands in (0, 1] regardless of the
          embedding scale.
        """
        space = self.cfg.distance
        if space == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))
        if space == "l2":
            return 1.0 / (1.0 + max(0.0, distance))
        if space == "ip":
            return 1.0 / (1.0 + max(0.0, -distance))
        raise VectorStoreError(f"unknown distance space: {space}")

    def get(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        result = self._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        by_id: dict[str, Chunk] = {}
        for cid, doc, meta in zip(
            result.get("ids") or [], result.get("documents") or [], result.get("metadatas") or [],
            strict=True,
        ):
            by_id[cid] = Chunk.from_store(cid, doc or "", meta or {})
        # Chroma does not guarantee result order matches the ids requested;
        # rebuild in request order and silently drop ids that weren't found.
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def iter_chunks(self) -> list[Chunk]:
        """All chunks, ordered by id for deterministic sparse-index builds."""
        got = self._collection.get(include=["documents", "metadatas"])
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        out = [
            Chunk.from_store(cid, text or "", md or {})
            for cid, text, md in zip(ids, docs, metas, strict=True)
        ]
        return sorted(out, key=lambda c: c.chunk_id)

    def document_ids(self) -> list[str]:
        """Distinct doc_ids present in the collection."""
        got = self._collection.get(include=["metadatas"])
        seen: dict[str, None] = {}
        for md in got.get("metadatas") or []:
            doc_id = (md or {}).get("doc_id")
            if doc_id:
                seen.setdefault(str(doc_id), None)
        return sorted(seen)

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(name=self.cfg.collection)
        self._collection = self._get_or_create()

    def stats(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "collection": self.cfg.collection,
            "path": str(self.cfg.store_path),
            "count": self.count(),
            "dimension": self.dimension,
            "distance": self.cfg.distance,
        }

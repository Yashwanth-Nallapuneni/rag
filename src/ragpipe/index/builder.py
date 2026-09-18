"""Build (and rebuild-on-drift) the Chroma index from processed chunks.

Embedding 1000+ chunks is the slowest step in the pipeline, so this module is
idempotent: it fingerprints the config plus the exact set of chunks indexed
and skips re-embedding when nothing has changed. It is also the one place
that decides *when* mixing embedding models would poison the index and resets
the collection first when that's about to happen.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from ..ingest.pipeline import read_chunks
from ..providers import get_embedder
from ..schemas import Chunk
from .base import VectorStoreError
from .chroma_store import ChromaStore

MANIFEST_NAME = "index_manifest.json"


def _chunks_hash(chunks: list[Chunk]) -> str:
    """Hash over ids + text so any edit to content (not just chunk count)
    invalidates the manifest, even if the count happens to stay the same."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(c.text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _manifest_path(settings: Settings) -> Path:
    return settings.corpus.processed_path / MANIFEST_NAME


def _current_manifest(settings: Settings, chunks: list[Chunk], dimension: int) -> dict[str, Any]:
    emb = settings.embeddings
    return {
        "config_fingerprint": settings.fingerprint(),
        "embedding_provider": emb.provider,
        "embedding_model": emb.model,
        "dimension": dimension,
        "distance": settings.vector_store.distance,
        "collection": settings.vector_store.collection,
        "chunk_count": len(chunks),
        "chunks_hash": _chunks_hash(chunks),
    }


def _load_manifest(settings: Settings) -> dict[str, Any] | None:
    path = _manifest_path(settings)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_manifest(settings: Settings, manifest: dict[str, Any]) -> None:
    _manifest_path(settings).write_text(json.dumps(manifest, indent=2))


# Manifest fields that, if changed, mean vectors already in the collection
# were produced by a different model/space than the one configured now --
# mixing those in one collection would give meaningless nearest neighbours,
# so the collection must be wiped before rebuilding rather than merged into.
_MODEL_FIELDS = ("embedding_provider", "embedding_model", "dimension", "distance", "collection")


def get_store(settings: Settings) -> ChromaStore:
    """Single accessor for the configured vector store."""
    if settings.vector_store.provider != "chroma":
        raise VectorStoreError(f"unsupported vector store provider: {settings.vector_store.provider}")
    return ChromaStore(settings.vector_store, settings.embeddings.dimension)


def build_index(
    settings: Settings,
    chunks: list[Chunk] | None = None,
    *,
    force: bool = False,
    batch_size: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()

    if chunks is None:
        chunks_path = settings.corpus.processed_path / "chunks.jsonl"
        if not chunks_path.exists():
            raise VectorStoreError(
                f"no processed chunks at {chunks_path}; run `ragpipe ingest` first"
            )
        chunks = read_chunks(chunks_path)

    embedder = get_embedder(settings)
    dimension = embedder.dimension
    manifest = _current_manifest(settings, chunks, dimension)

    store = get_store(settings)
    previous = _load_manifest(settings)

    if not force and previous == manifest and store.count() == len(chunks):
        elapsed = round(time.perf_counter() - started, 3)
        if progress:
            print(f"[index] manifest unchanged, {len(chunks)} chunks already indexed -- skipping")
        return {
            "skipped": True,
            "chunks_indexed": len(chunks),
            "documents": len({c.doc_id for c in chunks}),
            "dimension": dimension,
            "embedding_model": f"{settings.embeddings.provider}:{settings.embeddings.model}",
            "collection_count": store.count(),
            "elapsed_s": elapsed,
            "config_fingerprint": settings.fingerprint(),
        }

    model_changed = previous is not None and any(
        previous.get(f) != manifest.get(f) for f in _MODEL_FIELDS
    )
    if force or model_changed:
        store.reset()

    batch = batch_size or settings.embeddings.batch_size
    n = len(chunks)
    written = 0
    n_batches = max(1, (n + batch - 1) // batch)
    log_every = max(1, n_batches // 20) if progress else n_batches + 1
    embed_started = time.perf_counter()

    for i in range(0, n, batch):
        batch_chunks = chunks[i : i + batch]
        vectors = embedder.embed_documents([c.text for c in batch_chunks])
        written += store.upsert(batch_chunks, vectors)

        batch_num = i // batch + 1
        if progress and (batch_num % log_every == 0 or batch_num == n_batches):
            elapsed = time.perf_counter() - embed_started
            rate = written / elapsed if elapsed > 0 else 0.0
            print(
                f"[index] batch {batch_num}/{n_batches}  {written}/{n} chunks  "
                f"{rate:.1f} chunks/s"
            )

    _write_manifest(settings, manifest)

    elapsed = round(time.perf_counter() - started, 3)
    return {
        "skipped": False,
        "chunks_indexed": written,
        "documents": len({c.doc_id for c in chunks}),
        "dimension": dimension,
        "embedding_model": f"{settings.embeddings.provider}:{settings.embeddings.model}",
        "collection_count": store.count(),
        "elapsed_s": elapsed,
        "config_fingerprint": settings.fingerprint(),
    }

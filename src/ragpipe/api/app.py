"""FastAPI service layer over the RAG pipeline.

The `Answerer` loads a sentence-transformers model on construction (~7s), so
it is built once at startup via `lifespan` and stashed on `app.state` --
never per request. If that build fails (most commonly: no index has been
built yet), the app still comes up so `/health` can report the problem
instead of the process crashing at import time; `/query` then answers with a
clear 503 rather than a bare stack trace.

Refusal is a normal product outcome, not a failure: `POST /query` returns
HTTP 200 with `status="refused_*"` when the pipeline can't support an answer,
so the client can render it like any other answer.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import Settings, load_settings
from ..generation.answerer import Answerer, build_answerer
from ..logging_utils import get_logger, setup_logging
from ..providers import get_embedder, get_llm, get_reranker
from ..schemas import Answer
from .models import (
    ChunkResponse,
    ErrorResponse,
    HealthResponse,
    IndexResponse,
    ProviderHealth,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)

log = get_logger(__name__)


class _AnswererState:
    """Holds the one `Answerer` instance (or the reason it couldn't build)."""

    def __init__(self) -> None:
        self.answerer: Answerer | None = None
        self.error: str | None = None


def _build_state(settings: Settings) -> _AnswererState:
    state = _AnswererState()
    try:
        state.answerer = build_answerer(settings)
    except Exception as exc:  # noqa: BLE001 - report, never crash startup
        log.warning("answerer failed to build at startup: %s", exc)
        state.error = str(exc)
    return state


def create_app(settings: Settings | None = None) -> FastAPI:
    """App factory so tests can inject offline settings (mock providers,
    a temp vector store) instead of touching the real, model-backed one."""

    settings = settings or load_settings()
    setup_logging(settings.logging.level, settings.logging.json_output)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.answerer_state = _build_state(settings)
        yield

    app = FastAPI(title="ragpipe", version="1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Stamp every response with a request id and log its latency, so a
        slow or wrong answer in a demo can be traced to one log line."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000
            log.exception(
                "request failed  id=%s method=%s path=%s  %.1fms",
                request_id, request.method, request.url.path, elapsed,
            )
            raise
        elapsed = (time.perf_counter() - started) * 1000
        log.info(
            "request  id=%s method=%s path=%s status=%d  %.1fms",
            request_id, request.method, request.url.path,
            response.status_code, elapsed,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Never let an unexpected error leak a raw HTML traceback."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        log.exception("unhandled error  id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                detail=str(exc),
                request_id=request_id,
            ).model_dump(),
        )

    def get_settings_dep() -> Settings:
        return app.state.settings

    def get_answerer(request: Request) -> Answerer:
        """Injects the single startup-built `Answerer`; 503s with a clear
        remedy when the index/providers were never available."""
        state: _AnswererState = request.app.state.answerer_state
        if state.answerer is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "answerer is not available "
                    f"({state.error or 'not initialized'}); "
                    "run `ragpipe index` to build the vector store, then restart"
                ),
            )
        return state.answerer

    def _request_id(request: Request) -> str:
        return getattr(request.state, "request_id", str(uuid.uuid4()))

    # -- routes -----------------------------------------------------------

    @app.get("/", response_model=IndexResponse)
    async def index(request: Request) -> IndexResponse:
        return IndexResponse(
            request_id=_request_id(request),
            service="ragpipe",
            endpoints={
                "POST /query": "ask a question, get a grounded answer",
                "GET /chunk/{chunk_id}": "fetch a chunk's full text by id",
                "GET /health": "liveness + provider/store health",
                "GET /stats": "store and config statistics",
            },
        )

    @app.post("/query", response_model=QueryResponse)
    async def query(
        request: Request,
        body: QueryRequest,
        answerer: Answerer = Depends(get_answerer),
    ) -> QueryResponse:
        answer: Answer = answerer.answer(body.question, k=body.k, where=body.filters)
        return QueryResponse(request_id=_request_id(request), answer=answer)

    @app.get("/chunk/{chunk_id}", response_model=ChunkResponse)
    async def get_chunk(
        chunk_id: str,
        request: Request,
        settings: Settings = Depends(get_settings_dep),
    ) -> ChunkResponse:
        # Citation click-through: a reader lands on the exact passage an
        # answer cited, so this must return the full chunk text, not a
        # snippet. Store loaded straight from settings (cheap, no model
        # weights) rather than through the Answerer, so this endpoint keeps
        # working even if the Answerer itself failed to build.
        from ..index.builder import get_store

        store = get_store(settings)
        found = store.get([chunk_id])
        if not found:
            raise HTTPException(status_code=404, detail=f"unknown chunk_id: {chunk_id}")
        return ChunkResponse(request_id=_request_id(request), chunk=found[0])

    @app.get("/health", response_model=HealthResponse)
    async def health(
        request: Request,
        settings: Settings = Depends(get_settings_dep),
    ) -> HealthResponse:
        # Provider .health() implementations never make paid calls (they
        # check local state / cached clients), so this stays free and fast
        # to poll from an orchestrator.
        providers: dict[str, dict[str, Any]] = {}
        for label, getter in (
            ("llm", get_llm),
            ("embeddings", get_embedder),
            ("reranker", get_reranker),
        ):
            try:
                providers[label] = getter(settings).health()
            except Exception as exc:  # noqa: BLE001 - never 500 a health check
                providers[label] = {"ready": False, "error": str(exc)}

        state: _AnswererState = request.app.state.answerer_state
        store_count: int | None = None
        try:
            from ..index.builder import get_store

            store_count = get_store(settings).count()
        except Exception as exc:  # noqa: BLE001
            log.warning("health: could not read store count: %s", exc)

        providers_ready = all(p.get("ready") for p in providers.values())
        ready = state.answerer is not None and providers_ready and bool(store_count)

        return HealthResponse(
            request_id=_request_id(request),
            ready=ready,
            answerer_loaded=state.answerer is not None,
            error=state.error,
            providers=providers,
            store_count=store_count,
            config=settings.describe(),
        )

    @app.get("/stats", response_model=StatsResponse)
    async def stats(
        request: Request,
        settings: Settings = Depends(get_settings_dep),
    ) -> StatsResponse:
        from ..index.builder import get_store

        store = get_store(settings)
        store_stats = store.stats()
        return StatsResponse(
            request_id=_request_id(request),
            store=store_stats,
            # The VectorStore protocol only exposes chunk-level count/get, not
            # a distinct-document listing, so document count isn't derivable
            # without reading every chunk's metadata.
            documents=None,
            chunks=store_stats.get("count"),
            config_fingerprint=settings.fingerprint(),
        )

    return app


app = create_app()

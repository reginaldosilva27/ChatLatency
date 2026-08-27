"""Measurement engine API.

POST /v1/chat/stream        SSE - the measured path (events: meta, token, trace, done)
POST /v1/chat               non-streaming, to compare perception with total time
GET  /v1/traces/summary     per-stage percentiles from the in-process buffer
POST /v1/traces/reset       clears the buffer between experiments
POST /v1/cache/reset        clears the cache (to measure a clean cache-miss p50)
POST /v1/cache/invalidate   invalidation by topic (content version)
GET  /v1/corpus             what is indexed, so the UI can show it
GET  /healthz               dependency check + effective configuration
GET  /                      the ChatLatency front end
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .cache import LayeredCache
from .config import get_settings
from .graph import SENTINEL_DONE, AgentRuntime, warm_tokenizer
from .llm import LLM
from .pricing import CSV_PATH, get_price_book
from .retrieval import build_retriever
from .telemetry import Trace, TraceBuffer, summarize
from .tools import (
    KnowledgeBase,
    ToolBox,
    WebSearch,
    load_corpus,
)

logger = logging.getLogger("harness")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
UI_FILE = "index.html"


class ChatRequest(BaseModel):
    question: str
    locale: str = "en-US"
    topic: str | None = None
    user_context: str | None = Field(
        default=None,
        description="Payload/markdown about the signed-in user, assembled by the "
        "application backend at login",
    )
    history: list[dict[str, str]] = Field(default_factory=list)

    # per-request overrides - they allow an A/B inside the same process
    speculative_retrieval: bool | None = None
    detect_locale: bool | None = None
    classify_intent: bool | None = None
    cache_l1: bool | None = None
    cache_l2: bool | None = None
    force_tier: Literal["nano", "mini", "frontier"] | None = None
    agentic: bool | None = None

    # the client's send instant (bench perf_counter), to attribute network time
    client_sent_ms: float | None = None


def _price_health(settings: Any) -> dict[str, Any]:
    """The /healthz price block.

    It shows the price ALREADY RESOLVED per tier, not the raw config: with a
    per-model catalog, reading PRICE_IN_PER_MTOK from .env no longer tells you
    what each tier is paying. `origin` and `fetched_at` travel with it, because a
    cost without provenance and without an age is not usable in a proposal.
    """
    book = get_price_book(settings)
    tiers = {
        "nano": settings.nano_model,
        "mini": settings.mini_model,
        "frontier": settings.frontier_model,
        "embedding": settings.embedding_model,
    }
    if "web_browse" in settings.enabled_tools_list or settings.enable_web_browse:
        tiers["stagehand"] = settings.stagehand_model
    resolved = {t: book.resolve(m).as_dict() for t, m in tiers.items()}
    return {
        "catalog": {
            "models": len(book),
            "fetched_at": book.fetched_at,
            "source": book.source,
            "path": settings.price_catalog_path or str(CSV_PATH),
            "prefix": settings.price_catalog_prefix,
        },
        "currency": settings.price_currency,
        # placeholder = some tier fell back, and its cost is a guess
        "is_placeholder": any(r["is_placeholder"] for r in resolved.values()),
        "by_tier": resolved,
        # compatibility with the UI chip: the price of the answering tier
        "in_per_mtok": resolved["frontier"]["in_per_mtok"],
        "cached_in_per_mtok": resolved["frontier"]["cached_in_per_mtok"],
        "out_per_mtok": resolved["frontier"]["out_per_mtok"],
    }


def build_app() -> FastAPI:
    settings = get_settings()
    buffer = TraceBuffer(settings.trace_buffer_size)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        llm = LLM(settings)
        cache = LayeredCache(settings.redis_url, settings.cache_ttl_s, settings.cache_l2_threshold)
        await cache.ping()
        retriever, table = build_retriever(settings)

        # ChromaDB: index at startup, never on the hot path.
        corpus = load_corpus()
        kb = KnowledgeBase(settings.chroma_collection, settings.chroma_persist_dir)
        web = WebSearch(settings)
        toolbox: ToolBox | None = None
        if settings.enabled_tools_list:
            await asyncio.to_thread(kb.build, corpus["documents"])
            toolbox = ToolBox(settings, kb, table, web)

        # A degraded search backend is too quiet to discover only at /healthz: a
        # number measured with duckduckgo does not hold for browserbase.
        if web.degraded:
            logger.warning(
                "WEB_SEARCH_BACKEND=%s was requested but the credential is missing - "
                "using '%s'. Fill the key in .env and RESTART the server "
                "(--reload watches .py, not .env).",
                settings.web_search_backend,
                web.backend,
            )
        else:
            logger.info("internet search via '%s'", web.backend)

        app.state.rt = AgentRuntime(settings, llm, cache, retriever, table, toolbox)
        app.state.cache = cache
        app.state.buffer = buffer
        app.state.kb = kb
        app.state.web = web
        app.state.toolbox = toolbox
        app.state.corpus = corpus
        # pre-compile the variants used in the A/B so the first request does not
        # pay for graph compilation
        for agentic in (True, False):
            for spec in (True, False):
                app.state.rt.graph(
                    spec, settings.detect_locale, settings.classify_intent, agentic
                )
        # warm the encoder (the first get_encoding may download the BPE file) so
        # the first measured request does not carry that cost
        await asyncio.to_thread(warm_tokenizer)
        yield
        await web.close()

    app = FastAPI(
        title="ChatLatency - a latency and streaming measurement engine for LLM agents",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Local harness: the front end can be served by this app (/) or from another
    # origin (http.server, file://). Open CORS is acceptable because this is a
    # measurement instrument running locally, not an exposed service.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def resolve_opts(req: ChatRequest) -> dict[str, Any]:
        pick = lambda override, default: default if override is None else override  # noqa: E731
        return {
            "speculative": pick(req.speculative_retrieval, settings.speculative_retrieval),
            "detect_locale": pick(req.detect_locale, settings.detect_locale),
            "classify_intent": pick(req.classify_intent, settings.classify_intent),
            "cache_l1": pick(req.cache_l1, settings.cache_l1_enabled),
            "cache_l2": pick(req.cache_l2, settings.cache_l2_enabled),
            "force_tier": req.force_tier,
            "agentic": pick(req.agentic, settings.agentic),
        }

    async def run_graph(req: ChatRequest, trace: Trace, emit: asyncio.Queue) -> dict[str, Any]:
        rt: AgentRuntime = app.state.rt
        opts = resolve_opts(req)
        graph = rt.graph(
            opts["speculative"],
            opts["detect_locale"],
            opts["classify_intent"],
            opts["agentic"],
        )
        state = {
            "question": req.question,
            "locale": req.locale,
            "topic": req.topic,
            "user_context": req.user_context,
            "history": req.history,
            "scratch": {},
        }
        config = {"configurable": {"trace": trace, "emit": emit, "opts": opts}}
        try:
            final = await graph.ainvoke(state, config=config)
        finally:
            await emit.put(SENTINEL_DONE)
        return final

    def finalize(trace: Trace, req: ChatRequest, final: dict[str, Any] | None) -> dict[str, Any]:
        cache: LayeredCache = app.state.cache
        trace.set(
            cache_tier=(final or {}).get("cache_tier", "miss"),
            cache_similarity=(final or {}).get("cache_similarity"),
            output_tokens=(final or {}).get("output_tokens"),
            cache_backend=cache.backend,
            provider=settings.llm_provider,
            retriever=settings.retriever,
            speculative=resolve_opts(req)["speculative"],
            agentic=resolve_opts(req)["agentic"],
            tool_calls=(final or {}).get("tool_records") or [],
            locale=req.locale,
            topic=req.topic,
            history_turns=len(req.history) // 2,
        )
        payload = trace.to_dict()
        app.state.buffer.add(payload)
        return payload

    # ---------------- streaming (the measured path) ----------------

    @app.post("/v1/chat/stream")
    async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
        trace = Trace()

        async def gen() -> AsyncIterator[bytes]:
            emit: asyncio.Queue = asyncio.Queue()
            task = asyncio.create_task(run_graph(req, trace, emit))

            # 'meta' goes out immediately: it gives the client its first byte and
            # makes it possible to separate connection/network latency from
            # reasoning latency.
            yield _sse("meta", {"request_id": trace.request_id, "t_ms": trace.now_ms()})

            error: str | None = None
            try:
                while True:
                    item = await emit.get()
                    if item is SENTINEL_DONE:
                        break
                    if await request.is_disconnected():
                        task.cancel()
                        break
                    yield _sse("token", {"d": item})
            finally:
                final: dict[str, Any] | None = None
                try:
                    final = await task
                except asyncio.CancelledError:
                    error = "client_disconnected"
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                if error:
                    trace.set(error=error)
                payload = finalize(trace, req, final)
                yield _sse("trace", payload)
                yield _sse("done", {"ok": error is None, "error": error})

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",  # no proxy buffering on the hot path
                "Connection": "keep-alive",
            },
        )

    # ---------------- non-streaming (comparison) ----------------

    @app.post("/v1/chat")
    async def chat(req: ChatRequest) -> JSONResponse:
        trace = Trace()
        emit: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(run_graph(req, trace, emit))
        chunks: list[str] = []
        while True:
            item = await emit.get()
            if item is SENTINEL_DONE:
                break
            chunks.append(item)  # type: ignore[arg-type]
        final: dict[str, Any] | None = None
        error = None
        try:
            final = await task
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            trace.set(error=error)
        payload = finalize(trace, req, final)
        return JSONResponse({"answer": "".join(chunks), "trace": payload, "error": error})

    # ---------------- harness operation ----------------

    @app.get("/v1/traces/summary")
    async def traces_summary() -> dict[str, Any]:
        return summarize(app.state.buffer.all())

    @app.get("/v1/traces")
    async def traces_raw(limit: int = 200) -> dict[str, Any]:
        rows = app.state.buffer.all()[-limit:]
        return {"count": len(rows), "traces": rows}

    @app.post("/v1/traces/reset")
    async def traces_reset() -> dict[str, str]:
        app.state.buffer.clear()
        return {"status": "cleared"}

    @app.post("/v1/cache/reset")
    async def cache_reset() -> dict[str, str]:
        await app.state.cache.clear()
        return {"status": "cleared"}

    @app.get("/v1/tools")
    async def list_tools() -> dict[str, Any]:
        """The effective tool schemas - what the model actually sees."""
        tb: ToolBox | None = getattr(app.state, "toolbox", None)
        if tb is None:
            return {"enabled": [], "schemas": []}
        return {"enabled": sorted(tb.enabled), "schemas": tb.schemas()}

    @app.get("/v1/corpus")
    async def corpus_info() -> dict[str, Any]:
        """What is indexed. The UI shows it so a reader can tell which questions
        the assistant can ground and which it cannot."""
        corpus: dict[str, Any] = getattr(app.state, "corpus", None) or load_corpus()
        docs = corpus.get("documents", [])
        topics: dict[str, int] = {}
        for d in docs:
            topics[d.get("topic") or "general"] = topics.get(d.get("topic") or "general", 0) + 1
        return {
            "documents": len(docs),
            "topics": topics,
            "locales": sorted({d.get("locale", "en-US") for d in docs}),
            "glossary": [
                {"id": g["id"], "name": g["name"], "category": g.get("category")}
                for g in corpus.get("glossary", [])
            ],
            "titles": [
                {"id": d["id"], "title": d["title"], "topic": d.get("topic"), "type": d.get("type")}
                for d in docs
            ],
        }

    @app.post("/v1/cache/invalidate")
    async def cache_invalidate(topic: str) -> dict[str, Any]:
        removed = await app.state.cache.invalidate_topic(topic)
        return {"topic": topic, "removed": removed}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        cache: LayeredCache = app.state.cache
        return {
            "status": "ok",
            "provider": settings.llm_provider,
            "models": {
                "nano": settings.nano_model,
                "mini": settings.mini_model,
                "frontier": settings.frontier_model,
            },
            "reasoning_effort": settings.reasoning_effort,
            "endpoint": settings.foundry_base_url
            if settings.llm_provider == "foundry"
            else settings.azure_openai_endpoint,
            "retriever": settings.retriever,
            "agentic": settings.agentic,
            "tools": settings.enabled_tools_list,
            "max_tool_hops": settings.max_tool_hops,
            # the only tool that opens a browser session - off by default
            "web_browse_enabled": "web_browse" in settings.enabled_tools_list,
            "web_search": {
                "configured": settings.web_search_backend,
                # EFFECTIVE backend: falls back to duckduckgo when the credential
                # is missing, and a number measured with one does not hold for
                # the other
                "effective": getattr(app.state, "web", None).backend
                if hasattr(app.state, "web")
                else None,
                "degraded": app.state.web.degraded if hasattr(app.state, "web") else None,
                "fetch_max_chars": settings.web_fetch_max_chars,
            },
            "knowledge_base": {
                "docs": getattr(app.state, "kb", None).n_docs if hasattr(app.state, "kb") else 0,
                "index_ms": round(app.state.kb.index_ms, 1) if hasattr(app.state, "kb") else None,
                "startup_ms": round(app.state.kb.startup_ms, 1)
                if hasattr(app.state, "kb")
                else None,
            },
            "cache_backend": cache.backend,
            "price": _price_health(settings),
            "levers": {
                "detect_locale": settings.detect_locale,
                "classify_intent": settings.classify_intent,
                "speculative_retrieval": settings.speculative_retrieval,
                "cache_l1": settings.cache_l1_enabled,
                "cache_l2": settings.cache_l2_enabled,
            },
        }

    def _page(name: str, fallback: str) -> str:
        path = STATIC_DIR / name
        if path.exists():
            return path.read_text(encoding="utf-8")
        return fallback

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """ChatLatency: a real chat where every turn carries its own stopwatch,
        token accounting and timeline."""
        return _page(UI_FILE, f"<h1>static/{UI_FILE} not found</h1>")

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page() -> str:
        """Kept as an alias so older links do not break."""
        return _page(UI_FILE, f"<h1>static/{UI_FILE} not found</h1>")

    return app


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


app = build_app()

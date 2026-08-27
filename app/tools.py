"""Agent tools, each one individually timed.

The point of this file is not to have interesting tools - it is to have tools
with **deliberately different latency profiles**, so the timeline shows what an
average hides:

    metric_lookup   in-memory dict                             ~0.03 ms
    latency_budget  pure arithmetic                            ~0.01 ms
    kb_search       local ChromaDB, in-process ONNX embedding   ~20-45 ms
    web_search      Browserbase Search - titles and URLs        ~1 s
    web_fetch       Browserbase Fetch - page as markdown        ~1-3 s
    web_browse      Stagehand - real browser running JavaScript ~10-30 s

Six orders of magnitude between the fastest and the slowest tool. The
conclusion that falls out of it is counter-intuitive and shows up measured in
the findings: **the cost of a tool is rarely the tool.** It is the model hop
that decided to call it.

Every tool returns a `ToolResult`, and the executor records a `tool:<name>` span
with absolute start and end times in the trace - that is what makes it possible
to draw a waterfall and see parallelism instead of just summing durations.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .pricing import get_price_book
from .retrieval import GlossaryTable
from .telemetry import Trace, clip

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class ToolResult:
    """A tool's output: the text that goes back to the model, plus metadata that
    matters to measurement (backend used, hit count) and never to the model."""

    content: str
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ToolCallRecord:
    """One tool execution, positioned in time."""

    name: str
    args: dict[str, Any]
    start_ms: float
    end_ms: float
    ok: bool
    result_chars: int
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # What the tool actually returned, capped. `result_chars` stays the TRUE
    # size: a 12 character result and a 40,000 character result truncated to
    # 4,000 must not look the same in the trace.
    result: str | None = None

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": self.args,
            "start_ms": round(self.start_ms, 2),
            "end_ms": round(self.end_ms, 2),
            "duration_ms": round(self.duration_ms, 2),
            "ok": self.ok,
            "result_chars": self.result_chars,
            "meta": self.meta,
            "error": self.error,
            "result": self.result,
        }


# ---------------------------------------------------------------------------
# 1. Local knowledge base - ChromaDB + in-process ONNX embedding
# ---------------------------------------------------------------------------


class KnowledgeBase:
    """Real local RAG: ChromaDB with an ONNX embedding model (all-MiniLM-L6-v2)
    running **in-process**, with no network call.

    It exists to put a number on a recommendation this harness produced itself:
    a semantic cache with a remote embedding cost ~360 ms per request. With a
    local embedding, embedding **and** vector search together cost ~20-40 ms.
    That is an order of magnitude, measured in the same place.

    Indexing (~1 s for a few dozen documents) happens at startup, never on the
    hot path.
    """

    def __init__(self, collection: str = "kb_local", persist_dir: str | None = None) -> None:
        self.collection_name = collection
        self.persist_dir = persist_dir
        self._col: Any = None
        self.n_docs = 0
        self.index_ms = 0.0
        self.startup_ms = 0.0

    def build(self, documents: list[dict[str, Any]]) -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        t0 = time.perf_counter()
        ef = embedding_functions.ONNXMiniLM_L6_V2()  # type: ignore[attr-defined]
        client = (
            chromadb.PersistentClient(path=self.persist_dir)
            if self.persist_dir
            else chromadb.EphemeralClient()
        )
        # always recreated: the corpus is small, so the index is never stale
        try:
            client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist
            pass
        self._col = client.create_collection(self.collection_name, embedding_function=ef)
        self.startup_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._col.add(
            ids=[d["id"] for d in documents],
            documents=[f"{d['title']}\n{d['text']}" for d in documents],
            metadatas=[
                {
                    "title": d["title"],
                    "type": d.get("type", "concept"),
                    "topic": d.get("topic") or "general",
                    "locale": d.get("locale", "en-US"),
                }
                for d in documents
            ],
        )
        self.index_ms = (time.perf_counter() - t0) * 1000
        self.n_docs = len(documents)

    async def search(
        self, query: str, top_k: int = 3, locale: str | None = None, topic: str | None = None
    ) -> ToolResult:
        if self._col is None:
            return ToolResult(content="", error="knowledge base not initialised")

        clauses = []
        if locale:
            clauses.append({"locale": locale})
        if topic:
            clauses.append({"topic": topic})
        where: dict[str, Any] | None = None
        if len(clauses) == 1:
            where = clauses[0]
        elif clauses:
            where = {"$and": clauses}

        # ChromaDB is synchronous; it goes to a thread so it cannot block the
        # event loop - blocking the loop would distort every parallel measurement.
        res = await asyncio.to_thread(
            self._col.query, query_texts=[query], n_results=top_k, where=where
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        if not docs:
            return ToolResult(content="No document found.", meta={"hits": 0})

        parts = []
        for doc, meta, dist in zip(docs, metas, dists, strict=False):
            parts.append(f"[{meta.get('title', '?')} · distance {float(dist):.3f}] {doc}")
        return ToolResult(
            content="\n\n".join(parts),
            meta={
                "hits": len(docs),
                "top_distance": round(float(dists[0]), 4) if dists else None,
                "titles": [m.get("title") for m in metas],
            },
        )


# ---------------------------------------------------------------------------
# 2. Pure arithmetic - the tool that costs ~zero
# ---------------------------------------------------------------------------


async def latency_budget(
    ttft_ms: float,
    output_tokens: int,
    tokens_per_second: float,
    hops: int = 1,
) -> ToolResult:
    """Turn a set of latency numbers into the shape of the user's experience.

    Pure arithmetic: no network, no disk, no model. It exists so the timeline
    makes it obvious that **a tool can cost 0.01 ms** - and that the time the
    user felt was the model hop that decided to call it.

    The interesting output is not the total. It is the split between the wait
    before anything appears and the stream the user can already read, plus what
    each extra hop costs in perceived latency.
    """
    hops = max(1, int(hops))
    ttft_ms = max(0.0, float(ttft_ms))
    output_tokens = max(0, int(output_tokens))
    tps = float(tokens_per_second) if tokens_per_second else 0.0

    if tps <= 0:
        return ToolResult(
            content="tokens_per_second must be greater than zero.",
            meta={"valid": False},
        )

    # Intermediate hops produce tool calls, not visible text, so each one pays a
    # full prefill before the user sees anything.
    first_visible_ms = ttft_ms * hops
    stream_ms = output_tokens / tps * 1000.0
    total_ms = first_visible_ms + stream_ms
    # An adult reads prose at roughly 5 words/s, ~4 tokens/s per word boundary.
    reading_tokens_per_s = 6.0
    reading_ms = output_tokens / reading_tokens_per_s * 1000.0
    faster_than_reading = tps > reading_tokens_per_s

    lines = [
        f"Hops: {hops} · TTFT per hop: {ttft_ms:.0f} ms · rate: {tps:.0f} tokens/s "
        f"· answer: {output_tokens} tokens.",
        f"First visible token: {first_visible_ms:.0f} ms"
        + (f" ({hops} hops x {ttft_ms:.0f} ms - only the last one emits text)." if hops > 1 else "."),
        f"Streaming the answer: {stream_ms:.0f} ms. Total: {total_ms:.0f} ms.",
        f"Share of the wait spent before any text appears: "
        f"{(first_visible_ms / total_ms * 100):.0f}%.",
        (
            f"Generation is faster than reading ({tps:.0f} vs ~{reading_tokens_per_s:.0f} "
            f"tokens/s), so the user reads for ~{reading_ms:.0f} ms and never waits for "
            "the stream - only the first token is felt as a wait."
            if faster_than_reading
            else f"Generation is slower than reading ({tps:.0f} vs "
            f"~{reading_tokens_per_s:.0f} tokens/s), so the user waits on the stream too."
        ),
    ]
    if hops > 1:
        saved = ttft_ms * (hops - 1)
        lines.append(
            f"Removing {hops - 1} hop(s) would cut {saved:.0f} ms off the first token - "
            f"{(saved / total_ms * 100):.0f}% of the total turn."
        )
    return ToolResult(
        content="\n".join(lines),
        meta={
            "valid": True,
            "first_visible_ms": round(first_visible_ms, 1),
            "stream_ms": round(stream_ms, 1),
            "total_ms": round(total_ms, 1),
            "faster_than_reading": faster_than_reading,
        },
    )


# ---------------------------------------------------------------------------
# 3. The internet - the round trip that dominates the timeline
# ---------------------------------------------------------------------------


class WebSearch:
    """Three ways to touch the internet, with an order of magnitude between them.

    Browserbase exposes distinct primitives for the same goal, and choosing
    between them is a latency decision - exactly the kind of thing this harness
    exists to measure:

        Search  POST /v1/search   ~1 s      titles + URLs, 5-10 KB, NO snippet
        Fetch   POST /v1/fetch    ~1-3 s    page content as markdown
        Browse  Stagehand         ~10-30 s  real browser, act/extract, JS runs

    The detail that changes the design: **Search returns no snippet**, only
    title and URL. An agent that only searches has nothing to read - it needs a
    Fetch afterwards. That turns "look it up on the internet" into two external
    round trips **plus a model hop** to decide which URL to open. It is the hop
    multiplication problem happening again, this time outside the process.

    `duckduckgo` stays as the keyless fallback so the repository runs out of the
    box. If the chosen backend has no credential the code falls back to it and
    the trace records which one actually ran - a number measured with one
    backend cannot be read as a number for the other.
    """

    BB_BASE = "https://api.browserbase.com/v1"
    UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.backend = self._resolve_backend(settings)
        # persistent client: a cold TLS handshake costs ~500 ms and would show
        # up as tool latency, masking the number that matters
        self._client = httpx.AsyncClient(
            timeout=settings.web_search_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": self.UA},
        )
        self._stagehand: Any = None

    @staticmethod
    def _resolve_backend(s: Settings) -> str:
        """Effective backend: falls back to duckduckgo when the key is missing."""
        want = s.web_search_backend
        keys = {
            "browserbase": s.browserbase_api_key,
            "tavily": s.tavily_api_key,
            "brave": s.brave_api_key,
            "serper": s.serper_api_key,
        }
        if want in keys and not keys[want]:
            return "duckduckgo"
        return want

    @property
    def degraded(self) -> bool:
        """True when the requested backend could not be used."""
        return self.backend != self.s.web_search_backend

    # ---------------- Search ----------------

    async def search(self, query: str, top_k: int = 3) -> ToolResult:
        try:
            if self.backend == "browserbase":
                return await self._browserbase_search(query, top_k)
            if self.backend == "tavily":
                return await self._tavily(query, top_k)
            if self.backend == "brave":
                return await self._brave(query, top_k)
            if self.backend == "serper":
                return await self._serper(query, top_k)
            return await self._duckduckgo(query, top_k)
        except httpx.TimeoutException:
            return ToolResult(
                content="The web search timed out.",
                meta={"backend": self.backend},
                error="timeout",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                content="The web search failed.",
                meta={"backend": self.backend},
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _browserbase_search(self, query: str, top_k: int) -> ToolResult:
        """Browserbase Search API.

        `numResults` accepts 1-25 and `query` 1-200 characters; the rate limit
        is 120 req/min per project (429 when exceeded). The response carries
        `results[]` with `id`, `url`, `title` guaranteed and `author`,
        `publishedDate`, `image`, `favicon` optional.
        """
        r = await self._client.post(
            f"{self.BB_BASE}/search",
            headers={
                "x-bb-api-key": self.s.browserbase_api_key or "",
                "Content-Type": "application/json",
            },
            json={"query": query[:200], "numResults": max(1, min(top_k, 25))},
        )
        if r.status_code == 401:
            # measured against the real API: an invalid key returns 401, not the
            # 403 the documentation cites for "search not enabled"
            return ToolResult(
                content="Browserbase key is invalid or missing.",
                meta={"backend": "browserbase", "status": 401},
                error="unauthorized",
            )
        if r.status_code == 403:
            return ToolResult(
                content="Search is not enabled for this Browserbase project.",
                meta={"backend": "browserbase", "status": 403},
                error="search_not_enabled",
            )
        if r.status_code == 429:
            return ToolResult(
                content="Search rate limit reached (120 req/min).",
                meta={"backend": "browserbase", "status": 429},
                error="rate_limited",
            )
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []

        # No snippet: we hand over title + URL and tell the model how to read the
        # page. Pretending there is a summary here would invite a made-up answer.
        lines = []
        for x in results[:top_k]:
            when = x.get("publishedDate")
            stamp = f" ({when[:10]})" if when else ""
            lines.append(f"- {x.get('title', '(no title)')}{stamp}\n  {x.get('url', '')}")
        body = "\n".join(lines) or "No results."
        if results:
            body += (
                "\n\nThese are titles and URLs only - this search returns no snippet. "
                "To read the content of a result, call web_fetch with the URL."
            )
        return ToolResult(
            content=body,
            meta={
                "backend": "browserbase",
                "hits": len(results),
                "request_id": data.get("requestId"),
                "has_snippets": False,
            },
        )

    # ---------------- Fetch ----------------

    async def fetch(self, url: str, max_chars: int | None = None) -> ToolResult:
        """Browserbase Fetch API - a page's content as markdown.

        Browserbase caps at 5 MB and 60 s; `web_fetch_max_chars` caps what
        reaches the model. That cap is not a detail: without it a single turn
        swallows a whole page as input tokens, and that is cost and latency.
        No JavaScript - a page that only renders client-side needs web_browse.
        """
        cap = max_chars or self.s.web_fetch_max_chars
        if not self.s.browserbase_api_key:
            return ToolResult(
                content="web_fetch requires BROWSERBASE_API_KEY.",
                meta={"backend": "browserbase"},
                error="missing_key",
            )
        try:
            r = await self._client.post(
                f"{self.BB_BASE}/fetch",
                headers={
                    "x-bb-api-key": self.s.browserbase_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "url": url,
                    "format": self.s.web_fetch_format,
                    "allowRedirects": True,
                },
            )
            r.raise_for_status()
            data = r.json()
        except httpx.TimeoutException:
            return ToolResult(
                content="Reading the page timed out.",
                meta={"backend": "browserbase", "url": url},
                error="timeout",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                content="Reading the page failed.",
                meta={"backend": "browserbase", "url": url},
                error=f"{type(exc).__name__}: {exc}",
            )

        content = data.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        raw = len(content)
        truncated = raw > cap
        if truncated:
            content = content[:cap] + f"\n\n[... truncated at {cap} of {raw} characters]"

        # TWO statuses, and only one of them was being checked.
        #
        # `raise_for_status` above covers the call to Browserbase, which answers
        # 200 for a fetch it performed correctly. The status of the PAGE comes
        # back inside the payload as `statusCode`, and a 403 there used to
        # produce a tool call with error=None and the content "(empty page)":
        # green in the waterfall, ok=True in the trace, and a model apologising
        # that it could not read the page. The instrument said the tool worked
        # while the turn visibly failed - the one thing a measurement must never
        # do. An unreadable page is an error, and it is reported as one.
        status = data.get("statusCode")
        meta = {
            "backend": "browserbase",
            "url": url,
            "status": status,
            "content_type": data.get("contentType"),
            "chars_raw": raw,
            "chars_sent": len(content),
            "truncated": truncated,
        }
        if isinstance(status, int) and status >= 400:
            return ToolResult(
                content=f"The page answered HTTP {status} and could not be read.",
                meta=meta,
                error=f"http_{status}",
            )
        if not content.strip():
            # 200 with nothing in it: almost always a page that renders
            # client-side, which is exactly what web_browse exists for.
            return ToolResult(
                content=(
                    "The page returned no text. It most likely renders client-side, "
                    "which this fetch does not execute."
                ),
                meta=meta,
                error="empty_page",
            )
        return ToolResult(content=content, meta=meta)

    # ---------------- Browse (Stagehand) ----------------

    async def browse(self, url: str, instruction: str) -> ToolResult:
        """Stagehand: a real browser, with JavaScript, `act` and `extract`.

        An order of magnitude slower than Fetch because it does three expensive
        things in sequence: it creates a cloud browser session, loads the page,
        and runs a model *inside* Stagehand to interpret the instruction.

        The sub-steps are timed separately (`launch_ms`, `create_ms`, `goto_ms`,
        `extract_ms`) because the useful question is not "how much did it cost" -
        it is **which of the three** cost it. A browser session is a fixed cost
        per call and can be amortised with `keep_alive`; the extract is a cost
        per instruction and does not amortise.

        The dependency is imported lazily: whoever does not use the tool pays
        neither the (heavy) import nor the install.
        """
        if not self.s.browserbase_api_key:
            return ToolResult(
                content="web_browse requires BROWSERBASE_API_KEY.",
                meta={"backend": "stagehand"},
                error="missing_key",
            )
        try:
            from stagehand import Stagehand, browserbase  # type: ignore[import-not-found]
        except ImportError:
            return ToolResult(
                content="web_browse requires the stagehand package: uv sync --extra browse",
                meta={"backend": "stagehand"},
                error="stagehand_not_installed",
            )

        marks: dict[str, float] = {}
        browser = None
        sh = None
        try:
            t = time.perf_counter()
            browser = await browserbase.launch(
                api_key=self.s.browserbase_api_key,
                timeout=self.s.stagehand_timeout_s,
            )
            marks["launch_ms"] = round((time.perf_counter() - t) * 1000, 1)

            t = time.perf_counter()
            # model=None lets Stagehand use the Browserbase Model Gateway, so no
            # separate provider key is required.
            sh = await Stagehand.create(
                browser=browser, model=self.s.stagehand_model or None
            )
            marks["create_ms"] = round((time.perf_counter() - t) * 1000, 1)

            t = time.perf_counter()
            pages = await browser.context.pages()
            page = pages[0] if pages else await browser.context.new_page()
            await page.goto(url)
            marks["goto_ms"] = round((time.perf_counter() - t) * 1000, 1)

            t = time.perf_counter()
            # With no schema, Stagehand uses DefaultExtract, whose only field is
            # `extraction: str` - which is what we want for free text.
            res = await sh.extract(instruction)
            marks["extract_ms"] = round((time.perf_counter() - t) * 1000, 1)

            # Stagehand runs a model INSIDE the tool to interpret the
            # instruction. Those tokens are paid for by the turn but never show
            # up in our provider's usage - they are INVISIBLE cost if nobody
            # looks. Here they come out of the metadata and become a number in
            # the trace.
            hidden: dict[str, Any] = {}
            md = getattr(res, "metadata", None)
            u = getattr(md, "usage", None)
            if u is not None:
                hidden = {
                    "llm_input_tokens": getattr(u, "input_tokens", None),
                    "llm_output_tokens": getattr(u, "output_tokens", None),
                    "llm_cached_input_tokens": getattr(u, "cached_input_tokens", None),
                    "llm_inference_ms": getattr(u, "inference_time_ms", None),
                }
                cache_info = getattr(md, "cache", None)
                if cache_info is not None:
                    hidden["llm_cache_status"] = getattr(cache_info, "status", None)
                # Price those tokens at the rate of the model STAGEHAND runs
                # (STAGEHAND_MODEL), not the engine's frontier tier. They are
                # another model's tokens on another price sheet; charging them
                # at the frontier rate was silently wrong.
                ins = hidden.get("llm_input_tokens") or 0
                outs = hidden.get("llm_output_tokens") or 0
                sp = get_price_book(self.s).resolve(self.s.stagehand_model)
                hidden["llm_price_model"] = sp.matched or sp.model
                hidden["llm_cost"] = round(
                    ins / 1e6 * sp.in_per_mtok + outs / 1e6 * sp.out_per_mtok,
                    6,
                )

            data = getattr(res, "data", None)
            text = getattr(data, "extraction", None)
            if text is None:
                text = (
                    json.dumps(data, ensure_ascii=False, default=str)
                    if data is not None
                    else ""
                )
            cap = self.s.web_fetch_max_chars
            raw = len(text)
            return ToolResult(
                content=text[:cap] or "(nothing extracted)",
                meta={
                    "backend": "stagehand",
                    "url": url,
                    "model": self.s.stagehand_model or "gateway-default",
                    "session_id": getattr(browser, "session_id", None),
                    "chars_raw": raw,
                    "truncated": raw > cap,
                    **marks,
                    **hidden,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                content="Browsing failed.",
                meta={"backend": "stagehand", "url": url, **marks},
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            # Stagehand.close is async; StagehandBrowser.close is NOT - the
            # documented example awaits both, which breaks on the second.
            # Verified by introspection: iscoroutinefunction(browser.close) is False.
            if sh is not None:
                with contextlib.suppress(Exception):
                    await sh.close()
            if browser is not None:
                with contextlib.suppress(Exception):
                    maybe = browser.close()
                    if inspect.isawaitable(maybe):
                        await maybe

    # ---------------- keyless / alternative backends ----------------

    async def _duckduckgo(self, query: str, top_k: int) -> ToolResult:
        r = await self._client.post("https://html.duckduckgo.com/html/", data={"q": query})
        r.raise_for_status()
        titles = re.findall(r'result__a"[^>]*>(.*?)</a>', r.text, re.S)
        snippets = re.findall(r'result__snippet"[^>]*>(.*?)</a>', r.text, re.S)
        strip = lambda h: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", h)).strip()  # noqa: E731
        items = []
        for i, t in enumerate(titles[:top_k]):
            sn = strip(snippets[i]) if i < len(snippets) else ""
            items.append(f"- {strip(t)}: {sn[:280]}")
        if not items:
            return ToolResult(
                content="No results.", meta={"backend": "duckduckgo", "hits": 0}
            )
        return ToolResult(
            content="\n".join(items),
            meta={
                "backend": "duckduckgo",
                "hits": len(items),
                "bytes": len(r.content),
                "has_snippets": True,
            },
        )

    async def _tavily(self, query: str, top_k: int) -> ToolResult:
        r = await self._client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.s.tavily_api_key,
                "query": query,
                "max_results": top_k,
                "search_depth": "basic",
            },
        )
        r.raise_for_status()
        items = [
            f"- {x.get('title', '')}: {(x.get('content') or '')[:280]}"
            for x in r.json().get("results", [])[:top_k]
        ]
        return ToolResult(
            content="\n".join(items) or "No results.",
            meta={"backend": "tavily", "hits": len(items), "has_snippets": True},
        )

    async def _brave(self, query: str, top_k: int) -> ToolResult:
        r = await self._client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": top_k},
            headers={
                "X-Subscription-Token": self.s.brave_api_key or "",
                "Accept": "application/json",
            },
        )
        r.raise_for_status()
        items = [
            f"- {x.get('title', '')}: {(x.get('description') or '')[:280]}"
            for x in r.json().get("web", {}).get("results", [])[:top_k]
        ]
        return ToolResult(
            content="\n".join(items) or "No results.",
            meta={"backend": "brave", "hits": len(items), "has_snippets": True},
        )

    async def _serper(self, query: str, top_k: int) -> ToolResult:
        r = await self._client.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": top_k},
            headers={"X-API-KEY": self.s.serper_api_key or ""},
        )
        r.raise_for_status()
        items = [
            f"- {x.get('title', '')}: {(x.get('snippet') or '')[:280]}"
            for x in r.json().get("organic", [])[:top_k]
        ]
        return ToolResult(
            content="\n".join(items) or "No results.",
            meta={"backend": "serper", "hits": len(items), "has_snippets": True},
        )

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Registry + timed executor
# ---------------------------------------------------------------------------


def tool_schemas(metric_ids: list[str], enabled: set[str]) -> list[dict[str, Any]]:
    """Schemas in OpenAI tool-calling format.

    The descriptions are written for routing: they say explicitly **when not**
    to use the tool. That is the difference between an agent that calls the
    internet to answer a definition and one that does not.
    """
    ids = ", ".join(metric_ids)
    all_tools = {
        "kb_search": {
            "type": "function",
            "function": {
                "name": "kb_search",
                "description": (
                    "Search the indexed documentation about latency, streaming, RAG, "
                    "agents, caching and cost. Use it for how-something-works, "
                    "trade-offs, guides and pitfalls. Do NOT use it for the exact value "
                    "of a metric attribute - use metric_lookup for that."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Question in natural language."},
                        "top_k": {"type": "integer", "description": "Passages to return (1-5).", "default": 3},
                    },
                    "required": ["query"],
                },
            },
        },
        "metric_lookup": {
            "type": "function",
            "function": {
                "name": "metric_lookup",
                "description": (
                    "Look up an exact attribute of a latency metric. Instant and exact. "
                    f"Metrics: {ids}. Attributes: unit, measures, how_to_measure, "
                    "good_target, dominated_by, common_mistake. Always prefer this tool "
                    "over searching for the value in prose."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "description": f"One of: {ids}"},
                        "attribute": {
                            "type": "string",
                            "description": "A specific attribute. Omit to get the whole entry.",
                        },
                    },
                    "required": ["metric"],
                },
            },
        },
        "latency_budget": {
            "type": "function",
            "function": {
                "name": "latency_budget",
                "description": (
                    "Compute what a set of latency numbers feels like: when the first "
                    "token appears, how long the stream lasts, how much of the wait is "
                    "spent before any text shows up, and what an extra model hop costs. "
                    "Use it whenever the question involves sizing, a budget, or a "
                    "before/after comparison in milliseconds."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ttft_ms": {
                            "type": "number",
                            "description": "Time to first token of a single model hop, in ms.",
                        },
                        "output_tokens": {
                            "type": "integer",
                            "description": "Length of the answer in output tokens.",
                        },
                        "tokens_per_second": {
                            "type": "number",
                            "description": "Generation rate in output tokens per second.",
                        },
                        "hops": {
                            "type": "integer",
                            "description": "Model round trips in the turn. Default 1.",
                            "default": 1,
                        },
                    },
                    "required": ["ttft_ms", "output_tokens", "tokens_per_second"],
                },
            },
        },
        "web_search": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the public internet. Returns ONLY titles and URLs - no "
                    "summary of the content. To read a result, call web_fetch with the "
                    "URL. Use it only when the question is about something outside the "
                    "indexed documentation (news, a specific vendor, a current "
                    "benchmark). Never use it for a concept that is documented "
                    "internally or for a metric attribute."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search terms."},
                        "top_k": {
                            "type": "integer",
                            "description": "Results (1-25).",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        "web_fetch": {
            "type": "function",
            "function": {
                "name": "web_fetch",
                "description": (
                    "Read the content of a URL and return it as markdown. Use it after a "
                    "web_search, with the most promising URL, when you need the content "
                    "and not just the title. It does not run JavaScript: if the page "
                    "comes back empty, the content probably only renders in a browser."
                ),
                # `max_chars` used to be offered here and was withdrawn on
                # evidence. Asked for a price on a documentation page, the model
                # called web_fetch with max_chars=1000, received 1,045
                # characters of navigation boilerplate, and answered that the
                # page was truncated and it could not find the price. It capped
                # itself below the useful content and then reported the
                # consequence as a limitation of the page. The cap is a cost
                # lever that belongs to the operator (WEB_FETCH_MAX_CHARS), and
                # the trace already reports chars_raw against chars_sent.
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Full URL to read."},
                    },
                    "required": ["url"],
                },
            },
        },
        "web_browse": {
            "type": "function",
            "function": {
                "name": "web_browse",
                "description": (
                    "Open the URL in a real browser, with JavaScript, and extract "
                    "whatever the instruction asks for. It is the SLOWEST tool in the "
                    "box by an order of magnitude (10-30 s). Use it only when web_fetch "
                    "already failed because the page depends on JavaScript, or when the "
                    "page must be interacted with."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to open."},
                        "instruction": {
                            "type": "string",
                            "description": "What to extract, in natural language.",
                        },
                    },
                    "required": ["url", "instruction"],
                },
            },
        },
    }
    return [v for k, v in all_tools.items() if k in enabled]


class ToolBox:
    """Runs tools by name, timing each one into the trace."""

    def __init__(
        self,
        settings: Settings,
        kb: KnowledgeBase,
        table: GlossaryTable,
        web: WebSearch,
    ) -> None:
        self.s = settings
        self.kb = kb
        self.table = table
        self.web = web

    @property
    def enabled(self) -> set[str]:
        return set(self.s.enabled_tools_list)

    def schemas(self) -> list[dict[str, Any]]:
        return tool_schemas(self.table.ids(), self.enabled)

    async def _dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name == "kb_search":
            return await self.kb.search(
                args.get("query", ""),
                int(args.get("top_k") or self.s.retrieval_top_k),
                args.get("locale"),
                args.get("topic"),
            )
        if name == "metric_lookup":
            found = await self.table.lookup(args.get("metric", ""), args.get("attribute"))
            content = str(found.pop("content", ""))
            return ToolResult(content=content, meta=found)
        if name == "latency_budget":
            return await latency_budget(
                float(args.get("ttft_ms") or 0),
                int(args.get("output_tokens") or 0),
                float(args.get("tokens_per_second") or 0),
                int(args.get("hops") or 1),
            )
        if name == "web_search":
            return await self.web.search(args.get("query", ""), int(args.get("top_k") or 3))
        if name == "web_fetch":
            # No max_chars from the model: see the schema above. The operator's
            # WEB_FETCH_MAX_CHARS is the only cap.
            return await self.web.fetch(args.get("url", ""))
        if name == "web_browse":
            return await self.web.browse(
                args.get("url", ""), args.get("instruction", "Extract the main content.")
            )
        return ToolResult(content=f"Tool '{name}' does not exist.", error="unknown_tool")

    def _payload(self, content: str) -> str | None:
        """The tool's own output, for the trace. None when capture is off."""
        if not self.s.trace_payloads:
            return None
        return clip(content, self.s.trace_payload_chars)

    async def run(self, name: str, args: dict[str, Any], trace: Trace) -> ToolCallRecord:
        """Runs one tool inside a `tool:<name>` span.

        The span stores absolute start and end, not just a duration - that is
        what makes it possible to draw the waterfall and see that two tools
        overlapped.
        """
        start = trace.now_ms()
        async with trace.aspan(f"tool:{name}"):
            try:
                res = await self._dispatch(name, args)
            except Exception as exc:  # noqa: BLE001
                res = ToolResult(content="", error=f"{type(exc).__name__}: {exc}")
        end = trace.now_ms()

        return ToolCallRecord(
            name=name,
            args=args,
            start_ms=start,
            end_ms=end,
            ok=res.error is None,
            result_chars=len(res.content),
            meta=res.meta,
            error=res.error,
            result=self._payload(res.content),
        )

    async def run_many(
        self, calls: list[tuple[str, str, dict[str, Any]]], trace: Trace
    ) -> list[tuple[str, ToolCallRecord, str]]:
        """Runs the tools requested in one hop concurrently.

        Parallelising is both the right decision and the most interesting one to
        measure: the waterfall shows an 800 ms `web_search` completely covering a
        0.01 ms `metric_lookup`, and the cost of the step becomes the slowest
        tool rather than the sum.
        """

        async def one(name: str, args: dict[str, Any]) -> tuple[ToolCallRecord, ToolResult]:
            start = trace.now_ms()
            async with trace.aspan(f"tool:{name}"):
                try:
                    res = await self._dispatch(name, args)
                except Exception as exc:  # noqa: BLE001
                    res = ToolResult(content="", error=f"{type(exc).__name__}: {exc}")
            rec = ToolCallRecord(
                name=name,
                args=args,
                start_ms=start,
                end_ms=trace.now_ms(),
                ok=res.error is None,
                result_chars=len(res.content),
                meta=res.meta,
                error=res.error,
                result=self._payload(res.content),
            )
            return rec, res

        pairs = await asyncio.gather(*(one(name, args) for _, name, args in calls))
        out = []
        for (call_id, _, _), (rec, res) in zip(calls, pairs, strict=True):
            body = res.content if res.error is None else f"ERROR: {res.error}"
            out.append((call_id, rec, body))
        return out


def load_corpus(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or DATA_DIR / "corpus.json").read_text(encoding="utf-8"))

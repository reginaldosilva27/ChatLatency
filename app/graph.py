"""The agent graph in LangGraph, in two comparable topologies.

**Agentic** (`AGENTIC=true`, default) - the model picks the tools:

    cache - miss - agent --tool_calls--> tools --+
                     |                            | (loop, up to MAX_TOOL_HOPS)
                     |<---------------------------+
                     +-- text --> streamed to the user

**Fixed** (`AGENTIC=false`) - a deterministic pipeline, no tool decision:

    cache - miss - [locale] -+- intent ----+
                             +- retrieval -+- route - generate

Both exist in the same process on purpose: comparing them is the most useful
measurement this engine makes. The fixed pipeline has **one** round trip to the
model; the agentic one has at least **two** whenever it uses a tool. The whole
difference lands on the time to the first word.

Every hop is streamed (see `LLM.stream_with_tools`), so if the model decides to
answer without a tool, the first token comes out on the first hop and the TTFT
is real - not inflated by a non-streaming round trip whose only job was to
decide about tools.

Streaming: the node pushes chunks into an `asyncio.Queue` that the SSE endpoint
reads. That is deliberately more direct than `astream_events` - in a latency
instrument, no layer belongs between the socket and the stopwatch.
"""

from __future__ import annotations

import asyncio
import json
import re
from functools import lru_cache
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .cache import CacheEntry, LayeredCache, l1_key
from .config import Settings
from .llm import LLM, StreamOutcome, Tier, Usage
from .retrieval import Chunk, GlossaryTable, Retriever
from .telemetry import Trace
from .tools import ToolBox

SENTINEL_DONE = object()

SYSTEM_PROMPT = (
    "You are ChatLatency, an assistant that explains how conversational AI systems "
    "behave: latency, streaming, retrieval augmented generation, agent loops, caching "
    "and cost. Answer in the language the user wrote in, in at most 4 sentences, "
    "direct and practical. Use only information coming from the tools or the provided "
    "context. If there is not enough information, say so instead of guessing. Never "
    "invent a number, a threshold or a benchmark result."
)

TOOL_SYSTEM_PROMPT = (
    SYSTEM_PROMPT + " You have tools available, in increasing order of cost: "
    "metric_lookup (instant) for an exact metric attribute; latency_budget (instant) "
    "for sizing and before/after arithmetic; kb_search (fast) for the indexed "
    "documentation; web_search (~1s) only for a subject OUTSIDE that documentation; "
    "web_fetch (~2s) to read a URL that web_search returned; web_browse (~20s) only if "
    "web_fetch failed because the page depends on JavaScript. "
    "Always choose the cheapest tool that resolves the question, and request all "
    "mutually independent tools in the same step. Ground every answer: for an exact "
    "attribute call metric_lookup, and for a concept, mechanism or trade-off call "
    "kb_search - do not answer those from memory, even when you are confident. Only "
    "greetings, meta-questions about this conversation, and arithmetic you can hand to "
    "latency_budget may skip the tools."
)

INTENT_LABELS = [
    "metric_attribute",
    "concept_question",
    "budget_calculation",
    "external_subject",
    "complaint",
    "other",
]

INTENT_PROMPT = (
    "Classify the user's message into exactly one label from this list: "
    + ", ".join(INTENT_LABELS)
    + ". Answer with the label only, no punctuation and no explanation."
)


def _merge_dict(left: dict, right: dict) -> dict:
    """Reducer so concurrent branches of the graph can write the same field."""
    return {**left, **right}


class AgentInput(TypedDict):
    """Keys always present in the graph input."""

    question: str
    locale: str


class AgentState(AgentInput, total=False):
    # optional input
    topic: str | None  # corpus topic, when the channel already knows it
    user_context: str | None
    history: list[dict[str, str]]

    # derived
    intent: str
    chunks: list[Chunk]
    tier: Tier
    fixed_fact: str | None
    cache_tier: Literal["l1", "l2", "miss"]
    cached_answer: str | None
    cache_similarity: float | None
    answer: str
    output_tokens: int

    # agent loop
    messages: list[dict[str, Any]]
    hop: int
    tool_records: list[dict[str, Any]]
    scratch: Annotated[dict[str, Any], _merge_dict]


class AgentRuntime:
    """Dependencies resolved once at startup; the compiled graphs are reused
    across requests (compiling per request would add milliseconds that do not
    exist in production)."""

    def __init__(
        self,
        settings: Settings,
        llm: LLM,
        cache: LayeredCache,
        retriever: Retriever,
        table: GlossaryTable,
        toolbox: ToolBox | None = None,
    ) -> None:
        self.s = settings
        self.llm = llm
        self.cache = cache
        self.retriever = retriever
        self.table = table
        self.toolbox = toolbox
        self._graphs: dict[tuple, Any] = {}

    # ---------------------- cache ----------------------

    async def n_cache_lookup(self, state: AgentState, config) -> dict[str, Any]:
        trace: Trace = config["configurable"]["trace"]
        opts = config["configurable"]["opts"]
        question, locale, topic = state["question"], state["locale"], state.get("topic")

        if opts["cache_l1"]:
            key = l1_key(question, topic, locale)
            async with trace.aspan("cache_l1"):
                hit = await self.cache.get_l1(key)
            if hit:
                return {"cache_tier": "l1", "cached_answer": hit["answer"]}

        if opts["cache_l2"]:
            # L2 costs an embedding on the hot path - measured separately.
            async with trace.aspan("cache_l2_embed"):
                vector = await self.llm.embed(question)
            async with trace.aspan("cache_l2_search"):
                found = await self.cache.search_l2(vector, locale, topic)
            if found:
                entry, sim = found
                return {
                    "cache_tier": "l2",
                    "cached_answer": entry["answer"],
                    "cache_similarity": sim,
                    "scratch": {"l2_vector": vector},
                }
            return {"cache_tier": "miss", "scratch": {"l2_vector": vector}}

        return {"cache_tier": "miss"}

    async def n_serve_cached(self, state: AgentState, config) -> dict[str, Any]:
        """Serving a cache hit. By default it sends the answer at once (which
        measures the real latency floor); CACHED_REPLAY_TOKENS_PER_S > 0 slices
        the delivery only so the visual demo resembles model streaming."""
        trace: Trace = config["configurable"]["trace"]
        emit: asyncio.Queue = config["configurable"]["emit"]
        answer = state.get("cached_answer") or ""
        rate = self.s.cached_replay_tokens_per_s

        if rate <= 0:
            trace.mark("first_token")
            await emit.put(answer)
            trace.mark("last_token")
        else:
            delay = 1.0 / rate
            for i, word in enumerate(answer.split(" ")):
                if i == 0:
                    trace.mark("first_token")
                await emit.put(word + " ")
                await asyncio.sleep(delay)
            trace.mark("last_token")

        trace.set(
            tier="cache",
            intent=state.get("intent"),
            usage_source="cache",
            input_tokens=0,
            output_tokens=_count_tokens(answer),
            cached_input_tokens=0,
            llm_hops=0,
            cost_total=0.0,
            currency=self.s.price_currency,
        )
        return {"answer": answer, "output_tokens": _count_tokens(answer)}

    # ---------------------- agent loop ----------------------

    def _base_messages(self, state: AgentState, system: str) -> list[dict[str, Any]]:
        # A stable system prompt at the start of the payload is a prerequisite
        # for the provider's prompt cache. Anything variable goes after it.
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if state.get("user_context"):
            messages.append(
                {"role": "system", "content": f"USER CONTEXT:\n{state['user_context']}"}
            )
        if state.get("topic"):
            messages.append(
                {"role": "system", "content": f"The user is asking about {state['topic']}."}
            )
        messages += list(state.get("history") or [])
        messages.append({"role": "user", "content": state["question"]})
        return messages

    async def n_agent(self, state: AgentState, config) -> dict[str, Any]:
        """One model hop, streamed, with tools available.

        Each run of this node opens a `hop:N` span. The first hop that produces
        text marks `first_token` - and that is where the real cost of an agentic
        architecture becomes visible: if the model asked for a tool on hop 1, the
        user only sees the first word after hop 2.
        """
        trace: Trace = config["configurable"]["trace"]
        emit: asyncio.Queue = config["configurable"]["emit"]
        opts = config["configurable"]["opts"]

        hop = int(state.get("hop") or 0) + 1
        messages = list(state.get("messages") or self._base_messages(state, TOOL_SYSTEM_PROMPT))
        usage: Usage = (state.get("scratch") or {}).get("usage") or Usage()

        # On the last hop the tools are withdrawn: without that the model can ask
        # for a tool again and the turn ends with no answer.
        tools = (
            self.toolbox.schemas()
            if (self.toolbox and hop < self.s.max_tool_hops)
            else None
        )
        outcome = StreamOutcome()
        tier: Tier = state.get("tier") or opts.get("force_tier") or "mini"

        pieces: list[str] = []
        span = trace.span(f"hop:{hop}")
        span.__enter__()
        stream_span = None
        first = True
        try:
            async for delta in self.llm.stream_with_tools(
                messages, tools, outcome, tier=tier, usage_out=usage
            ):
                if first:
                    trace.mark("first_token")
                    first = False
                    stream_span = trace.span("model_stream")
                    stream_span.__enter__()
                pieces.append(delta)
                await emit.put(delta)
        except Exception as exc:  # noqa: BLE001
            trace.set(error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if stream_span is not None:
                stream_span.__exit__(None, None, None)
            span.__exit__(None, None, None)

        scratch: dict[str, Any] = {"usage": usage}

        # The model asked for tools: this hop's text (if any) is a preamble and
        # is not emitted - emitting it would inflate TTFT, which must mean "first
        # word of the final answer".
        if outcome.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": outcome.text or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(c["args"], ensure_ascii=False),
                            },
                        }
                        for c in outcome.tool_calls
                    ],
                }
            )
            trace.set(tier=tier)
            return {
                "hop": hop,
                "messages": messages,
                "tier": tier,
                "scratch": scratch | {"pending": outcome.tool_calls},
            }

        # No tools: this hop is the answer.
        answer = "".join(pieces) or outcome.text
        trace.mark("last_token")
        trace.set(tier=tier, llm_hops_graph=hop)
        await self._finalize(state, config, answer, usage, hops=hop)
        return {
            "hop": hop,
            "messages": messages,
            "tier": tier,
            "answer": answer,
            "output_tokens": usage.output_tokens,
            "scratch": scratch | {"pending": []},
        }

    async def n_tools(self, state: AgentState, config) -> dict[str, Any]:
        """Runs the tools requested by the previous hop concurrently.

        Parallelising is the right decision and also the most interesting one to
        measure: the waterfall shows an ~800 ms `web_search` fully covering a
        0.01 ms `metric_lookup`, and the cost of the step becomes the slowest
        tool rather than the sum.
        """
        trace: Trace = config["configurable"]["trace"]
        assert self.toolbox is not None

        pending: list[dict[str, Any]] = (state.get("scratch") or {}).get("pending") or []
        messages = list(state.get("messages") or [])
        calls = [(c["id"], c["name"], c["args"]) for c in pending]

        results = await self.toolbox.run_many(calls, trace)

        records: list[dict[str, Any]] = list(state.get("tool_records") or [])
        for call_id, rec, body in results:
            records.append(rec.to_dict() | {"hop": state.get("hop")})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": body})

        trace.set(tool_calls=records)
        return {"messages": messages, "tool_records": records, "scratch": {"pending": []}}

    def _route_after_agent(self, state: AgentState) -> str:
        pending = (state.get("scratch") or {}).get("pending") or []
        if pending and int(state.get("hop") or 0) < self.s.max_tool_hops:
            return "tools"
        return END

    # ---------------------- fixed pipeline (comparison) ----------------------

    async def n_locale(self, state: AgentState, config) -> dict[str, Any]:
        """Language/region detection by LLM.

        Off by default: when the channel already reports the locale (web, app),
        this step is a wasted network call on the critical path. It only earns
        its place as a fallback in channels that do not carry that metadata."""
        trace: Trace = config["configurable"]["trace"]
        async with trace.aspan("locale"):
            detected = await self.llm.complete(
                [
                    {
                        "role": "system",
                        "content": "Reply with only the BCP-47 language/region code of the "
                        "message (e.g. pt-BR, es-CL, en-US). The code only.",
                    },
                    {"role": "user", "content": state["question"]},
                ],
                tier="nano",
                max_tokens=8,
            )
        match = re.search(r"[a-z]{2}-[A-Z]{2}", detected or "")
        return {"locale": match.group(0) if match else state["locale"]}

    async def n_intent(self, state: AgentState, config) -> dict[str, Any]:
        trace: Trace = config["configurable"]["trace"]
        opts = config["configurable"]["opts"]
        if not opts["classify_intent"]:
            return {"intent": "unknown"}

        async with trace.aspan("intent"):
            raw = await self.llm.complete(
                [
                    {"role": "system", "content": INTENT_PROMPT},
                    {"role": "user", "content": state["question"]},
                ],
                tier="nano",
                max_tokens=8,
            )
        label = (raw or "").strip().lower().strip(".")
        return {"intent": label if label in INTENT_LABELS else "other"}

    async def n_retrieval(self, state: AgentState, config) -> dict[str, Any]:
        trace: Trace = config["configurable"]["trace"]

        # Deterministic lookup first: an exact attribute is not a RAG problem.
        fixed = self.table.resolve_fixed_fact(state["question"], state.get("topic"))

        async with trace.aspan("retrieval"):
            chunks = await self.retriever.search(
                state["question"], state["locale"], state.get("topic"), self.s.retrieval_top_k
            )
        return {"chunks": chunks, "fixed_fact": fixed}

    async def n_route(self, state: AgentState, config) -> dict[str, Any]:
        """Tier router. Hot-path rule: always non-reasoning; frontier only when
        the question actually needs it."""
        trace: Trace = config["configurable"]["trace"]
        opts = config["configurable"]["opts"]
        with trace.span("route"):
            forced = opts.get("force_tier")
            if forced:
                tier: Tier = forced
            else:
                intent = state.get("intent", "other")
                chunks = state.get("chunks") or []
                long_question = len(state["question"].split()) > 28
                weak_context = (not chunks) or (chunks[0].score < 1.0)
                tier = (
                    "frontier"
                    if (intent == "complaint" or long_question or weak_context)
                    else "mini"
                )
        return {"tier": tier}

    async def n_generate(self, state: AgentState, config) -> dict[str, Any]:
        trace: Trace = config["configurable"]["trace"]
        emit: asyncio.Queue = config["configurable"]["emit"]

        chunks: list[Chunk] = state.get("chunks") or []
        fixed = state.get("fixed_fact")
        parts = []
        if fixed:
            parts.append(f"[exact attribute] {fixed}")
        parts += [c.as_context() for c in chunks]
        context = "\n\n".join(parts) or "(nothing retrieved)"

        messages = self._base_messages(state, SYSTEM_PROMPT)
        messages[-1] = {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION: {state['question']}",
        }

        tier: Tier = state.get("tier", "mini")
        trace.set(tier=tier, intent=state.get("intent"), chunk_ids=[c.id for c in chunks])

        pieces: list[str] = []
        usage = Usage()
        ttft_span = trace.span("model_ttft")
        ttft_span.__enter__()
        first = True
        stream_span = None
        try:
            async for delta in self.llm.stream(messages, tier=tier, usage_out=usage):
                if first:
                    ttft_span.__exit__(None, None, None)
                    trace.mark("first_token")
                    stream_span = trace.span("model_stream")
                    stream_span.__enter__()
                    first = False
                pieces.append(delta)
                await emit.put(delta)
        except Exception as exc:  # noqa: BLE001
            if first:
                ttft_span.__exit__(None, None, None)
            trace.set(error=f"{type(exc).__name__}: {exc}")
            raise
        if first:
            ttft_span.__exit__(None, None, None)
        elif stream_span is not None:
            stream_span.__exit__(None, None, None)
        trace.mark("last_token")

        answer = "".join(pieces)
        await self._finalize(state, config, answer, usage, hops=1)
        return {"answer": answer, "output_tokens": usage.output_tokens}

    # ---------------------- shared ----------------------

    async def _finalize(
        self, state: AgentState, config, answer: str, usage: Usage, hops: int
    ) -> None:
        """Records usage/cost and writes to the cache. Called by whichever node
        produced the answer, agentic or fixed."""
        trace: Trace = config["configurable"]["trace"]
        opts = config["configurable"]["opts"]

        model = self.llm.model_for(state.get("tier") or "mini")
        if usage.output_tokens:
            trace.set(usage_source="provider", **usage.cost(self.s, model))
            if usage.latency_checkpoint:
                trace.set(azure_latency=usage.latency_checkpoint)
        else:
            usage.output_tokens = _count_tokens(answer)
            usage.hops = max(usage.hops, hops)
            trace.set(usage_source="estimated (tiktoken)", **usage.cost(self.s, model))

        if opts["cache_l1"] and answer:
            key = l1_key(state["question"], state.get("topic"), state["locale"])
            await self.cache.set_l1(
                key,
                CacheEntry(
                    answer=answer,
                    topic=state.get("topic"),
                    locale=state["locale"],
                    question=state["question"],
                ),
            )
            vector = (state.get("scratch") or {}).get("l2_vector")
            if opts["cache_l2"] and vector:
                await self.cache.index_l2(key, vector, state["locale"], state.get("topic"))

    # ---------------------- assembly ----------------------

    def graph(
        self,
        speculative: bool,
        detect_locale: bool,
        use_intent: bool,
        agentic: bool | None = None,
    ):
        agentic = self.s.agentic if agentic is None else agentic
        if agentic and self.toolbox is None:
            agentic = False
        key = (speculative, detect_locale, use_intent, agentic)
        if key in self._graphs:
            return self._graphs[key]

        g = StateGraph(AgentState)
        g.add_node("cache_lookup", self.n_cache_lookup)
        g.add_node("serve_cached", self.n_serve_cached)
        g.add_edge(START, "cache_lookup")
        g.add_edge("serve_cached", END)

        if agentic:
            g.add_node("agent", self.n_agent)
            g.add_node("tools", self.n_tools)
            g.add_conditional_edges(
                "cache_lookup",
                lambda s: "serve_cached" if s.get("cache_tier") in ("l1", "l2") else "agent",
                ["serve_cached", "agent"],
            )
            g.add_conditional_edges("agent", self._route_after_agent, ["tools", END])
            g.add_edge("tools", "agent")
        else:
            g.add_node("route", self.n_route)
            g.add_node("generate", self.n_generate)
            g.add_node("retrieval", self.n_retrieval)
            if detect_locale:
                g.add_node("locale", self.n_locale)
            if use_intent:
                g.add_node("intent", self.n_intent)

            if detect_locale:
                work_entry: list[str] = ["locale"]
            elif speculative and use_intent:
                work_entry = ["intent", "retrieval"]
            elif use_intent:
                work_entry = ["intent"]
            else:
                work_entry = ["retrieval"]

            def _after_cache(s: AgentState) -> list[str]:
                if s.get("cache_tier") in ("l1", "l2"):
                    return ["serve_cached"]
                return work_entry

            g.add_conditional_edges(
                "cache_lookup", _after_cache, ["serve_cached", *set(work_entry)]
            )
            if speculative and use_intent:
                if detect_locale:
                    g.add_edge("locale", "intent")
                    g.add_edge("locale", "retrieval")
                g.add_edge("intent", "route")
                g.add_edge("retrieval", "route")
            else:
                if detect_locale:
                    g.add_edge("locale", "intent" if use_intent else "retrieval")
                if use_intent:
                    g.add_edge("intent", "retrieval")
                g.add_edge("retrieval", "route")
            g.add_edge("route", "generate")
            g.add_edge("generate", END)

        compiled = g.compile()
        self._graphs[key] = compiled
        return compiled


@lru_cache(maxsize=1)
def _encoder():
    """tiktoken encoder (o200k_base covers the gpt-4o/gpt-5 family). Loaded
    once; counting per answer costs microseconds and stays OUT of the measured
    path - it happens after last_token."""
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:  # noqa: BLE001
        return None


def _count_tokens(text: str) -> int:
    """Real output tokens, so tokens/s is a number and not an estimate.
    Falls back to ~4 chars/token if tiktoken is unavailable."""
    enc = _encoder()
    if enc is None:
        return max(1, round(len(text) / 4))
    return max(1, len(enc.encode(text)))


def warm_tokenizer() -> None:
    """Called at startup: tiktoken's first get_encoding may download the BPE
    file, and that must not land inside a measured request."""
    _count_tokens("warmup")

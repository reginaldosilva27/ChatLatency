"""The model layer, per tier (nano / mini / frontier).

It talks directly to the async OpenAI/Azure SDK instead of going through the
LangChain wrapper on the hot path: TTFT has to be measured with no abstraction
layer between the socket and the stopwatch.

Two responsibilities that are not obvious:

1. **A compatibility shim per model family.** gpt-5.x and the o-series require
   `max_completion_tokens`, reject any `temperature` other than 1 and accept
   `reasoning_effort`; gpt-4.x uses `max_tokens` and accepts `temperature`.
   Sending the wrong parameter does not degrade gracefully - it returns 400.

2. **Capturing the real `usage`.** Cost per interaction cannot come from a
   character estimate. `stream_options={"include_usage": True}` makes the
   provider send a final chunk with input, output and **cached** token counts
   (the provider's prompt cache, measured instead of assumed).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from openai import AsyncAzureOpenAI, AsyncOpenAI

from .config import Settings
from .pricing import ModelPrice, get_price_book

Tier = Literal["nano", "mini", "frontier"]

# Synthetic text for the mock provider: an English paragraph of ~250 tokens.
_MOCK_WORDS = (
    "Time to first token is the moment the user starts reading, and everything that "
    "runs before the model is called is paid in full before a single character "
    "appears on screen, which is why topology matters more than micro-optimisation. "
).split()


@dataclass
class StreamOutcome:
    """What one streaming hop produced.

    A hop can end in two ways: with text (that is the answer) or with tool_calls
    (the model wants data before answering). Streaming EVERY hop - instead of
    using a non-streaming call to decide about tools - matters for the
    measurement: if the model chooses to answer directly, the first token comes
    out on the first hop and the TTFT is real, not inflated by an extra round
    trip.
    """

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""
    finish_reason: str | None = None


@dataclass
class Usage:
    """A turn's real consumption, from the provider - never estimated."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    hops: int = 0  # how many model calls this turn cost
    # The deployment that produced this consumption. It is what allows each
    # tier (and the third-party model inside Stagehand) to be billed at ITS
    # own price rather than at one global in/out pair - see app/pricing.py.
    model: str = ""
    # The latency breakdown Azure returns inside usage itself (only on
    # non-streaming responses): it separates inference from queueing.
    latency_checkpoint: dict[str, Any] = field(default_factory=dict)

    def cost(self, s: Settings, model: str | None = None) -> dict[str, Any]:
        """The turn's cost, at the price OF THE MODEL that ran.

        Cached input is billed separately and cheaper - which is what makes the
        prompt cache a measurable cost lever. The price comes from the catalogue
        (data/model_prices.csv); `price_origin` says where it came from, because
        a cost without provenance is not usable in a proposal.
        """
        p: ModelPrice = get_price_book(s).resolve(model or self.model)
        fresh_in = max(self.input_tokens - self.cached_input_tokens, 0)
        c_in = fresh_in / 1e6 * p.in_per_mtok
        c_cached = self.cached_input_tokens / 1e6 * p.cached_in_per_mtok
        c_out = self.output_tokens / 1e6 * p.out_per_mtok
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "fresh_input_tokens": fresh_in,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "llm_hops": self.hops,
            "cost_input": round(c_in, 6),
            "cost_cached_input": round(c_cached, 6),
            "cost_output": round(c_out, 6),
            "cost_total": round(c_in + c_cached + c_out, 6),
            "currency": p.currency,
            "price_model": p.matched or p.model,
            "price_origin": p.origin,  # catalog | override | fallback
            "price_in_per_mtok": p.in_per_mtok,
            "price_out_per_mtok": p.out_per_mtok,
            "price_is_placeholder": p.is_placeholder,
        }


def _family(model: str) -> str:
    """A deployment's parameter family, inferred from its name.

    Foundry deployments are usually named after the model, which makes the name
    a reliable signal. When it is not, an explicit override resolves it.
    """
    m = model.lower()
    if m.startswith(("gpt-5", "gpt5", "o1", "o3", "o4")):
        return "reasoning_capable"
    return "classic"


class LLM:
    """A model client with streaming, real token counts and cost."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self._client: AsyncOpenAI | AsyncAzureOpenAI | None = None

        if settings.llm_provider == "foundry":
            base = settings.foundry_base_url
            if not (base and settings.azure_ai_api_key):
                raise RuntimeError(
                    "LLM_PROVIDER=foundry requires AZURE_AI_ENDPOINT and AZURE_AI_API_KEY"
                )
            # The AI Services v1 endpoint accepts the key as a Bearer token and
            # also in the api-key header; we send both to cover either mode.
            self._client = AsyncOpenAI(
                base_url=base,
                api_key=settings.azure_ai_api_key,
                default_headers={"api-key": settings.azure_ai_api_key},
                max_retries=0,  # retries mask real latency; here we want to see the failure
            )
        elif settings.llm_provider == "openai":
            if not settings.openai_api_key:
                raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
            self._client = AsyncOpenAI(api_key=settings.openai_api_key, max_retries=0)
        elif settings.llm_provider == "azure":
            if not (settings.azure_openai_endpoint and settings.azure_openai_api_key):
                raise RuntimeError(
                    "LLM_PROVIDER=azure requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY"
                )
            self._client = AsyncAzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                api_version=settings.azure_openai_api_version,
                max_retries=0,
            )

    def model_for(self, tier: Tier) -> str:
        return {
            "nano": self.s.nano_model,
            "mini": self.s.mini_model,
            "frontier": self.s.frontier_model,
        }[tier]

    # ---------- shim de parametros ----------

    def _params(self, model: str, max_tokens: int, *, deterministic: bool) -> dict[str, Any]:
        """Assembles the kwargs that model family accepts."""
        if _family(model) == "reasoning_capable":
            params: dict[str, Any] = {"max_completion_tokens": max_tokens}
            # temperature is unsupported (only the default 1) - omitted on purpose
            if self.s.reasoning_effort:
                params["reasoning_effort"] = self.s.reasoning_effort
            return params
        return {
            "max_tokens": max_tokens,
            "temperature": 0 if deterministic else self.s.temperature,
        }

    # ---------- hop de decisao de tools ----------

    async def decide_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tier: Tier = "mini",
        max_tokens: int | None = None,
        usage_out: Usage | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """One non-streaming hop that decides which tools to call.

        Returns (tool_calls, text). If `tool_calls` comes back empty, the model
        decided to answer directly and the text is the answer.

        This hop is **non-streaming by nature** - there is no "first token" of a
        tool decision. That is exactly why it is expensive: a turn with tools
        costs at least two complete round trips before the first word reaches the
        screen. It is the central finding of the version with tools.
        """
        if self._client is None:  # mock
            await asyncio.sleep(self.s.mock_ttft_ms / 1000.0 * 0.5)
            return [], "simulated answer"

        model = self.model_for(tier)
        if usage_out is not None:
            usage_out.model = model
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            tool_choice="auto",
            **self._params(model, max_tokens or self.s.max_output_tokens, deterministic=False),
        )
        if usage_out is not None and resp.usage:
            _accumulate_usage(usage_out, resp.usage)

        msg = resp.choices[0].message
        calls: list[dict[str, Any]] = []
        for tc in msg.tool_calls or []:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            try:
                args = json.loads(fn.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": fn.name, "args": args})
        return calls, (msg.content or "")

    # ---------- a hop with BOTH streaming and tool calling ----------

    async def stream_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        outcome: StreamOutcome,
        tier: Tier = "mini",
        max_tokens: int | None = None,
        usage_out: Usage | None = None,
    ) -> AsyncIterator[str]:
        """Yields text deltas and accumulates tool_calls into `outcome`.

        tool_call fragments arrive chopped up in the stream (`id` in one chunk,
        pieces of `arguments` in the following ones) and are reassembled by index.
        """
        max_tokens = max_tokens or self.s.max_output_tokens

        if self._client is None:  # mock
            await asyncio.sleep(self.s.mock_ttft_ms / 1000.0)
            delay = 1.0 / max(self.s.mock_tokens_per_s, 1)
            n = min(self.s.mock_output_tokens, max_tokens)
            for i in range(n):
                yield _MOCK_WORDS[i % len(_MOCK_WORDS)] + " "
                await asyncio.sleep(delay)
            outcome.text = "mock"
            if usage_out is not None:
                usage_out.model = self.model_for(tier)
                usage_out.input_tokens += 3000
                usage_out.output_tokens += n
                usage_out.hops += 1
            return

        model = self.model_for(tier)
        if usage_out is not None:
            usage_out.model = model
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
            **self._params(model, max_tokens, deterministic=False),
        )

        acc: dict[int, dict[str, Any]] = {}
        pieces: list[str] = []
        async for chunk in stream:
            if getattr(chunk, "usage", None) and usage_out is not None:
                _accumulate_usage(usage_out, chunk.usage)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                outcome.finish_reason = choice.finish_reason

            delta = choice.delta
            for tc in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc, "index", 0) or 0
                slot = acc.setdefault(idx, {"id": None, "name": None, "args": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments

            if delta.content:
                pieces.append(delta.content)
                yield delta.content

        outcome.text = "".join(pieces)
        for idx in sorted(acc):
            slot = acc[idx]
            if not slot["name"]:
                continue
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            outcome.tool_calls.append(
                {"id": slot["id"] or f"call_{idx}", "name": slot["name"], "args": args}
            )

    # ---------- short generations (locale, intent) ----------

    async def complete(
        self,
        messages: list[dict[str, str]],
        tier: Tier = "nano",
        max_tokens: int = 16,
    ) -> str:
        if self._client is None:  # mock
            await asyncio.sleep(self.s.mock_ttft_ms / 1000.0 * 0.38)
            return "suporte_produto"

        model = self.model_for(tier)
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **self._params(model, max_tokens, deterministic=True),
        )
        return (resp.choices[0].message.content or "").strip()

    # ---------- streaming generation (the user's answer) ----------

    async def stream(
        self,
        messages: list[dict[str, str]],
        tier: Tier = "mini",
        max_tokens: int | None = None,
        usage_out: Usage | None = None,
    ) -> AsyncIterator[str]:
        """Yields text deltas. If `usage_out` is passed, it is filled with the
        real consumption when the provider sends the final usage chunk."""
        max_tokens = max_tokens or self.s.max_output_tokens

        if self._client is None:  # mock
            await asyncio.sleep(self.s.mock_ttft_ms / 1000.0)
            delay = 1.0 / max(self.s.mock_tokens_per_s, 1)
            n = min(self.s.mock_output_tokens, max_tokens)
            for i in range(n):
                yield _MOCK_WORDS[i % len(_MOCK_WORDS)] + " "
                await asyncio.sleep(delay)
            if usage_out is not None:
                usage_out.model = self.model_for(tier)
                usage_out.input_tokens = 3000  # a typical prompt size with RAG
                usage_out.output_tokens = n
            return

        model = self.model_for(tier)
        if usage_out is not None:
            usage_out.model = model
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            stream_options={"include_usage": True},
            **self._params(model, max_tokens, deterministic=False),
        )
        async for chunk in stream:
            if getattr(chunk, "usage", None) and usage_out is not None:
                _accumulate_usage(usage_out, chunk.usage)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ---------- non-streaming generation ----------

    async def complete_full(
        self,
        messages: list[dict[str, str]],
        tier: Tier = "mini",
        max_tokens: int | None = None,
        usage_out: Usage | None = None,
    ) -> str:
        """The complete answer in one shot.

        Used for the streaming vs. non-streaming comparison, and because Azure
        only returns `latency_checkpoint` (the queue vs. inference breakdown) in
        this mode - on streaming it comes back empty.
        """
        max_tokens = max_tokens or self.s.max_output_tokens
        if self._client is None:  # mock
            await asyncio.sleep(self.s.mock_ttft_ms / 1000.0)
            return " ".join(_MOCK_WORDS[: self.s.mock_output_tokens])

        model = self.model_for(tier)
        if usage_out is not None:
            usage_out.model = model
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **self._params(model, max_tokens, deterministic=False),
        )
        if usage_out is not None and resp.usage:
            _fill_usage(usage_out, resp.usage)
        return resp.choices[0].message.content or ""

    # ---------- embeddings (the L2 semantic cache) ----------

    async def embed(self, text: str) -> list[float]:
        if self._client is None:  # mock
            await asyncio.sleep(0.045)
            return _cheap_hash_vector(text)

        resp = await self._client.embeddings.create(model=self.s.embedding_model, input=text)
        return resp.data[0].embedding


def _accumulate_usage(out: Usage, raw: Any) -> None:
    """Adds one more hop's usage.

    With tools, a turn makes several model calls. The turn's cost is their SUM -
    reporting only the last one would understate the bill, and that is exactly
    the mistake that makes an agent look cheap in testing and expensive on the
    invoice.
    """
    out.input_tokens += getattr(raw, "prompt_tokens", 0) or 0
    out.output_tokens += getattr(raw, "completion_tokens", 0) or 0
    out.hops += 1

    pd = getattr(raw, "prompt_tokens_details", None)
    if pd is not None:
        out.cached_input_tokens += getattr(pd, "cached_tokens", 0) or 0

    cd = getattr(raw, "completion_tokens_details", None)
    if cd is not None:
        out.reasoning_tokens += getattr(cd, "reasoning_tokens", 0) or 0

    extra = getattr(raw, "model_extra", None) or {}
    lc = extra.get("latency_checkpoint")
    if isinstance(lc, dict) and lc:
        out.latency_checkpoint = lc


def _fill_usage(out: Usage, raw: Any) -> None:
    """Copies the SDK's usage into our struct, tolerating missing fields - the
    cache and reasoning details vary by provider and by API version."""
    out.input_tokens = getattr(raw, "prompt_tokens", 0) or 0
    out.output_tokens = getattr(raw, "completion_tokens", 0) or 0

    pd = getattr(raw, "prompt_tokens_details", None)
    if pd is not None:
        out.cached_input_tokens = getattr(pd, "cached_tokens", 0) or 0

    cd = getattr(raw, "completion_tokens_details", None)
    if cd is not None:
        out.reasoning_tokens = getattr(cd, "reasoning_tokens", 0) or 0

    # latency_checkpoint is not in the SDK schema; it arrives via model_extra
    extra = getattr(raw, "model_extra", None) or {}
    lc = extra.get("latency_checkpoint")
    if isinstance(lc, dict) and lc:
        out.latency_checkpoint = lc


def _cheap_hash_vector(text: str, dim: int = 256) -> list[float]:
    """A deterministic bag-of-words hashing vector. Used by the mock provider and
    as a fallback when no embedding is available: it preserves the property
    "identical question => identical vector", which is enough to exercise L2."""
    import math
    from zlib import crc32

    vec = [0.0] * dim
    for tok in text.lower().split():
        # crc32 rather than hash(): hash() of a str is randomised per process.
        vec[crc32(tok.encode()) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

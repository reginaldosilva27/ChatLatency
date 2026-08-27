"""Answers one question per deployment: does it actually stream?

A response can use the streaming protocol and deliver nothing incrementally.
Proxies buffer, compression middleware waits, and some deployments generate the
whole answer before emitting the first chunk. None of that is visible from the
client SDK, because the SDK hands you deltas either way - it just hands them all
at once at the end.

So this probe reads **raw bytes** (`aiter_bytes`, no line buffering) straight off
the socket and timestamps every chunk that arrives. What it reports:

    first byte      when the HTTP response opened (SSE preamble)
    first content   when the first actual text delta arrived
    last content    when the last one did
    bursts          how many separate socket reads carried content
    verdict         INCREMENTAL or ONE BLOCK

The rule for the verdict is the same one `app/telemetry.py` uses to set
`stream_buffered`: if the whole content window fits inside a small fraction of
the total time, the answer was generated first and dispatched afterwards.

    PYTHONPATH=. uv run python scripts/probe_streaming.py
    PYTHONPATH=. uv run python scripts/probe_streaming.py gpt-4.1-mini gpt-5.4
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time

import httpx

from app.config import get_settings
from app.llm import _family  # the per-family parameter shim, reused on purpose

# ~350 words: long enough that the difference between generating and delivering
# is unambiguous, short enough to stay cheap.
PROMPT = (
    "Explain, in about 350 words, why time to first token matters more than total "
    "response time in a streaming chat interface. Use plain prose, no lists."
)

# A content delta at the raw byte level. We only need to know that a chunk
# carried *some* text, not what the text was.
CONTENT = re.compile(rb'"content"\s*:\s*"(?:[^"\\]|\\.)+?"')
USAGE = re.compile(rb'"completion_tokens"\s*:\s*(\d+)')


async def probe(
    client: httpx.AsyncClient, base: str, key: str, deployment: str, effort: str | None = "none"
) -> dict:
    body: dict = {
        "model": deployment,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # The same shim the engine uses: gpt-5.x/o-series take max_completion_tokens
    # and reject temperature; gpt-4.x takes max_tokens.
    if _family(deployment) == "reasoning_capable":
        body["max_completion_tokens"] = 700
        if effort:
            body["reasoning_effort"] = effort
    else:
        body["max_tokens"] = 700

    t0 = time.perf_counter()
    ms = lambda: (time.perf_counter() - t0) * 1000.0  # noqa: E731

    first_byte = first_content = last_content = None
    bursts = 0
    total_bytes = 0
    out_tokens = None
    arrivals: list[float] = []

    try:
        async with client.stream(
            "POST",
            f"{base}chat/completions",
            headers={"api-key": key, "Authorization": f"Bearer {key}"},
            json=body,
        ) as r:
            if r.status_code >= 400:
                detail = (await r.aread())[:300].decode(errors="replace")
                # Chat-tuned models in the gpt-5 family take max_completion_tokens
                # but reject reasoning_effort="none" (some only accept "medium").
                # Retry once without it rather than reporting a deployment as
                # broken when it is only fussy about one parameter.
                if r.status_code == 400 and "reasoning_effort" in detail and effort:
                    return await probe(client, base, key, deployment, effort=None)
                return {"deployment": deployment, "error": f"HTTP {r.status_code}: {detail}"}

            async for chunk in r.aiter_bytes():
                now = ms()
                if first_byte is None:
                    first_byte = now
                total_bytes += len(chunk)
                # Ignore the empty-string content of the role-only first delta.
                hits = [m for m in CONTENT.findall(chunk) if m not in (b'"content":""',)]
                if hits:
                    if first_content is None:
                        first_content = now
                    last_content = now
                    bursts += 1
                    arrivals.append(now)
                u = USAGE.search(chunk)
                if u:
                    out_tokens = int(u.group(1))
    except Exception as exc:  # noqa: BLE001
        return {"deployment": deployment, "error": f"{type(exc).__name__}: {exc}"}

    total = ms()
    window = (last_content - first_content) if (first_content and last_content) else 0.0
    share = (window / total * 100) if total else 0.0
    # Same threshold as stream_buffered: a content window under a quarter of the
    # total means the answer was already finished when it started arriving.
    incremental = bursts >= 5 and share >= 25.0
    gap = max(
        (b - a for a, b in zip(arrivals, arrivals[1:], strict=False)), default=0.0
    )
    return {
        "deployment": deployment,
        "first_byte_ms": round(first_byte or 0, 1),
        "first_content_ms": round(first_content or 0, 1),
        "total_ms": round(total, 1),
        "content_window_ms": round(window, 1),
        "content_share_pct": round(share, 1),
        "bursts": bursts,
        "largest_gap_ms": round(gap, 1),
        "out_tokens": out_tokens,
        "bytes": total_bytes,
        "incremental": incremental,
    }


async def main() -> int:
    s = get_settings()
    base, key = s.foundry_base_url, s.azure_ai_api_key
    if not (base and key):
        print("This probe needs AZURE_AI_ENDPOINT and AZURE_AI_API_KEY (LLM_PROVIDER=foundry).")
        return 1

    deployments = sys.argv[1:] or [s.mini_model]
    print(f"endpoint: {base}")
    print(f"probing : {', '.join(deployments)}\n")

    rows = []
    # Sequential on purpose: concurrent probes would compete for the same quota
    # and the capacity hypothesis is exactly what is being tested.
    async with httpx.AsyncClient(timeout=180.0) as client:
        for d in deployments:
            print(f"  {d} ...", end=" ", flush=True)
            row = await probe(client, base, key, d)
            rows.append(row)
            if row.get("error"):
                print(f"FAILED - {row['error'][:80]}")
            else:
                print(
                    f"{'INCREMENTAL' if row['incremental'] else 'ONE BLOCK':11s}  "
                    f"first content {row['first_content_ms']:.0f} ms  "
                    f"window {row['content_share_pct']:.0f}% of {row['total_ms']:.0f} ms"
                )

    ok = [r for r in rows if not r.get("error")]
    if ok:
        print(f"\n{'deployment':22s} {'1st content':>12s} {'total':>8s} {'window':>8s} "
              f"{'bursts':>7s} {'gap':>8s} {'tok':>5s}  verdict")
        print("-" * 92)
        for r in sorted(ok, key=lambda x: x["first_content_ms"]):
            print(
                f"{r['deployment']:22s} {r['first_content_ms']:11.0f}m {r['total_ms']:7.0f}m "
                f"{r['content_share_pct']:7.0f}% {r['bursts']:7d} {r['largest_gap_ms']:7.0f}m "
                f"{str(r['out_tokens'] or '-'):>5s}  "
                f"{'INCREMENTAL' if r['incremental'] else 'ONE BLOCK'}"
            )
        print(
            "\nwindow = share of the total spent receiving content. A single-digit"
            "\npercentage means the answer was generated first and dispatched after."
        )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

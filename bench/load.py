"""Load driver and latency report.

It measures two views that must not be confused with each other:

  client : the user's wall-clock time - network, TLS, framework and SSE included.
           It is the number that becomes a per-endpoint p95 commitment.
  server : the per-stage breakdown from the trace - it says WHERE the time went.

The difference between the first token seen by the client and the one marked by
the server is the "network + gateway" slice of the budget (~40 ms in the
reference).

Usage:
  python -m bench.load --requests 60 --concurrency 6
  python -m bench.load --scenario cold        # 100% cache miss, mini route
  python -m bench.load --scenario ab-spec     # A/B of speculative retrieval
  python -m bench.load --scenario cache-curve # latency per cache hit rate
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

from app.budget import BUDGET

from .workload import USER_CONTEXT, Turn, make_workload

console = Console()
REPORTS = Path(__file__).resolve().parent.parent / "reports"


@dataclass
class Result:
    ok: bool
    kind: str
    # the client view (local perf_counter)
    client_ttfb_ms: float | None = None  # first byte (the meta event)
    client_first_token_ms: float | None = None
    client_complete_ms: float | None = None
    # the server view
    trace: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def cache_tier(self) -> str:
        return self.trace.get("cache_tier", "miss")

    @property
    def transport_ms(self) -> float | None:
        """Network + framework + SSE: client minus server on the same event."""
        srv = self.trace.get("first_token_ms")
        if self.client_first_token_ms is None or srv is None:
            return None
        return round(self.client_first_token_ms - srv, 2)


async def one_request(
    client: httpx.AsyncClient, base: str, turn: Turn, overrides: dict[str, Any]
) -> Result:
    body: dict[str, Any] = {
        "question": turn.question,
        "locale": turn.locale,
        "topic": turn.topic,
        "user_context": USER_CONTEXT,
        **overrides,
    }
    t0 = time.perf_counter()
    res = Result(ok=False, kind=turn.kind)

    def ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    try:
        async with client.stream("POST", f"{base}/v1/chat/stream", json=body) as resp:
            resp.raise_for_status()
            event: str | None = None
            async for line in resp.aiter_lines():
                if res.client_ttfb_ms is None:
                    res.client_ttfb_ms = ms()
                if line.startswith("event: "):
                    event = line[7:].strip()
                elif line.startswith("data: "):
                    payload = json.loads(line[6:])
                    if event == "token":
                        if res.client_first_token_ms is None:
                            res.client_first_token_ms = ms()
                    elif event == "trace":
                        res.trace = payload
                        res.client_complete_ms = ms()
                    elif event == "done":
                        res.ok = bool(payload.get("ok"))
                        res.error = payload.get("error")
    except Exception as exc:  # noqa: BLE001
        res.error = f"{type(exc).__name__}: {exc}"
    return res


async def run_load(
    base: str,
    turns: list[Turn],
    concurrency: int,
    overrides: dict[str, Any] | None = None,
    warmup: int = 2,
) -> list[Result]:
    overrides = overrides or {}
    limits = httpx.Limits(
        max_connections=concurrency + 4, max_keepalive_connections=concurrency + 4
    )
    async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:
        # warmup: pays for slow imports, the TCP connection and graph compilation
        for turn in turns[:warmup]:
            await one_request(client, base, turn, overrides)

        sem = asyncio.Semaphore(concurrency)

        async def guarded(turn: Turn) -> Result:
            async with sem:
                return await one_request(client, base, turn, overrides)

        return await asyncio.gather(*(guarded(t) for t in turns))


# ---------------------------------------------------------------- report


def pct(vals: list[float | None], p: float) -> float | None:
    xs = sorted(v for v in vals if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return round(xs[0], 1)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:,.0f}".replace(",", ".")


# The reference budget lives in app/budget.py and is served to the UI at
# /v1/budget. Declared once: the number the bench compares against and the
# number the waterfall draws are the same promise, or the comparison is theatre.


def report(results: list[Result], label: str) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    hits = [r for r in ok if r.cache_tier in ("l1", "l2")]
    miss = [r for r in ok if r.cache_tier not in ("l1", "l2")]

    console.print()
    console.rule(f"[bold]{label}[/]  ·  {len(ok)} ok / {len(fail)} failed")

    # -------- headline: first token, the client view --------
    t = Table(title="First token - the client view (what the user feels)", box=None)
    t.add_column("slice")
    t.add_column("n", justify="right")
    for c in ("p50", "p90", "p95", "p99", "max"):
        t.add_column(c, justify="right")
    t.add_column("budget", justify="right")
    t.add_column("", justify="left")

    for name, rows, target in (
        ("all", ok, None),
        ("cache MISS", miss, BUDGET["first_token"]),
        ("cache HIT", hits, BUDGET["cache_hit"]),
    ):
        vals = [r.client_first_token_ms for r in rows]
        p50, p95 = pct(vals, 50), pct(vals, 95)
        verdict = ""
        if target and p50 is not None:
            verdict = "[green]within[/]" if p50 <= target else f"[red]+{p50 - target:.0f}ms[/]"
        t.add_row(
            name,
            str(len(rows)),
            fmt(p50),
            fmt(pct(vals, 90)),
            fmt(p95),
            fmt(pct(vals, 99)),
            fmt(pct(vals, 100)),
            fmt(target),
            verdict,
        )
    console.print(t)

    # -------- waterfall: where the time went (server) --------
    stage_names: list[str] = []
    for r in ok:
        for k in r.trace.get("stages_ms", {}):
            if k not in stage_names:
                stage_names.append(k)
    order = [
        "cache_l1",
        "cache_l2_embed",
        "cache_l2_search",
        "locale",
        "intent",
        "retrieval",
        "route",
        "model_ttft",
        "model_stream",
    ]
    stage_names.sort(key=lambda n: order.index(n) if n in order else 99)

    w = Table(title="Per-stage breakdown - server (cache miss)", box=None)
    w.add_column("stage")
    w.add_column("n", justify="right")
    w.add_column("p50", justify="right")
    w.add_column("p95", justify="right")
    w.add_column("budget", justify="right")
    w.add_column("% of the 1st-token p50", justify="right")

    ft_p50 = pct([r.client_first_token_ms for r in miss], 50) or 0
    for name in stage_names:
        vals = [r.trace.get("stages_ms", {}).get(name) for r in miss]
        p50 = pct(vals, 50)
        share = f"{p50 / ft_p50 * 100:.0f}%" if (p50 and ft_p50) else "—"
        w.add_row(
            name,
            str(len([v for v in vals if v is not None])),
            fmt(p50),
            fmt(pct(vals, 95)),
            fmt(BUDGET.get(name)),
            share,
        )

    transport = [r.transport_ms for r in miss]
    w.add_row(
        "[dim]transport (network+SSE)[/]",
        str(len([v for v in transport if v is not None])),
        fmt(pct(transport, 50)),
        fmt(pct(transport, 95)),
        fmt(BUDGET["gateway"]),
        "",
    )
    console.print(w)

    # -------- complete answer, cache and tiers --------
    c = Table(title="Complete answer · cache · tiers", box=None)
    c.add_column("metric")
    c.add_column("value", justify="right")
    comp_all = [r.client_complete_ms for r in ok]
    tiers = [r.trace.get("tier") for r in ok]
    tok_s = [r.trace.get("tokens_per_s") for r in ok]
    saving = [r.trace.get("parallel_saving_ms") for r in miss]
    c.add_row("complete p50 / p95", f"{fmt(pct(comp_all, 50))} / {fmt(pct(comp_all, 95))} ms")
    c.add_row("cache hit rate", f"{len(hits) / max(len(ok), 1) * 100:.1f}%")
    c.add_row(
        "  L1 exact / L2 semantic",
        f"{sum(1 for r in hits if r.cache_tier == 'l1')} / "
        f"{sum(1 for r in hits if r.cache_tier == 'l2')}",
    )
    c.add_row("tokens/s p50", fmt(pct(tok_s, 50)))
    c.add_row("speculative fan-out saving p50", f"{fmt(pct(saving, 50))} ms")
    for tier in ("cache", "mini", "frontier", "nano"):
        n = sum(1 for x in tiers if x == tier)
        if n:
            c.add_row(f"  turns on tier {tier}", f"{n} ({n / max(len(ok), 1) * 100:.0f}%)")
    if fail:
        c.add_row("[red]errors[/]", str(len(fail)))
    console.print(c)

    if fail:
        seen: dict[str, int] = {}
        for r in fail:
            seen[r.error or "?"] = seen.get(r.error or "?", 0) + 1
        for err, n in sorted(seen.items(), key=lambda kv: -kv[1])[:5]:
            console.print(f"  [red]{n}x[/] {err[:160]}")

    return {
        "label": label,
        "n_ok": len(ok),
        "n_fail": len(fail),
        "cache_hit_rate": round(len(hits) / max(len(ok), 1), 4),
        "first_token_ms": {
            "all": {p: pct([r.client_first_token_ms for r in ok], p) for p in (50, 90, 95, 99)},
            "miss": {p: pct([r.client_first_token_ms for r in miss], p) for p in (50, 90, 95, 99)},
            "hit": {p: pct([r.client_first_token_ms for r in hits], p) for p in (50, 90, 95, 99)},
        },
        "complete_ms": {p: pct(comp_all, p) for p in (50, 95)},
        "stages_p50_ms": {
            n: pct([r.trace.get("stages_ms", {}).get(n) for r in miss], 50) for n in stage_names
        },
        "transport_p50_ms": pct(transport, 50),
        "tokens_per_s_p50": pct(tok_s, 50),
        "parallel_saving_p50_ms": pct(saving, 50),
        "tier_mix": {tier: sum(1 for x in tiers if x == tier) for tier in set(tiers) if tier},
    }


# ---------------------------------------------------------------- scenarios


async def post(base: str, path: str) -> None:
    async with httpx.AsyncClient(timeout=30.0) as c:
        await c.post(f"{base}{path}")


async def scenario_mixed(base: str, args) -> list[dict[str, Any]]:
    turns = make_workload(args.requests, seed=args.seed)
    res = await run_load(base, turns, args.concurrency)
    return [report(res, f"mixed load · {args.concurrency} concurrent")]


async def scenario_cold(base: str, args) -> list[dict[str, Any]]:
    """100% cache miss: measures the clean budget, with no help from the cache."""
    await post(base, "/v1/cache/reset")
    turns = make_workload(args.requests, seed=args.seed, mix=(0.0, 1.0, 0.0, 0.0), paraphrase_rate=0)
    res = await run_load(
        base, turns, args.concurrency, overrides={"cache_l1": False, "cache_canonical": False, "cache_l2": False}
    )
    return [report(res, "pure cache miss · mini route (the reference budget)")]


async def scenario_ab_spec(base: str, args) -> list[dict[str, Any]]:
    """Speculative retrieval on vs. off, with the cache off in both."""
    out = []
    for spec in (False, True):
        await post(base, "/v1/cache/reset")
        turns = make_workload(args.requests, seed=args.seed, mix=(0.0, 1.0, 0.0, 0.0), paraphrase_rate=0)
        res = await run_load(
            base,
            turns,
            args.concurrency,
            overrides={"speculative_retrieval": spec, "cache_l1": False, "cache_canonical": False, "cache_l2": False},
        )
        out.append(report(res, f"speculative retrieval = {spec}"))
    a, b = out
    delta = (a["first_token_ms"]["miss"][50] or 0) - (b["first_token_ms"]["miss"][50] or 0)
    console.print(
        f"\n[bold]Speculative fan-out saving:[/] {delta:.0f} ms on the first-token p50 "
        f"(the reference estimates 200-300 ms)"
    )
    return out


async def scenario_ab_intent(base: str, args) -> list[dict[str, Any]]:
    """Intent classification on vs. off the critical path.

    The reference budget puts intent (nano) as a ~400 ms sequential step before
    generation. Because it is a network call to a model, in practice it costs a
    whole round trip - it is the largest CONTROLLABLE slice of the budget. This
    A/B measures how much the first token improves when intent leaves the
    critical path (it is still needed for tier routing and for analytics events,
    but it can be asynchronous or heuristic).

    It sends `intent_mode` explicitly rather than relying on the default. Since
    the heuristic router shipped, the default is `heuristic`, so a scenario that
    only toggled the old boolean would be comparing "no label" against "a free
    label" and reporting ~0 ms - a scenario that had quietly stopped measuring
    its own subject. `ab-router` is the one that measures the replacement.
    """
    out = []
    for use_intent in (True, False):
        await post(base, "/v1/cache/reset")
        turns = make_workload(args.requests, seed=args.seed, mix=(0.0, 1.0, 0.0, 0.0), paraphrase_rate=0)
        res = await run_load(
            base,
            turns,
            args.concurrency,
            overrides={
                "intent_mode": "llm" if use_intent else "off",
                "cache_l1": False,
                "cache_canonical": False,
                "cache_l2": False,
                "force_tier": "mini",
            },
        )
        out.append(report(res, f"intent classification on the critical path = {use_intent}"))
    with_intent, without = out
    delta = (with_intent["first_token_ms"]["miss"][50] or 0) - (
        without["first_token_ms"]["miss"][50] or 0
    )
    console.print(
        f"\n[bold]Cost of intent on the critical path:[/] {delta:.0f} ms on the first-token p50."
        f"\n[dim]If tier routing can be heuristic (keyword + question length) and intent becomes"
        f" an asynchronous event, that time leaves what the user waits for.[/]"
    )
    return out


async def scenario_ab_router(base: str, args) -> list[dict[str, Any]]:
    """The model as router against a table as router - findings 02, 08 and 11.

    `ab-intent` measures what the classification COSTS. This measures what
    replacing it BUYS, which is a different question: the label still gets
    produced, it is just produced locally (`app/routing.py`) instead of by a
    round trip to a nano deployment.

    Two numbers matter and only one of them is latency. The first-token delta
    is the finding-02 saving, collected. The second is agreement: run with
    `intent_mode=async` and every turn carries `intent_agrees`, because a
    router that is fast and wrong is finding 08 with better timings.
    """
    out = []
    for mode in ("llm", "heuristic"):
        await post(base, "/v1/cache/reset")
        turns = make_workload(
            args.requests, seed=args.seed, mix=(0.0, 1.0, 0.0, 0.0), paraphrase_rate=0
        )
        res = await run_load(
            base,
            turns,
            args.concurrency,
            overrides={
                "intent_mode": mode,
                "cache_l1": False,
                "cache_canonical": False,
                "cache_l2": False,
                "force_tier": "mini",
            },
        )
        out.append(report(res, f"intent_mode = {mode}"))
    llm_run, heuristic_run = out
    delta = (llm_run["first_token_ms"]["miss"][50] or 0) - (
        heuristic_run["first_token_ms"]["miss"][50] or 0
    )
    console.print(
        f"\n[bold]Bought by routing locally:[/] {delta:.0f} ms on the first-token p50."
        f"\n[dim]Latency is the easy half. Run --scenario ab-router with INTENT_MODE=async"
        f" to collect `intent_agrees` and find out what the speed cost in labels.[/]"
    )
    return out


async def scenario_ab_agentic(base: str, args) -> list[dict[str, Any]]:
    """The agent loop (the model picks the tools) vs. the fixed pipeline.

    This is the most consequential comparison in the engine. The fixed pipeline
    makes ONE round trip to the model; the agentic one makes at least TWO when it
    uses a tool - and the second only starts after the tool has answered. The
    whole difference lands on the time to the first word, and it shows up in no
    throughput metric at all.
    """
    out = []
    for agentic in (False, True):
        await post(base, "/v1/cache/reset")
        turns = make_workload(args.requests, seed=args.seed, mix=(0.0, 1.0, 0.0, 0.0),
                              paraphrase_rate=0)
        res = await run_load(
            base, turns, args.concurrency,
            overrides={"agentic": agentic, "cache_l1": False, "cache_canonical": False, "cache_l2": False},
        )
        out.append(report(res, f"agentic = {agentic}"))
    fixed, agent = out
    d50 = (agent["first_token_ms"]["miss"][50] or 0) - (fixed["first_token_ms"]["miss"][50] or 0)
    d95 = (agent["first_token_ms"]["miss"][95] or 0) - (fixed["first_token_ms"]["miss"][95] or 0)
    console.print(
        f"\n[bold]Cost of the agent loop:[/] {d50:+.0f} ms on the p50 and {d95:+.0f} ms on the "
        f"p95 of the first token."
        f"\n[dim]The agent gains coverage (it picks the right source) and pays in latency "
        f"(one more round trip to the model). Picking the source heuristically, when that is "
        f"possible, is the fast version of the same behaviour.[/]"
    )
    return out


async def scenario_cache_curve(base: str, args) -> list[dict[str, Any]]:
    """Average latency per hit rate - it ties cost to latency in one measurement."""
    out = []
    for label, l1, l2 in (
        ("no cache", False, False),
        ("L1 only", True, False),
        ("L1 + L2", True, True),
    ):
        await post(base, "/v1/cache/reset")
        turns = make_workload(args.requests, seed=args.seed)
        res = await run_load(
            base, turns, args.concurrency, overrides={"cache_l1": l1, "cache_canonical": False, "cache_l2": l2}
        )
        out.append(report(res, f"cache: {label}"))
    return out


async def scenario_tiers(base: str, args) -> list[dict[str, Any]]:
    """TTFT per tier - it isolates the slice that is the provider's, not ours."""
    out = []
    for tier in ("nano", "mini", "frontier"):
        await post(base, "/v1/cache/reset")
        turns = make_workload(
            max(args.requests // 2, 6), seed=args.seed, mix=(0.0, 1.0, 0.0, 0.0), paraphrase_rate=0
        )
        res = await run_load(
            base,
            turns,
            args.concurrency,
            overrides={"force_tier": tier, "cache_l1": False, "cache_canonical": False, "cache_l2": False},
        )
        out.append(report(res, f"forced tier = {tier}"))
    return out


SCENARIOS = {
    "mixed": scenario_mixed,
    "cold": scenario_cold,
    "ab-spec": scenario_ab_spec,
    "ab-intent": scenario_ab_intent,
    "ab-router": scenario_ab_router,
    "ab-agentic": scenario_ab_agentic,
    "cache-curve": scenario_cache_curve,
    "tiers": scenario_tiers,
}


async def main() -> None:
    ap = argparse.ArgumentParser(description="ChatLatency - load driver")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="mixed")
    ap.add_argument(
        "--out", default=None, help="output JSON file (default: reports/<ts>.json)"
    )
    args = ap.parse_args()

    async with httpx.AsyncClient(timeout=10.0) as c:
        health = (await c.get(f"{args.base}/healthz")).json()
    console.print("[bold]effective server configuration[/]")
    console.print_json(data=health)

    await post(args.base, "/v1/traces/reset")
    t0 = time.perf_counter()
    reports = await SCENARIOS[args.scenario](args.base, args)
    wall = time.perf_counter() - t0

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario": args.scenario,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "wall_s": round(wall, 2),
        "server": health,
        "reports": reports,
    }
    REPORTS.mkdir(exist_ok=True)
    out = (
        Path(args.out)
        if args.out
        else REPORTS / (f"{args.scenario}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json")
    )
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]report saved to {out}  ·  wall {wall:.1f}s[/]")


if __name__ == "__main__":
    asyncio.run(main())

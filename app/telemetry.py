"""Latency instrumentation.

Every span corresponds to a line of the reference budget, so a measured number
is comparable to the promised one:

    internal network + gateway   ~40 ms     -> derived (client_ttfb - server_t0)
    intent (nano)                ~400 ms    -> span "intent"
    retrieval                    ~150 ms    -> span "retrieval"
    model prefill                ~1050 ms   -> span "model_ttft" / "hop:N"
    -------------------------------------------------------------
    first token                  ~1600 ms   -> mark "first_token"
    complete answer              ~3000 ms   -> mark "last_token"

Duration uses time.perf_counter (monotonic); time.time is used only for the
start label, so wall-clock never mixes with measurement.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

# Canonical display order in the report (waterfall).
STAGE_ORDER = [
    "cache_l1",
    "hop:1",
    "hop:2",
    "hop:3",
    "cache_l2_embed",
    "cache_l2_search",
    "locale",
    "intent",
    "retrieval",
    "route",
    "model_ttft",
    "model_stream",
]

# Prefix for tool spans. The waterfall groups them in their own band because
# tools from the same hop run in parallel and must not be summed.
TOOL_PREFIX = "tool:"


@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.end_ms is None:
            return None
        return self.end_ms - self.start_ms


@dataclass
class Trace:
    """One trace per conversation turn."""

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: float = field(default_factory=time.time)
    _t0: float = field(default_factory=time.perf_counter)

    spans: list[Span] = field(default_factory=list)
    marks: dict[str, float] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)

    # ---------- primitives ----------

    def now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def mark(self, name: str) -> float:
        """Records an instant. Idempotent: the first mark wins.

        That matters for first_token, which is called inside the streaming loop.
        """
        if name not in self.marks:
            self.marks[name] = self.now_ms()
        return self.marks[name]

    @contextmanager
    def span(self, name: str):
        s = Span(name=name, start_ms=self.now_ms())
        self.spans.append(s)
        try:
            yield s
        finally:
            s.end_ms = self.now_ms()

    @asynccontextmanager
    async def aspan(self, name: str):
        s = Span(name=name, start_ms=self.now_ms())
        self.spans.append(s)
        try:
            yield s
        finally:
            s.end_ms = self.now_ms()

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    # ---------- derived ----------

    def stage_ms(self) -> dict[str, float]:
        """Duration per stage. Concurrent spans (intent || retrieval) are kept
        separate on purpose - the sum of stages is NOT supposed to match the
        total when there is parallelism, and that gap is exactly the gain being
        measured."""
        out: dict[str, float] = {}
        for s in self.spans:
            if s.duration_ms is None:
                continue
            out[s.name] = out.get(s.name, 0.0) + s.duration_ms
        return {k: round(out[k], 2) for k in STAGE_ORDER if k in out} | {
            k: round(v, 2) for k, v in out.items() if k not in STAGE_ORDER
        }

    def model_phase_ms(self) -> float:
        """Wall time the model was working, in either topology.

        In the agent loop the model phase is the sum of the `hop:N` spans;
        `model_stream` is nested inside the last hop and would double count.
        In the fixed pipeline there are no hops, and the phase is
        prefill + stream.

        Getting this wrong is not cosmetic: dividing output tokens by the stream
        window alone reports thousands of tokens per second on an endpoint that
        delivers in one block, which describes the network and not the model.
        """
        stages = self.stage_ms()
        hops = sum(v for k, v in stages.items() if k.startswith("hop:"))
        if hops:
            return hops
        return (stages.get("model_ttft") or 0) + (stages.get("model_stream") or 0)

    def critical_path_ms(self) -> float:
        """Wall time to the first token (the number the user feels)."""
        return self.marks.get("first_token", self.now_ms())

    def parallel_saving_ms(self) -> float | None:
        """What the fan-out saved: sequential sum - wall time of the parallel leg."""
        spans = [
            s for s in self.spans if s.name in ("intent", "retrieval") and s.end_ms is not None
        ]
        if len(spans) < 2:
            return None
        seq = sum(s.duration_ms or 0 for s in spans)
        ends = [s.end_ms for s in spans if s.end_ms is not None]
        wall = max(ends) - min(s.start_ms for s in spans)
        return round(seq - wall, 2)

    def to_dict(self) -> dict[str, Any]:
        first = self.marks.get("first_token")
        last = self.marks.get("last_token")
        out_tokens = self.attrs.get("output_tokens") or 0

        # Two rates, because they diverge when the endpoint buffers.
        #
        # delivery   : tokens / stream window (first to last token). This is what
        #              is usually called "tokens/s", and it is only valid if the
        #              provider emits incrementally.
        # generation : tokens / model phase. Always valid, and it is the rate
        #              that actually describes the model.
        #
        # On a deployment that generates everything and delivers in one block,
        # the delivery rate becomes an absurd number (thousands of tok/s)
        # because it measures the network. `stream_buffered` flags that case so
        # the number is not read as performance.
        #
        # The delivery rate only means anything if there is a measurable stream
        # window. A cache hit or a block delivery closes that window at ~0 ms,
        # which would produce hundreds of thousands of "tok/s" - noise, not a
        # metric.
        delivery_tps = None
        if (
            first is not None
            and last is not None
            and (last - first) >= 5.0
            and out_tokens > 1
        ):
            delivery_tps = round((out_tokens - 1) / ((last - first) / 1000.0), 1)

        model_phase = self.model_phase_ms()
        gen_tps = None
        if out_tokens > 0 and model_phase > 0:
            gen_tps = round(out_tokens / (model_phase / 1000.0), 1)

        stream_ms = (last - first) if (first is not None and last is not None) else None
        # Below ~30 output tokens there is no way to tell "generated fast" from
        # "delivered in one block", so nothing is claimed. A cache hit does not
        # count either: there, instant delivery is the correct behaviour.
        buffered = bool(
            out_tokens > 30
            and self.attrs.get("cache_tier") not in ("l1", "l2")
            and stream_ms is not None
            and last is not None
            and last > 0
            and stream_ms < 0.25 * last
        )

        return {
            "request_id": self.request_id,
            "started_at": self.started_at,
            "stages_ms": self.stage_ms(),
            # Spans with ABSOLUTE start and end, in the order they began.
            # Duration alone cannot draw a waterfall and cannot show
            # parallelism - the offset is what most traces are missing.
            "spans": [
                {
                    "name": sp.name,
                    "start_ms": round(sp.start_ms, 2),
                    "end_ms": round(sp.end_ms, 2),
                    "duration_ms": round(sp.duration_ms, 2),
                }
                for sp in sorted(self.spans, key=lambda x: x.start_ms)
                if sp.end_ms is not None and sp.duration_ms is not None
            ],
            "marks_ms": {k: round(v, 2) for k, v in self.marks.items()},
            "first_token_ms": round(first, 2) if first is not None else None,
            "complete_ms": round(last, 2) if last is not None else None,
            "stream_ms": round(stream_ms, 2) if stream_ms is not None else None,
            "model_phase_ms": round(model_phase, 2) if model_phase else None,
            # tokens_per_s = GENERATION rate (always valid). The delivery rate
            # goes separately, next to the buffering flag.
            "tokens_per_s": gen_tps,
            "delivery_tokens_per_s": delivery_tps,
            "stream_buffered": buffered,
            "parallel_saving_ms": self.parallel_saving_ms(),
            **self.attrs,
        }


class TraceBuffer:
    """In-process ring buffer. In production this becomes an event stream plus
    an APM backend; here it is enough for the local /v1/traces/summary."""

    def __init__(self, maxlen: int = 5000) -> None:
        self._buf: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = Lock()

    def add(self, trace: dict[str, Any]) -> None:
        with self._lock:
            self._buf.append(trace)

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()


def percentile(values: list[float | None], p: float) -> float | None:
    """Percentile by linear interpolation (the same convention as numpy)."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], 2)
    k = (len(vals) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 2)


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-stage percentiles + a first-token headline, hits kept apart from misses."""
    if not traces:
        return {"count": 0}

    def pcts(raw: list[float | None]) -> dict[str, float | None]:
        vals = [v for v in raw if v is not None]
        return {
            "n": len(vals),
            "p50": percentile(vals, 50),
            "p90": percentile(vals, 90),
            "p95": percentile(vals, 95),
            "p99": percentile(vals, 99),
            "max": round(max(vals), 2) if vals else None,
        }

    hits = [t for t in traces if t.get("cache_tier") in ("l1", "l2")]
    misses = [t for t in traces if t.get("cache_tier") not in ("l1", "l2")]

    stage_names: list[str] = []
    for t in traces:
        for k in t.get("stages_ms", {}):
            if k not in stage_names:
                stage_names.append(k)
    stage_names.sort(key=lambda n: STAGE_ORDER.index(n) if n in STAGE_ORDER else 99)

    return {
        "count": len(traces),
        "cache_hit_rate": round(len(hits) / len(traces), 4),
        "cache_tiers": {
            tier: sum(1 for t in traces if t.get("cache_tier") == tier)
            for tier in ("l1", "l2", "miss")
        },
        "first_token_ms": {
            "all": pcts([t.get("first_token_ms") for t in traces]),
            "cache_miss": pcts([t.get("first_token_ms") for t in misses]),
            "cache_hit": pcts([t.get("first_token_ms") for t in hits]),
        },
        "complete_ms": {
            "all": pcts([t.get("complete_ms") for t in traces]),
            "cache_miss": pcts([t.get("complete_ms") for t in misses]),
            "cache_hit": pcts([t.get("complete_ms") for t in hits]),
        },
        "tokens_per_s": pcts([t.get("tokens_per_s") for t in traces]),
        "delivery_tokens_per_s": pcts([t.get("delivery_tokens_per_s") for t in traces]),
        "stream_buffered_rate": round(
            sum(1 for t in traces if t.get("stream_buffered")) / len(traces), 4
        ),
        "stages_ms": {
            name: pcts([t.get("stages_ms", {}).get(name) for t in traces])
            for name in stage_names
        },
        "parallel_saving_ms": pcts([t.get("parallel_saving_ms") for t in traces]),
        "errors": sum(1 for t in traces if t.get("error")),
    }

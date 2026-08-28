"""The reference budget: what each slice of a turn is *supposed* to cost.

A waterfall without a budget is descriptive - it says where the time went. With
one it becomes evaluative: it says where the time went *over*, which is the only
form an engineer can act on.

This lived in `bench/load.py`, where the report compared against it and the
interface knew nothing about it. One number, declared twice, drifts; declared
once and imported by both, it is the same promise the bench measures and the UI
draws. The API serves it at `/v1/budget` so the browser reads the same values.

Adjust these to your own target. They are a typical streaming RAG agent's
budget, not a law - and the point of the exercise is to compare a measured
number against a promised one, whatever you promised.
"""

from __future__ import annotations

# Per-span targets, in milliseconds.
BUDGET: dict[str, int] = {
    "gateway": 40,  # internal network + API gateway
    "intent": 400,  # nano classification, short output
    "retrieval": 150,  # remote index
    "model_ttft": 1050,  # prefill of the answering hop
    "first_token": 1600,  # p50, cache miss - what the user feels
    "first_token_p95": 3000,  # the tail is the experience
    "cache_hit": 300,  # complete answer from cache
    "complete": 3000,  # ~250 tokens at ~170 tok/s
}

# Span name in the trace -> budget key. The agent loop's hops have no separate
# budgeted line: the budget describes ONE model round trip, and a turn that
# spends three is over budget by construction - which is finding 13, and
# exactly what the drawing should make obvious.
SPAN_BUDGET: dict[str, str] = {
    "intent": "intent",
    "retrieval": "retrieval",
    "model_ttft": "model_ttft",
    "hop:1": "model_ttft",
}


def target_for(span_name: str) -> int | None:
    """The budgeted milliseconds for a span, or None if it has no line.

    No line is not the same as no cost: a tool span, a second hop and a cache
    lookup all return None here, and the reader should read that as "this was
    never in the budget", not as "this was free".
    """
    key = SPAN_BUDGET.get(span_name)
    return BUDGET.get(key) if key else None

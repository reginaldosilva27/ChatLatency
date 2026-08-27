"""The buffering detector, pinned to measured deployments.

`stream_buffered` is the flag that tells the reader "streaming bought you
nothing on this deployment". It is a strong claim, so it has to be wrong in
neither direction: a false positive slanders a deployment that streams fine,
and a false negative hides the single worst latency defect this engine finds.

Every case below is a real measurement (docs/FINDINGS.md 09 and 10) reduced to
its spans, which is why the numbers look arbitrary - they are not.
"""

from __future__ import annotations

from app.telemetry import Span, Trace


def trace_of(
    *,
    hops: list[tuple[float, float]],
    first: float,
    last: float,
    out: int,
    cache_tier: str = "miss",
) -> dict:
    t = Trace()
    for i, (start, end) in enumerate(hops, 1):
        t.spans.append(Span(name=f"hop:{i}", start_ms=start, end_ms=end))
    t.marks["first_token"] = first
    t.marks["last_token"] = last
    t.set(output_tokens=out, cache_tier=cache_tier)
    return t.to_dict()


def test_agent_loop_short_answer_is_not_buffered():
    """The regression this test exists for.

    gpt-4.1-mini, 2 hops, 86 tokens delivered over 556 ms in 51 chunks - one
    chunk every 11 ms, which is generation pacing the wire. The old detector
    divided the stream window by the WHOLE TURN (556 / 2710 = 21%) and flagged
    it, because hop 1 - which only chose a tool - had inflated the denominator.
    Against the hop that actually produced the text the share is 34%.
    """
    d = trace_of(hops=[(14, 1074), (1075, 2706)], first=2154, last=2710, out=86)
    assert d["answer_hop_ms"] == 1631.0
    assert d["stream_buffered"] is False


def test_low_capacity_deployment_is_buffered():
    """gpt-5.6-terra, 250 TPM: 92 tokens in 288 ms of a 3,295 ms hop (9%)."""
    d = trace_of(hops=[(14, 2900), (2905, 6200)], first=5912, last=6200, out=92)
    assert d["stream_buffered"] is True


def test_raw_byte_probe_case_is_buffered():
    """Same deployment under the probe: 476 tokens in the last 5% of the call."""
    d = trace_of(hops=[(5, 8187)], first=7754, last=8187, out=476)
    assert d["stream_buffered"] is True


def test_fixed_pipeline_single_hop_is_not_buffered():
    """One hop, 250 tokens spread over 64% of it - textbook incremental."""
    d = trace_of(hops=[(10, 3000)], first=1100, last=3000, out=250)
    assert d["stream_buffered"] is False


def test_answer_too_short_claims_nothing():
    """Under ~30 tokens the share cannot tell fast generation from a dump."""
    d = trace_of(hops=[(10, 2300)], first=2000, last=2260, out=28)
    assert d["stream_buffered"] is False


def test_cache_hit_is_never_buffered():
    """A cache hit delivers instantly because that is the point of a cache."""
    d = trace_of(hops=[(1, 6)], first=5, last=6, out=90, cache_tier="l1")
    assert d["stream_buffered"] is False


def test_answer_hop_falls_back_to_the_fixed_pipeline_spans():
    """With no `hop:N` spans the answering hop is prefill + stream."""
    t = Trace()
    t.spans.append(Span(name="model_ttft", start_ms=100, end_ms=1200))
    t.spans.append(Span(name="model_stream", start_ms=1200, end_ms=3000))
    t.marks["first_token"] = 1200
    t.marks["last_token"] = 3000
    t.set(output_tokens=250, cache_tier="miss")
    d = t.to_dict()
    assert d["answer_hop_ms"] == 2900.0
    assert d["stream_buffered"] is False

"""The correctness gate for the canonical cache tier.

Finding 07 ended with a process recommendation, and this file is it: every
cache hit needs a correctness gate, not only a latency one - a set of
adversarial pairs that MUST miss. No performance test can stand in for it,
because the failure this guards against is *fast*. The hit rate goes up, p50
goes down, and the wrong number arrives sooner.

Two different claims are pinned here, and they are not equally strict:

SAFETY  no adversarial pair may share a canonical key. A shared key means one
        question can be served the other's answer, which is the failure mode
        that made finding 06 the most serious of the eighteen. Non-negotiable.

RECALL  paraphrases that name a glossary entity must share a key, or the tier
        costs a lookup and buys nothing. Deliberately allowed to be partial:
        this tier's safety comes from declining whenever it cannot name
        (entity, attribute), so questions outside the glossary produce no key
        at all. A smaller mouth, not a tuned threshold.

The asymmetry is the design. Similarity always answers - it returns its
nearest neighbour whatever the distance - so the only defence available to it
is a cut-off, and finding 06 measured that no cut-off separates a paraphrase
from an opposite. Declining is a defence similarity does not have.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.cache import CacheEntry, LayeredCache, lc_key
from app.retrieval import GlossaryTable

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus.json"


@pytest.fixture(scope="module")
def table() -> GlossaryTable:
    return GlossaryTable(json.loads(CORPUS.read_text(encoding="utf-8"))["glossary"])


# ---------------------------------------------------------------------------
# SAFETY
# ---------------------------------------------------------------------------

# Pairs whose correct answers DIFFER. Three kinds are represented on purpose,
# because they fail for three different reasons: a different entity with the
# same attribute, the same entity with a different attribute, and a polarity
# flip with no entity at all - the last being the pair that scored 0.79 and
# broke the semantic cache.
ADVERSARIAL = [
    # same attribute, different entity - finding 07's failure, one tier up
    ("what is a good target for TTFT?", "what is a good target for inter-token latency?"),
    ("what unit is TTFT measured in?", "what unit is throughput measured in?"),
    ("what unit is TTFT measured in?", "what unit is end-to-end latency measured in?"),
    # same entity, different attribute
    ("what does TTFT measure?", "how do I measure TTFT?"),
    (
        "what is the common mistake with tokens per second?",
        "what is a good target for tokens per second?",
    ),
    # an entity the glossary does not model, against one it does
    ("what does p50 measure?", "what does p95 measure?"),
    # polarity flips with no entity - similarity scored these 0.79 and 0.72
    ("how do I enable streaming?", "how do I disable streaming?"),
    ("when should I use retrieval?", "when should I avoid retrieval?"),
]


@pytest.mark.parametrize(("first", "second"), ADVERSARIAL)
def test_no_adversarial_pair_shares_a_key(table: GlossaryTable, first: str, second: str) -> None:
    """Either the keys differ, or at least one question declined. Both are
    safe; what must never happen is two different answers on one key."""
    left, right = table.canonical_key(first), table.canonical_key(second)
    assert not (left is not None and left == right), (
        f"{first!r} and {second!r} both resolved to {left} - one would be "
        f"served the other's answer"
    )


def test_percentile_aliases_do_not_collapse_onto_p95(table: GlossaryTable) -> None:
    """Regression. `p50` and `p99` were aliases of the P95 entry, whose
    `measures` reads "below which 95 percent of requests fall" - so a p50
    question resolved to (P95, measures) and would have been served the p95
    definition. The entry models one percentile; the aliases now say so.

    This is the shape of every defect this tier can have, and the reason it is
    worth preferring: the bug was one row of one table, and reading the row
    was enough to see it. A similarity score offers nothing to read.
    """
    assert table.canonical_key("what does p50 measure?") is None
    assert table.canonical_key("what is a good target for p99?") is None
    assert table.canonical_key("what does p95 measure?") == ("P95", "measures")


# ---------------------------------------------------------------------------
# RECALL
# ---------------------------------------------------------------------------

# Paraphrases: different wording, same question, therefore one key. These are
# the hits L2 was wanted for, bought here without an embedding.
PARAPHRASES = [
    ("what unit is TTFT measured in?", "in which unit do you report time to first token?"),
    ("what is a good target for the first token?", "how fast should the first token arrive?"),
    ("what is a good target for TTFT?", "what target should time to first token have?"),
    ("what does E2E measure?", "what is end-to-end latency?"),
    ("what is the common mistake with p95?", "what pitfall should I avoid with tail latency?"),
]


@pytest.mark.parametrize(("first", "second"), PARAPHRASES)
def test_paraphrases_share_one_key(table: GlossaryTable, first: str, second: str) -> None:
    left, right = table.canonical_key(first), table.canonical_key(second)
    assert left is not None and left == right, f"{first!r} -> {left}, {second!r} -> {right}"


# Questions with no glossary entity. Declining is the whole safety property,
# so it is asserted rather than assumed.
OUT_OF_SCOPE = [
    "how does token streaming work over server sent events?",
    "why is a semantic cache risky?",
    "how do I structure a prompt so the provider caches it?",
    "what does chunking do to input tokens?",
]


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_a_question_without_an_entity_declines(table: GlossaryTable, question: str) -> None:
    assert table.canonical_key(question) is None


# Recall gaps that exist today, pinned so that a change to `_ATTRIBUTE_CUES`
# or `_METRIC_ALIASES` shows up here as a diff rather than passing unnoticed.
# Not defects: a missed paraphrase costs a model call, which is what would
# have happened anyway. Closing them means editing the cue table, and that
# edit needs its own safety pass - a looser cue can map a question onto the
# wrong attribute, which is not a recall question any more.
KNOWN_RECALL_GAPS = [
    # "how is X measured" is not among the how_to_measure cues; the near-miss
    # "how is it measured" is. Widening it to a bare "measured" would swallow
    # "what unit is TTFT measured in" and answer the wrong attribute.
    ("how do I measure tokens per second?", "how is throughput measured?"),
]


@pytest.mark.parametrize(("first", "second"), KNOWN_RECALL_GAPS)
def test_known_recall_gaps_still_gap(table: GlossaryTable, first: str, second: str) -> None:
    left, right = table.canonical_key(first), table.canonical_key(second)
    assert left != right, (
        f"{first!r} and {second!r} now share {left} - a recall gap closed. "
        f"Move this pair into PARAPHRASES."
    )


# ---------------------------------------------------------------------------
# The key itself
# ---------------------------------------------------------------------------


def test_wording_does_not_reach_the_key(table: GlossaryTable) -> None:
    """The point of the tier: two spellings of one question, one key."""
    a = table.canonical_key("what unit is TTFT measured in?")
    b = table.canonical_key("in which unit do you report time to first token?")
    assert a is not None and b is not None
    assert lc_key(*a, "en-US") == lc_key(*b, "en-US")


def test_entity_partitions_the_key() -> None:
    """Finding 07: a cache partitioned by language alone served one entity's
    number to another. The entity is IN the key here, so the collision is not
    reachable rather than filtered out afterwards."""
    assert lc_key("TTFT", "unit", "en-US") != lc_key("ITL", "unit", "en-US")


def test_attribute_partitions_the_key() -> None:
    assert lc_key("TTFT", "unit", "en-US") != lc_key("TTFT", "good_target", "en-US")


def test_locale_partitions_the_key() -> None:
    assert lc_key("TTFT", "unit", "en-US") != lc_key("TTFT", "unit", "pt-BR")


def test_the_key_is_case_insensitive_but_not_value_insensitive() -> None:
    assert lc_key("ttft", "UNIT", "en-US") == lc_key("TTFT", "unit", "en-US")


# ---------------------------------------------------------------------------
# Round trip through the cache, which is where a wrong answer would surface
# ---------------------------------------------------------------------------


def _cache() -> LayeredCache:
    return LayeredCache(redis_url=None, ttl_s=60, l2_threshold=0.95)


def _store(cache: LayeredCache, table: GlossaryTable, question: str, answer: str) -> None:
    pair = table.canonical_key(question)
    assert pair is not None, f"{question!r} produced no key to store under"
    asyncio.run(
        cache.set_l1(
            lc_key(*pair, "en-US"),
            CacheEntry(answer=answer, topic=None, locale="en-US", question=question),
        )
    )


def _fetch(cache: LayeredCache, table: GlossaryTable, question: str) -> str | None:
    pair = table.canonical_key(question)
    if pair is None:
        return None
    hit = asyncio.run(cache.get_l1(lc_key(*pair, "en-US")))
    return hit["answer"] if hit else None


def test_a_paraphrase_hits_what_another_phrasing_stored(table: GlossaryTable) -> None:
    cache = _cache()
    _store(cache, table, "what unit is TTFT measured in?", "milliseconds")
    assert _fetch(cache, table, "in which unit do you report time to first token?") == "milliseconds"


def test_another_entity_does_not_read_this_one(table: GlossaryTable) -> None:
    """The finding-07 scenario at the canonical tier: same attribute asked of a
    different entity must not be served."""
    cache = _cache()
    _store(cache, table, "what is a good target for TTFT?", "under 1000 ms")
    assert _fetch(cache, table, "what is a good target for inter-token latency?") is None


def test_a_polarity_flip_cannot_be_served(table: GlossaryTable) -> None:
    """Neither side of the pair that broke the semantic cache has a key, so
    neither can be stored under one or read from one."""
    cache = _cache()
    assert table.canonical_key("how do I enable streaming?") is None
    assert _fetch(cache, table, "how do I disable streaming?") is None


# ---------------------------------------------------------------------------
# The routing predicate
# ---------------------------------------------------------------------------


def test_every_tier_that_produces_an_answer_is_served_from_the_cache() -> None:
    """Regression, and the reason it is worth a test of its own.

    The post-cache edge used to enumerate tier names - `in ("l1", "l2")`,
    written once per topology. Adding a third tier made both branches fall
    through to the full pipeline: the trace reported `cache_tier=canonical`
    while the answer was regenerated, so the hit cost 1,043 ms instead of
    1 ms. That is finding 07's failure mode inverted - the instrument claiming
    a hit the engine did not take - and enumerating tiers is what invited it.
    """
    from app.graph import AgentRuntime

    for tier in ("l1", "canonical", "l2"):
        assert AgentRuntime._was_a_hit({"cache_tier": tier, "cached_answer": "x"}) is True
    assert AgentRuntime._was_a_hit({"cache_tier": "miss"}) is False
    # A tier name without an answer is not servable, whatever it is called.
    assert AgentRuntime._was_a_hit({"cache_tier": "canonical", "cached_answer": ""}) is False

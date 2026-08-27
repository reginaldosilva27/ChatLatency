"""Load workload with the traffic mix a documentation assistant actually sees.

Two properties define the mix:
  - most questions ask for a fact that has been asked before;
  - traffic concentrates on a few topics rather than spreading evenly.

So the topics follow a concentrated distribution and the questions repeat with
variations in wording - which exercises L1 (exact repetition) and L2
(paraphrase).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# 70% of traffic on 2 of 6 topics: real documentation traffic is concentrated
TOPIC_WEIGHTS = [
    ("latency", 0.40),
    ("streaming", 0.25),
    ("agents", 0.14),
    ("rag", 0.10),
    ("caching", 0.08),
    ("", 0.03),
]


@dataclass
class Turn:
    question: str
    topic: str | None
    locale: str
    kind: str  # attribute | concept | external | complex


# exact attribute - should resolve by lookup + cache, never by RAG
ATTRIBUTE = [
    "what unit is TTFT measured in?",
    "what is a good target for time to first token?",
    "how do I measure inter-token latency?",
    "what does p95 mean?",
    "what dominates end-to-end latency?",
    "what is the common mistake with tokens per second?",
    "what does cost per turn measure?",
]
# paraphrases of the same questions - the target of the semantic L2
ATTRIBUTE_PARAPHRASE = [
    "in which unit do you report time to first token?",
    "how fast should the first token be?",
    "what is the right way to measure the gap between tokens?",
    "what is the 95th percentile?",
    "what drives total response time?",
    "what do people get wrong about throughput?",
    "what goes into the cost of one turn?",
]
# concept question - real RAG (kb_search), mini route
CONCEPT = [
    "how does token streaming work over server sent events?",
    "why does my streaming endpoint deliver everything at once?",
    "when should I not use retrieval and use a lookup table instead?",
    "how do chunking and top_k affect cost?",
    "what does an agent loop actually do?",
    "why is a semantic cache risky?",
    "how do I structure a prompt so the provider caches it?",
]
# external - should trigger web_search, the slowest tool
EXTERNAL = [
    "what is the current industry practice for streaming LLM responses over HTTP?",
    "which vendors publish time to first token benchmarks?",
    "what do people recommend today for measuring LLM latency in production?",
]
# complex / multi-step - should chain more than one tool
COMPLEX = [
    "my first token takes 1800 ms and the model generates 60 tokens per second for "
    "a 250 token answer, and I have two model hops - how much would removing one "
    "hop actually save the user, and is that the biggest win available to me?",
    "I need to choose between a remote embedding for a semantic cache and a local "
    "one for RAG, considering hit rate, correctness and the cost on the hot path - "
    "which makes more sense and why?",
    "my p95 is four times my p50 and my agent turns average two hops, are those "
    "two facts related and what do I measure first?",
]

USER_CONTEXT = (
    "# Session context\n"
    "- Reader profile: engineer instrumenting a streaming chat for the first time\n"
    "- Stack: FastAPI + SSE, one model hop, retrieval on every turn\n"
    "- Current numbers: TTFT p50 1.8 s, p95 4.2 s, 60 tokens/s\n"
)


def pick_topic(rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for topic, w in TOPIC_WEIGHTS:
        acc += w
        if r <= acc:
            return topic
    return TOPIC_WEIGHTS[0][0]


def make_workload(
    n: int,
    seed: int = 7,
    mix: tuple[float, float, float, float] = (0.45, 0.35, 0.10, 0.10),
    paraphrase_rate: float = 0.35,
    locale: str = "en-US",
) -> list[Turn]:
    """mix = (attribute, concept, external, complex).

    The default reflects the observation that most assistant traffic is an exact
    attribute that has been asked before - the case where the cheap path resolves
    the turn and the model should never be touched."""
    rng = random.Random(seed)
    p_attr, p_concept, p_ext, _ = mix
    turns: list[Turn] = []
    for _ in range(n):
        r = rng.random()
        if r < p_attr:
            idx = rng.randrange(len(ATTRIBUTE))
            pool = ATTRIBUTE_PARAPHRASE if rng.random() < paraphrase_rate else ATTRIBUTE
            turns.append(Turn(pool[idx], pick_topic(rng), locale, "attribute"))
        elif r < p_attr + p_concept:
            turns.append(Turn(rng.choice(CONCEPT), pick_topic(rng), locale, "concept"))
        elif r < p_attr + p_concept + p_ext:
            turns.append(Turn(rng.choice(EXTERNAL), pick_topic(rng), locale, "external"))
        else:
            turns.append(Turn(rng.choice(COMPLEX), pick_topic(rng), locale, "complex"))
    return turns


def unique_cold_workload(seed: int = 7, locale: str = "en-US") -> list[Turn]:
    """One distinct question per topic: guarantees a 100% cache miss so the clean
    latency budget of the mini route can be measured."""
    rng = random.Random(seed)
    return [Turn(q, pick_topic(rng), locale, "concept") for q in CONCEPT]

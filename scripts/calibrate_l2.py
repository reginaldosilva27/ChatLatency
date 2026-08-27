"""Calibrates the L2 semantic-cache threshold with data instead of a guess.

It measures embedding similarity across three kinds of pair:
  paraphrase -> SAME intent, SAME answer          => we want a hit
  neighbour  -> similar wording, DIFFERENT answer => we want a miss
  distinct   -> different subjects                => we want a miss

The useful threshold is the one that separates the lowest paraphrase from the
highest neighbour. If those two ranges overlap, there is NO safe threshold for
that set - and the right way out is to partition the cache better (by topic, by
question type), not to loosen the cut-off.

Run it before enabling L2 in any new language or domain: the answer changes with
the vocabulary.

    PYTHONPATH=. uv run python scripts/calibrate_l2.py
"""

import asyncio

from app.config import get_settings
from app.llm import LLM

PARAPHRASES = [
    (
        "what unit is TTFT measured in?",
        "in which unit do you report time to first token?",
    ),
    (
        "how does token streaming work?",
        "how are tokens delivered to a browser incrementally?",
    ),
    (
        "what is a good target for the first token?",
        "how fast should the first token arrive?",
    ),
    (
        "how do I measure inter-token latency?",
        "what is the right way to time the gap between tokens?",
    ),
    (
        "why is a semantic cache risky?",
        "what makes reusing the answer to a similar question dangerous?",
    ),
]

# Similar questions whose answer is DIFFERENT - the real risk of a semantic cache.
NEIGHBOURS = [
    ("what is a good target for TTFT?", "what is a good target for inter-token latency?"),
    ("how do I measure p50?", "how do I measure p95?"),
    ("when should I use retrieval?", "when should I avoid retrieval?"),
    ("what does an exact cache cost?", "what does a semantic cache cost?"),
    ("how do I enable streaming?", "how do I disable streaming?"),
]

DISTINCT = [
    ("what unit is TTFT measured in?", "how do I structure a prompt so the provider caches it?"),
    ("what is the 95th percentile?", "what does chunking do to input tokens?"),
]


async def cos(llm: LLM, a: str, b: str) -> float:
    va, vb = await asyncio.gather(llm.embed(a), llm.embed(b))
    na = sum(x * x for x in va) ** 0.5
    nb = sum(x * x for x in vb) ** 0.5
    return sum(x * y for x, y in zip(va, vb, strict=True)) / (na * nb)


async def main() -> None:
    llm = LLM(get_settings())
    groups = {"paraphrase": PARAPHRASES, "neighbour": NEIGHBOURS, "distinct": DISTINCT}
    results: dict[str, list[float]] = {}

    for name, pairs in groups.items():
        sims = await asyncio.gather(*(cos(llm, a, b) for a, b in pairs))
        results[name] = list(sims)
        print(f"\n{name.upper()}")
        for (a, b), s in zip(pairs, sims, strict=True):
            print(f"  {s:.4f}  {a[:44]:44s} | {b[:44]}")

    p_min = min(results["paraphrase"])
    n_max = max(results["neighbour"])
    print("\n" + "=" * 78)
    print(f"lowest paraphrase (we want a HIT) : {p_min:.4f}")
    print(f"highest neighbour (we want a MISS): {n_max:.4f}")
    if p_min > n_max:
        thr = (p_min + n_max) / 2
        print(f"\nA safe window EXISTS. Suggested threshold: {thr:.3f}")
        print(f"  margin: {p_min - n_max:.4f}")
    else:
        print("\nThere is NO safe threshold for these pairs:")
        print("  neighbours with a different answer score ABOVE legitimate paraphrases.")
        print("  => the semantic cache needs finer partitioning (topic, question type,")
        print("     document section), not a looser cut-off.")
        print(f"  With a cut at {n_max:.3f} the cache returns a WRONG answer;")
        print(f"  with a cut at {p_min:.3f} it loses legitimate paraphrases.")


if __name__ == "__main__":
    asyncio.run(main())

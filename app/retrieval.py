"""Retrieval and structured lookup.

Two paths are routinely confused with each other:

  exact attribute of a known entity (a unit, a target, a threshold)
      -> deterministic lookup in a serving table
  open question about a concept, a guide, a policy
      -> hybrid index (BM25 + vector)

This harness implements both because their latency differs by orders of
magnitude: the lookup is sub-millisecond, the hybrid index is tens of
milliseconds. Sending an exact-attribute question through the index is the most
common avoidable cost in a retrieval augmented system.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

_TOKEN = re.compile(r"\w+", re.UNICODE)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fold(text: str) -> str:
    """Lowercase and strip diacritics, keeping word boundaries intact."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(fold(text))


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    topic: str | None
    kind: str
    locale: str
    score: float = 0.0

    def as_context(self) -> str:
        return f"[{self.id} | {self.title}] {self.text}"


class Retriever(Protocol):
    async def search(
        self, query: str, locale: str, topic: str | None = None, top_k: int = 3
    ) -> list[Chunk]: ...


# --------------------------------------------------------------------------
# Structured lookup - the serving table
# --------------------------------------------------------------------------

# Every alias a user might type for a metric. Cheap to extend, and the reason
# a deterministic lookup can answer questions phrased in prose.
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "TTFT": ("ttft", "time to first token", "first token", "primeiro token"),
    "ITL": (
        "itl",
        "inter token latency",
        "inter-token latency",
        "tpot",
        "time per output token",
        "latencia entre tokens",
    ),
    "TPS": ("tps", "tokens per second", "tokens/s", "tok/s", "throughput", "tokens por segundo"),
    "E2E": ("e2e", "end to end", "end-to-end", "total time", "total latency", "tempo total"),
    # "p50" and "p99" are deliberately NOT aliases. This entry is the 95th
    # percentile specifically - its `measures` reads "below which 95 percent of
    # requests fall" - so accepting them made "what does p50 measure?" and
    # "what does p95 measure?" the same (entity, attribute) and let the
    # canonical cache serve the p95 definition for p50. Declining is the
    # correct answer for an entity the glossary does not model; see finding 06.
    "P95": ("p95", "percentile", "percentil", "tail latency"),
    "HOP": ("hop", "hops", "round trip", "round-trip", "model call"),
    "CACHE_HIT_RATE": ("hit rate", "cache hit rate", "taxa de acerto"),
    "COST_PER_TURN": ("cost per turn", "cost per interaction", "custo por turno"),
}

# Which attribute the question is asking for. Order matters: the first match
# wins, so the more specific phrasings come first.
_ATTRIBUTE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("how_to_measure", ("how to measure", "how do i measure", "how is it measured", "como medir")),
    ("common_mistake", ("common mistake", "mistake", "pitfall", "gets wrong", "erro comum")),
    ("dominated_by", ("dominated by", "what affects", "depends on", "driven by", "depende de")),
    ("good_target", ("good target", "target", "acceptable", "should be", "how fast", "alvo")),
    ("unit", ("unit", "measured in", "what unit", "unidade")),
    ("measures", ("what does", "what is", "definition", "define", "o que e", "significa")),
)

_ATTRIBUTE_LABEL = {
    "unit": "unit",
    "measures": "what it measures",
    "how_to_measure": "how to measure it",
    "good_target": "good target",
    "dominated_by": "dominated by",
    "common_mistake": "common mistake",
}


class GlossaryTable:
    """Pre-computed serving table: entity plus exact attributes.

    Cost is a dict lookup. Running a vector search to retrieve a unit or a
    threshold wastes latency and input tokens, and it can return an
    approximately correct number - which a dict cannot.
    """

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.by_id = {e["id"].upper(): e for e in entries}

    def ids(self) -> list[str]:
        return list(self.by_id)

    def get(self, metric_id: str | None) -> dict[str, Any] | None:
        if not metric_id:
            return None
        return self.by_id.get(metric_id.upper())

    def detect_metric(self, question: str) -> str | None:
        """Find the metric a question is about, by id or alias."""
        q = " ".join(tokenize(question))
        padded = f" {q} "
        best: tuple[int, str] | None = None
        for metric_id, aliases in _METRIC_ALIASES.items():
            for alias in (metric_id, *aliases):
                needle = " ".join(tokenize(alias))
                if not needle:
                    continue
                if f" {needle} " in padded:
                    # longest alias wins: "time to first token" over "first token"
                    if best is None or len(needle) > best[0]:
                        best = (len(needle), metric_id)
        return best[1] if best else None

    async def lookup(self, metric_id: str, attribute: str | None = None) -> dict[str, Any]:
        """Tool-facing lookup: one entry, optionally one attribute.

        Async only to match the shape of every other tool - there is no I/O
        here, and that is the point: it is the cheapest tool in the box by
        several orders of magnitude.
        """
        entry = self.get(metric_id)
        if entry is None:
            return {
                "content": (
                    f"'{metric_id}' is not a known metric. "
                    f"Available: {', '.join(self.ids())}."
                ),
                "found": False,
            }
        if attribute:
            key = attribute.strip().lower()
            if key not in entry:
                available = ", ".join(k for k in entry if k not in ("id", "name", "category"))
                return {
                    "content": (
                        f"'{attribute}' is not an attribute of {entry['name']}. "
                        f"Attributes: {available}."
                    ),
                    "found": False,
                }
            return {
                "content": f"{entry['name']} ({entry['id']}) · {key} = {entry[key]}",
                "found": True,
                "attribute": key,
            }
        return {"content": json.dumps(entry, ensure_ascii=False), "found": True}

    def canonical_key(self, question: str) -> tuple[str, str] | None:
        """The `(entity, attribute)` pair a question asks for, or None.

        This is the cache key finding 06 asks for, and the reason it is safe is
        the None. Similarity always has an answer - it returns its nearest
        neighbour whatever the distance - so its only defence is a threshold,
        and finding 06 measured that no threshold separates a paraphrase from
        an opposite. This function instead *declines*: a question whose entity
        or attribute it cannot name produces no key, and no key is a miss.

        The consequence is worth stating plainly, because it is the whole
        trade. "How do I enable streaming?" and "how do I disable streaming?"
        - the pair that scored 0.79 and broke the semantic cache - name no
        entity in this corpus, so both produce None and neither can serve the
        other. Safety here is not a tuned cut-off, it is a smaller mouth.

        What is left is auditable: two questions collide only when a table says
        they name the same attribute of the same entity, and a table can be
        read. A similarity score cannot.
        """
        entry = self.get(self.detect_metric(question))
        if entry is None:
            return None

        q = " ".join(tokenize(question))
        for attribute, cues in _ATTRIBUTE_CUES:
            if attribute not in entry:
                continue
            if any(" ".join(tokenize(cue)) in q for cue in cues):
                return entry["id"], attribute
        return None

    def resolve_fixed_fact(self, question: str, topic: str | None = None) -> str | None:
        """Return a ready answer when the question asks for an exact attribute
        of a known metric. Otherwise None, and the question goes to retrieval."""
        key = self.canonical_key(question)
        if key is None:
            return None
        metric_id, attribute = key
        entry = self.by_id[metric_id]
        label = _ATTRIBUTE_LABEL.get(attribute, attribute)
        return f"{entry['name']} ({entry['id']}): {label} = {entry[attribute]}."


# --------------------------------------------------------------------------
# Local hybrid index: BM25 + topic boost
# --------------------------------------------------------------------------


class LocalHybridRetriever:
    """In-memory index with the same result shape as a managed search service.

    It does not replace a managed index in absolute numbers - it replaces it in
    *shape*: the harness measures the real cost of a hybrid retrieval step
    (tokenising, scoring, ranking) without depending on provisioned cloud. To
    measure a managed index, set RETRIEVER=search; to isolate the model's share
    of the budget, set RETRIEVER=stub.
    """

    def __init__(self, documents: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = [
            Chunk(
                id=d["id"],
                title=d["title"],
                text=d["text"],
                topic=d.get("topic"),
                kind=d.get("type", "concept"),
                locale=d.get("locale", "en-US"),
            )
            for d in documents
        ]
        self.k1, self.b = k1, b
        self._tokens = [tokenize(f"{d.title} {d.text}") for d in self.docs]
        self._tf = [Counter(t) for t in self._tokens]
        self._len = [len(t) for t in self._tokens]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0
        n = len(self.docs)
        df: Counter[str] = Counter()
        for toks in self._tokens:
            df.update(set(toks))
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def _bm25(self, query_tokens: list[str], idx: int) -> float:
        score = 0.0
        tf, dl = self._tf[idx], self._len[idx]
        for term in query_tokens:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self._idf.get(term, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
            score += idf * (f * (self.k1 + 1)) / (denom or 1)
        return score

    async def search(
        self, query: str, locale: str, topic: str | None = None, top_k: int = 3
    ) -> list[Chunk]:
        qt = tokenize(query)
        scored: list[Chunk] = []
        for i, doc in enumerate(self.docs):
            # locale and topic filters - the equivalent of a managed index filter
            if doc.locale != locale:
                continue
            score = self._bm25(qt, i)
            if topic and doc.topic and doc.topic.lower() == topic.lower():
                score *= 1.6  # topic boost
            if score <= 0:
                continue
            scored.append(
                Chunk(
                    id=doc.id,
                    title=doc.title,
                    text=doc.text,
                    topic=doc.topic,
                    kind=doc.kind,
                    locale=doc.locale,
                    score=round(score, 4),
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


class StubRetriever:
    """Sleeps a fixed time and returns a canonical passage. Isolates the
    model's share of the latency budget from the retrieval step."""

    def __init__(self, latency_ms: int, sample: list[dict[str, Any]]) -> None:
        self.latency_ms = latency_ms
        self._sample = sample

    async def search(
        self, query: str, locale: str, topic: str | None = None, top_k: int = 3
    ) -> list[Chunk]:
        import asyncio

        await asyncio.sleep(self.latency_ms / 1000.0)
        return [
            Chunk(
                id=d["id"],
                title=d["title"],
                text=d["text"],
                topic=d.get("topic"),
                kind=d.get("type", "concept"),
                locale=d.get("locale", "en-US"),
                score=1.0,
            )
            for d in self._sample[:top_k]
        ]


class AzureAISearchRetriever:
    """Real hybrid query against Azure AI Search (vector + BM25 with filters).
    Enabled with RETRIEVER=search - the path for measuring managed retrieval."""

    def __init__(self, endpoint: str, api_key: str, index: str) -> None:
        import httpx

        self.url = f"{endpoint.rstrip('/')}/indexes/{index}/docs/search?api-version=2024-07-01"
        self._client = httpx.AsyncClient(
            timeout=10.0, headers={"api-key": api_key, "Content-Type": "application/json"}
        )

    async def search(
        self, query: str, locale: str, topic: str | None = None, top_k: int = 3
    ) -> list[Chunk]:
        filters = [f"locale eq '{locale}'"]
        if topic:
            filters.append(f"topic eq '{topic}'")
        body = {
            "search": query,
            "queryType": "simple",
            "filter": " and ".join(filters),
            "top": top_k,
            "select": "id,title,text,topic,type,locale",
        }
        resp = await self._client.post(self.url, json=body)
        resp.raise_for_status()
        return [
            Chunk(
                id=d.get("id", ""),
                title=d.get("title", ""),
                text=d.get("text", ""),
                topic=d.get("topic"),
                kind=d.get("type", "concept"),
                locale=d.get("locale", locale),
                score=d.get("@search.score", 0.0),
            )
            for d in resp.json().get("value", [])
        ]


def load_corpus(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or DATA_DIR / "corpus.json").read_text(encoding="utf-8"))


def build_retriever(settings: Any) -> tuple[Retriever, GlossaryTable]:
    corpus = load_corpus()
    table = GlossaryTable(corpus["glossary"])

    if settings.retriever == "stub":
        return StubRetriever(settings.retrieval_stub_ms, corpus["documents"]), table
    if settings.retriever == "search":
        if not (settings.azure_search_endpoint and settings.azure_search_api_key):
            raise RuntimeError(
                "RETRIEVER=search requires AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY"
            )
        return (
            AzureAISearchRetriever(
                settings.azure_search_endpoint,
                settings.azure_search_api_key,
                settings.azure_search_index,
            ),
            table,
        )
    return LocalHybridRetriever(corpus["documents"]), table

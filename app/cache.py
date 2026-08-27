"""Layered cache - the most talked-about latency lever, and the least measured.

L1  exact     : normalised key (question + topic + locale) -> stored answer
L2  semantic  : embedding similarity against questions already answered
L3  provider prompt cache : not implemented here (it lives on the model side),
                but the system prompt is kept stable and at the start of the
                payload, which is the prerequisite for the provider to cache it.

One point this engine makes explicit, and almost no design prices in: L2 is
NOT free - it costs an embedding call on the hot path (~40-120 ms). The
cache_l2_embed and cache_l2_search spans measure that separately, so the
decision of whether L2 pays for itself can be made with a number.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_question(q: str) -> str:
    """Aggressive normalisation: diacritics, punctuation, case and whitespace.
    Raises the L1 hit rate without needing an embedding."""
    q = unicodedata.normalize("NFKD", q)
    q = "".join(c for c in q if not unicodedata.combining(c))
    q = _PUNCT.sub(" ", q.lower())
    return _WS.sub(" ", q).strip()


def _partition(locale: str, topic: str | None) -> str:
    """Namespace for the semantic cache. See search_l2's docstring for why."""
    return f"{locale}|{(topic or '_no_topic').lower()}"


def l1_key(question: str, topic: str | None, locale: str) -> str:
    payload = json.dumps(
        {"q": normalize_question(question), "t": (topic or "").lower(), "l": locale},
        sort_keys=True,
    )
    return "lat:l1:" + hashlib.sha256(payload.encode()).hexdigest()[:32]


@dataclass
class CacheEntry:
    answer: str
    topic: str | None
    locale: str
    question: str
    content_version: int = 1
    created_at: float = 0.0


class LayeredCache:
    """Redis backend when REDIS_URL is set; an in-process dict otherwise. The
    trace records which backend ran (`cache_backend`) so a number measured with
    a dict is never read as a production number."""

    def __init__(self, redis_url: str | None, ttl_s: int, l2_threshold: float) -> None:
        self.ttl_s = ttl_s
        self.l2_threshold = l2_threshold
        self._mem: dict[str, tuple[float, dict[str, Any]]] = {}
        self._mem_vectors: list[tuple[str, list[float], str]] = []  # (key, vec, partition)
        self._redis = None
        self.backend = "memory"

        if redis_url:
            try:
                import redis.asyncio as aioredis

                self._redis = aioredis.from_url(redis_url, decode_responses=True)
                self.backend = "redis"
            except Exception:  # noqa: BLE001 - deliberate fallback
                self._redis = None
                self.backend = "memory(redis-unavailable)"

    async def ping(self) -> bool:
        if self._redis is None:
            return True
        try:
            await self._redis.ping()
            return True
        except Exception:  # noqa: BLE001
            self.backend = "memory(redis-unreachable)"
            self._redis = None
            return False

    # ---------- L1 ----------

    async def get_l1(self, key: str) -> dict[str, Any] | None:
        if self._redis is not None:
            raw = await self._redis.get(key)
            return json.loads(raw) if raw else None

        item = self._mem.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.time():
            self._mem.pop(key, None)
            return None
        return value

    async def set_l1(self, key: str, entry: CacheEntry) -> None:
        value = {
            "answer": entry.answer,
            "question": entry.question,
            "topic": entry.topic,
            "locale": entry.locale,
            "content_version": entry.content_version,
            "created_at": entry.created_at or time.time(),
        }
        if self._redis is not None:
            await self._redis.set(key, json.dumps(value), ex=self.ttl_s)
        else:
            self._mem[key] = (time.time() + self.ttl_s, value)

    # ---------- L2 semantic ----------

    async def search_l2(
        self, vector: list[float], locale: str, topic: str | None = None
    ) -> tuple[dict[str, Any], float] | None:
        """Nearest neighbour by cosine, PARTITIONED by locale and topic.

        Partitioning is not an implementation detail, it is correctness: the same
        question asked about two different subjects is the same string -
        similarity 1.0 in embedding space - and has DIFFERENT correct answers. A
        semantic cache partitioned only by language returns the wrong answer with
        confidence, which is the worst possible failure mode. Each
        (locale, topic) pair is its own namespace.

        In production this is a vector index (RediSearch HNSW or a managed
        service) with the same filter; here it is a linear scan over the test
        set, and the span measures the real cost of the step."""
        partition = _partition(locale, topic)
        candidates = await self._all_vectors()
        best_key, best_sim = None, -1.0
        for key, vec, part in candidates:
            if part != partition or len(vec) != len(vector):
                continue
            sim = sum(a * b for a, b in zip(vector, vec, strict=False))
            if sim > best_sim:
                best_key, best_sim = key, sim

        if best_key is None or best_sim < self.l2_threshold:
            return None
        entry = await self.get_l1(best_key)
        if entry is None:
            return None
        return entry, round(best_sim, 4)

    async def index_l2(
        self, key: str, vector: list[float], locale: str, topic: str | None = None
    ) -> None:
        partition = _partition(locale, topic)
        if self._redis is not None:
            await self._redis.set(
                f"{key}:vec", json.dumps({"v": vector, "l": partition}), ex=self.ttl_s
            )
            await self._redis.sadd("lat:l2:keys", key)  # type: ignore[misc]
            await self._redis.expire("lat:l2:keys", self.ttl_s)
        else:
            # The stored value must be the PARTITION, not the bare locale:
            # search_l2 compares against the partition, so storing the locale
            # here would make the in-memory backend never hit.
            self._mem_vectors = [(k, v, p) for k, v, p in self._mem_vectors if k != key]
            self._mem_vectors.append((key, vector, partition))

    async def _all_vectors(self) -> list[tuple[str, list[float], str]]:
        if self._redis is None:
            return list(self._mem_vectors)
        keys = await self._redis.smembers("lat:l2:keys")  # type: ignore[misc]
        if not keys:
            return []
        raws = await self._redis.mget([f"{k}:vec" for k in keys])
        out = []
        for k, raw in zip(keys, raws, strict=False):
            if raw:
                d = json.loads(raw)
                out.append((k, d["v"], d["l"]))
        return out

    # ---------- invalidation by content version ----------

    async def invalidate_topic(self, topic: str) -> int:
        """Publishing content stamps a new version and invalidates the keys of
        the affected topic - invalidation by content version."""
        removed = 0
        if self._redis is None:
            for key in list(self._mem):
                value = self._mem[key][1]
                if (value.get("topic") or "").lower() == topic.lower():
                    self._mem.pop(key, None)
                    removed += 1
            self._mem_vectors = [(k, v, p) for k, v, p in self._mem_vectors if k in self._mem]
            return removed

        keys = await self._redis.smembers("lat:l2:keys")  # type: ignore[misc]
        for key in keys:
            raw = await self._redis.get(key)
            if raw and (json.loads(raw).get("topic") or "").lower() == topic.lower():
                await self._redis.delete(key, f"{key}:vec")
                await self._redis.srem("lat:l2:keys", key)  # type: ignore[misc]
                removed += 1
        return removed

    async def clear(self) -> None:
        self._mem.clear()
        self._mem_vectors.clear()
        if self._redis is not None:
            keys = await self._redis.smembers("lat:l2:keys")  # type: ignore[misc]
            for key in keys:
                await self._redis.delete(key, f"{key}:vec")
            await self._redis.delete("lat:l2:keys")

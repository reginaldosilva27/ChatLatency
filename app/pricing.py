"""A price catalogue PER MODEL, read from data/model_prices.csv.

The problem this solves: a fixed price in .env is a single in/out pair applied
to everything. This engine runs three tiers (nano/mini/frontier) that may point
at different deployments, plus a third-party model inside Stagehand. With one
global price, Stagehand's gpt-4.1 tokens get billed at frontier rates and the
cost report is wrong without anybody noticing.

Resolution order, strongest to weakest:

1. **override** - PRICE_IN_PER_MTOK / PRICE_OUT_PER_MTOK in .env. It exists for a
   negotiated contract, where list price does not apply. When filled it wins for
   ALL models, which preserves the older behaviour.
2. **catalogue** - the CSV row that matches the deployment name.
3. **fallback** - no match: it uses `_LAST_RESORT` and sets `is_placeholder`,
   which the UI already paints amber. A visibly suspicious number beats a
   silently wrong cost.

The CSV comes from `scripts/fetch_prices.py`. It is never downloaded on the hot
path - runtime network calls would be latency inside an engine whose job is
measuring latency.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .config import Settings

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "model_prices.csv"

# Used only when the model is in neither the catalogue nor an override. It is
# the list price of gpt-5.6-terra in Aug 2026 - plausible for that family, wrong
# for any other. That is why it is flagged as a placeholder.
_LAST_RESORT = (2.00, 0.20, 12.00)

# Provider prefixes the catalogue uses in its keys. Order matters: the most
# specific first, otherwise "azure/" swallows "azure/us/".
_PREFIXES = ("azure/us/", "azure/eu/", "azure_ai/", "azure/", "openai/", "azure_openai/")

_DATED = re.compile(r"-(\d{4}-\d{2}-\d{2})$")

Origin = Literal["catalog", "override", "fallback"]


@dataclass(frozen=True)
class ModelPrice:
    """A model's resolved price, with its provenance attached.

    Provenance travels with the number on purpose: a cost of $0.004 with no way
    to tell whether it came from a catalogue, a contract or a guess is not
    information.
    """

    model: str  # what was asked for (the deployment name)
    in_per_mtok: float
    cached_in_per_mtok: float
    out_per_mtok: float
    currency: str = "USD"
    origin: Origin = "catalog"
    matched: str = ""  # the CSV key that matched (may differ from `model`)
    provider: str = ""
    mode: str = ""
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    fetched_at: str = ""
    # Cached input is not always published. When it is missing we charge the
    # full input price: err high, never low.
    cached_published: bool = True

    @property
    def is_placeholder(self) -> bool:
        return self.origin == "fallback"

    @property
    def stale_days(self) -> int | None:
        """The CSV's age in days. List prices move; the consumer decides what to
        do with the age, but cannot claim not to have known."""
        if not self.fetched_at:
            return None
        try:
            d = datetime.strptime(self.fetched_at, "%Y-%m-%d").date()
        except ValueError:
            return None
        return (datetime.now(UTC).date() - d).days

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "in_per_mtok": self.in_per_mtok,
            "cached_in_per_mtok": self.cached_in_per_mtok,
            "out_per_mtok": self.out_per_mtok,
            "currency": self.currency,
            "origin": self.origin,
            "matched": self.matched,
            "is_placeholder": self.is_placeholder,
            "cached_published": self.cached_published,
            "fetched_at": self.fetched_at,
            "stale_days": self.stale_days,
        }


def _f(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _i(v: str) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class PriceBook:
    """An index over the CSV plus the deployment-name matching rules."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        prefer_prefix: str = "",
        override: tuple[float, float, float] | None = None,
        currency: str = "USD",
        aliases: dict[str, str] | None = None,
    ) -> None:
        self._rows = {r["model"].strip().lower(): r for r in rows if r.get("model")}
        self._prefer = prefer_prefix
        self._override = override
        self._currency = currency
        self._aliases = {k.lower(): v for k, v in (aliases or {}).items()}
        self._cache: dict[str, ModelPrice] = {}
        self.fetched_at = next((r.get("fetched_at", "") for r in rows), "")
        self.source = next((r.get("source", "") for r in rows), "")

    def __len__(self) -> int:
        return len(self._rows)

    # ---------- name matching ----------

    def _candidates(self, name: str) -> list[str]:
        """A deployment name -> plausible catalogue keys.

        An Azure deployment is usually named after the model, which makes the
        name a reliable signal. When it is not, PRICE_MODEL_ALIASES resolves it -
        and there is deliberately NO progressive suffix trimming: falling from
        "gpt-5.6-terra" back to "gpt-5.6" would produce a wrong price silently.
        """
        n = self._aliases.get(name, name)
        base, explicito = n, False
        for p in _PREFIXES:
            if n.startswith(p):
                base, explicito = n[len(p) :], True
                break

        # A name that ALREADY declares its provider wins
        # (STAGEHAND_MODEL="openai/gpt-4.1" runs on OpenAI directly, not on Azure -
        # applying `prefer` on top would switch the price table). With no prefix,
        # `prefer` decides, because "gpt-5.6-terra" on a Foundry resource is
        # billed by the azure/ row.
        out: list[str] = [n]
        if not explicito and self._prefer:
            out.insert(0, self._prefer + base)
        out.append(base)
        # gpt-5.5-2026-04-23 -> gpt-5.5 (the catalogue keeps the stable alias)
        for c in list(out):
            if _DATED.search(c):
                out.append(_DATED.sub("", c))

        seen, uniq = set(), []
        for c in out:
            if c and c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    def resolve(self, model: str | None) -> ModelPrice:
        name = (model or "").strip().lower()
        hit = self._cache.get(name)
        if hit is not None:
            return hit
        price = self._resolve(name)
        self._cache[name] = price
        return price

    def _resolve(self, name: str) -> ModelPrice:
        if self._override is not None:
            i, c, o = self._override
            return ModelPrice(
                model=name or "(unknown)",
                in_per_mtok=i,
                cached_in_per_mtok=c,
                out_per_mtok=o,
                currency=self._currency,
                origin="override",
                matched="PRICE_*_PER_MTOK (.env)",
            )

        for cand in self._candidates(name):
            row = self._rows.get(cand)
            if row is None:
                continue
            pin = _f(row.get("in_per_mtok", ""))
            pout = _f(row.get("out_per_mtok", ""))
            if pin is None and pout is None:
                continue
            pcached = _f(row.get("cached_in_per_mtok", ""))
            return ModelPrice(
                model=name,
                in_per_mtok=pin or 0.0,
                cached_in_per_mtok=pcached if pcached is not None else (pin or 0.0),
                out_per_mtok=pout or 0.0,
                currency=(row.get("currency") or "USD").upper(),
                origin="catalog",
                matched=cand,
                provider=row.get("provider", ""),
                mode=row.get("mode", ""),
                max_input_tokens=_i(row.get("max_input_tokens", "")),
                max_output_tokens=_i(row.get("max_output_tokens", "")),
                fetched_at=row.get("fetched_at", ""),
                cached_published=pcached is not None,
            )

        i, c, o = _LAST_RESORT
        return ModelPrice(
            model=name or "(unknown)",
            in_per_mtok=i,
            cached_in_per_mtok=c,
            out_per_mtok=o,
            currency=self._currency,
            origin="fallback",
            matched="",
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _parse_aliases(raw: str | None) -> dict[str, str]:
    """`PRICE_MODEL_ALIASES="my-deploy=gpt-5.6-terra,other=gpt-4.1"`"""
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            k, v = k.strip(), v.strip()
            if k and v:
                out[k.lower()] = v.lower()
    return out


_BOOK: tuple[Any, PriceBook] | None = None


def get_price_book(s: Settings) -> PriceBook:
    """A PriceBook memoised on the configuration that affects it. The CSV is read
    from disk once per process - on the hot path this must not cost any IO."""
    global _BOOK

    override: tuple[float, float, float] | None = None
    if s.price_in_per_mtok is not None and s.price_out_per_mtok is not None:
        cached = s.price_cached_in_per_mtok
        override = (
            s.price_in_per_mtok,
            s.price_in_per_mtok if cached is None else cached,
            s.price_out_per_mtok,
        )

    prefer = s.price_catalog_prefix
    if prefer is None:
        prefer = "azure/" if s.llm_provider in ("foundry", "azure") else ""

    path = Path(s.price_catalog_path) if s.price_catalog_path else CSV_PATH
    key = (str(path), prefer, override, s.price_currency, s.price_model_aliases)
    if _BOOK is not None and _BOOK[0] == key:
        return _BOOK[1]

    book = PriceBook(
        _read_csv(path),
        prefer_prefix=prefer,
        override=override,
        currency=s.price_currency,
        aliases=_parse_aliases(s.price_model_aliases),
    )
    _BOOK = (key, book)
    return book


def reset_cache() -> None:
    """For tests, and so /healthz can reload after a fetch."""
    global _BOOK
    _BOOK = None


__all__ = [
    "CSV_PATH",
    "ModelPrice",
    "PriceBook",
    "get_price_book",
    "reset_cache",
]

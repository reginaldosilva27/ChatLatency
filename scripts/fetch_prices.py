"""Downloads the public model price table and writes data/model_prices.csv.

Why a downloaded CSV instead of a fixed price in .env:

- list prices move (gpt-5.6-terra's own price changed on 2026-07-30). A number
  hardcoded in .env ages silently and contaminates EVERY cost report produced
  after that;
- the engine uses different models per tier (nano/mini/frontier) plus a
  third-party model inside Stagehand. A single in/out pair in .env bills
  Stagehand's gpt-4.1 tokens at frontier rates - an error that only surfaces
  when somebody checks the invoice;
- cached input stops being derived ("10% of input, that must be it") and becomes
  a published number.

Source: the LiteLLM price map (`model_prices_and_context_window.json`), the most
widely used public catalogue for this - it covers OpenAI, Azure OpenAI, Azure AI,
Anthropic, Gemini and Bedrock, with input, output and cached input per token. It
is LIST price: an enterprise contract, a long-context tier and a dedicated region
can all diverge. Check your tenant's price sheet before taking a cost into a
proposal.

Usage:
    uv run python scripts/fetch_prices.py              # curated (main families)
    uv run python scripts/fetch_prices.py --all        # the whole catalogue
    uv run python scripts/fetch_prices.py --model x,y  # include extra models
    uv run python scripts/fetch_prices.py --check      # no write; compare only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

SOURCE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "model_prices.csv"

# Accepted provider prefixes. The same family appears several times in the
# catalogue (bare = OpenAI direct, azure/ = Azure OpenAI, azure/us/ and azure/eu/
# = region pinning, which costs ~10% more). We keep them all: anyone running in a
# pinned region needs the right row, not an approximation.
PROVIDER_PREFIXES = ("", "azure/", "azure/us/", "azure/eu/", "azure_ai/")

# "Main models": families this engine could plausibly run, plus the reference
# ones used for cost comparison in a proposal.
CURATED_PREFIXES = (
    "gpt-5.6",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.3",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5-",
    "gpt-5",
    "gpt-4.1",
    "gpt-4o",
    "o3",
    "o4-mini",
    "text-embedding-3-",
    "text-embedding-ada-002",
)

# A dated suffix (gpt-5.5-2026-04-23) duplicates the unversioned row. It stays
# out of the curated set - the resolver knows how to fall from the date to the
# stable alias.
DATED = re.compile(r"-\d{4}-\d{2}-\d{2}$")

MODES = ("chat", "embedding")

FIELDS = [
    "model",
    "provider",
    "mode",
    "in_per_mtok",
    "cached_in_per_mtok",
    "out_per_mtok",
    "currency",
    "max_input_tokens",
    "max_output_tokens",
    "source",
    "fetched_at",
]


def _per_mtok(v: Any) -> str:
    """The catalogue publishes cost PER TOKEN; the engine reasons per million."""
    if v is None:
        return ""
    try:
        return f"{float(v) * 1e6:.6f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return ""


def _keep(key: str, e: dict[str, Any]) -> bool:
    if e.get("mode") not in MODES:
        return False  # tts/transcription/image do not go through the token counter
    for pref in PROVIDER_PREFIXES:
        if not key.startswith(pref):
            continue
        rest = key[len(pref) :]
        if "/" in rest:  # a provider prefix not covered above
            continue
        if DATED.search(rest):
            continue
        if any(rest.startswith(c) for c in CURATED_PREFIXES):
            return True
    return False


def _row(key: str, e: dict[str, Any], fetched_at: str) -> dict[str, str]:
    return {
        "model": key,
        "provider": e.get("litellm_provider", "") or "",
        "mode": e.get("mode", "") or "",
        "in_per_mtok": _per_mtok(e.get("input_cost_per_token")),
        "cached_in_per_mtok": _per_mtok(e.get("cache_read_input_token_cost")),
        "out_per_mtok": _per_mtok(e.get("output_cost_per_token")),
        "currency": "USD",  # the entire catalogue is quoted in USD
        "max_input_tokens": str(e.get("max_input_tokens") or ""),
        "max_output_tokens": str(e.get("max_output_tokens") or ""),
        "source": SOURCE_URL,
        "fetched_at": fetched_at,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="write the whole catalogue")
    ap.add_argument("--model", default="", help="extra models, comma separated")
    ap.add_argument("--source", default=SOURCE_URL, help="alternative catalogue URL")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--check",
        action="store_true",
        help="no write: compares the current CSV with the source, exits !=0 if they differ",
    )
    args = ap.parse_args()

    print(f"downloading {args.source} ...", file=sys.stderr)
    r = httpx.get(args.source, timeout=60.0, follow_redirects=True)
    r.raise_for_status()
    catalog: dict[str, Any] = json.loads(r.text)

    extra = {m.strip() for m in args.model.split(",") if m.strip()}
    fetched_at = datetime.now(UTC).strftime("%Y-%m-%d")

    rows: list[dict[str, str]] = []
    for key, e in catalog.items():
        if not isinstance(e, dict):
            continue  # the file's own metadata (sample_spec)
        if key in extra:
            pass
        elif args.all:
            if e.get("mode") not in MODES:
                continue
        elif not _keep(key, e):
            continue
        if not e.get("input_cost_per_token") and not e.get("output_cost_per_token"):
            continue  # a row with no published price helps nobody
        rows.append(_row(key, e, fetched_at))

    rows.sort(key=lambda x: (x["mode"], x["model"]))
    if not rows:
        print("no rows selected - check the filters", file=sys.stderr)
        return 2

    out = Path(args.out)
    if args.check:
        if not out.exists():
            print(f"{out} does not exist", file=sys.stderr)
            return 1
        current = list(csv.DictReader(out.open()))
        cmp_ = lambda rs: {  # noqa: E731 - fetched_at changes daily; ignore it
            r["model"]: (r["in_per_mtok"], r["cached_in_per_mtok"], r["out_per_mtok"]) for r in rs
        }
        a, b = cmp_(current), cmp_(rows)
        diff = [k for k in b if a.get(k) != b[k]] + [k for k in a if k not in b]
        if diff:
            print(
                f"{len(diff)} model(s) differ: {', '.join(sorted(diff)[:12])}",
                file=sys.stderr,
            )
            return 1
        print(f"OK - {len(rows)} models match the source", file=sys.stderr)
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} models -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

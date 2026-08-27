"""Runs web_browse (Stagehand) once and shows the cost of each sub-step.

It serves two purposes:

1. **Creating a session visible in the Browserbase dashboard.** Search and Fetch
   are stateless REST APIs and do NOT open a browser - which is why the Sessions
   tab stays empty even when both work. Only Stagehand creates a session.

2. **Measuring where the seconds of a ~20 s tool go.** The agent picks
   `web_browse` rarely (by design: the prompt says to use it only when Fetch
   fails), so relying on the model to exercise the tool is unreliable. Here the
   call is direct.

Usage:
    PYTHONPATH=. uv run python scripts/browse_once.py
    PYTHONPATH=. uv run python scripts/browse_once.py https://news.ycombinator.com "Extract the first headline"

Requires `uv sync --extra browse` and BROWSERBASE_API_KEY in .env.
"""

from __future__ import annotations

import asyncio
import sys
import time

from app.config import Settings
from app.tools import WebSearch

DEFAULT_URL = "https://news.ycombinator.com"
DEFAULT_INSTRUCTION = "Extract the title of the first story in the list."


async def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    instruction = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_INSTRUCTION

    s = Settings()
    if not s.browserbase_api_key:
        print("BROWSERBASE_API_KEY is missing from .env.")
        return 1

    print(f"url         : {url}")
    print(f"instruction : {instruction}")
    print(f"model       : {s.stagehand_model or '(Browserbase Model Gateway)'}")
    print(f"timeout     : {s.stagehand_timeout_s}s")
    print("\nopening a browser session...\n")

    web = WebSearch(s)
    t0 = time.perf_counter()
    r = await web.browse(url, instruction)
    wall = (time.perf_counter() - t0) * 1000
    await web.close()

    if r.error:
        print(f"FAILED: {r.error}")
        if r.meta:
            print(f"meta: {r.meta}")
        return 1

    m = r.meta
    print("extracted:")
    print(f"  {r.content[:400]}")
    print()
    print(f"session_id : {m.get('session_id')}")
    print("  (it shows up under Sessions in the Browserbase dashboard, with a replay)")
    print()
    print("where the seconds went:")
    steps = [
        ("open a browser session", m.get("launch_ms"), "FIXED cost per call - amortisable with keep_alive"),
        ("create the Stagehand client", m.get("create_ms"), "handshake"),
        ("load the page", m.get("goto_ms"), "depends on the site"),
        ("extract (runs a model)", m.get("extract_ms"), "cost PER INSTRUCTION - never amortises"),
    ]
    total = 0.0
    for name, value, note in steps:
        if value is None:
            continue
        total += value
        pct = value / wall * 100 if wall else 0
        print(f"  {name:28s} {value:8.0f} ms  {pct:5.1f}%   {note}")
    print(f"  {'-' * 28} {'-' * 8}")
    print(f"  {'total wall time':28s} {wall:8.0f} ms")
    rest = wall - total
    if rest > 1:
        print(f"  {'closing the session':28s} {rest:8.0f} ms")

    print()
    print("Compare with the other internet tools, measured in the same environment:")
    print("  web_search (Search REST) ~283 ms   ·  web_fetch (Fetch REST) ~430-730 ms")
    print(f"  web_browse (Stagehand)   {wall:.0f} ms")
    if wall > 0:
        print(f"  => browse is ~{wall / 283:.0f}x the search. Use it only when Fetch cannot resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

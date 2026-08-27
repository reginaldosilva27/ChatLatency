"""Verifies the SHAPE of the Stagehand integration, with no credential needed.

Why this exists: the integration was written from the documentation, and the
documentation is wrong on one point - the Python example does
`await browser.close()`, but `StagehandBrowser.close` is not a coroutine.
Without a shape test, that kind of divergence only shows up in production, the
first time the tool is actually called.

The test replaces `browserbase.launch` and `Stagehand.create` with doubles that
imitate the real API (verified by introspecting the installed package) and
asserts that our code:

  1. accesses `browser.context` as a property, not as a coroutine;
  2. awaits `context.pages()`;
  3. calls `page.goto(url)`;
  4. awaits `sh.extract(instruction)` and reads `.data.extraction`;
  5. awaits `sh.close()` but NOT `browser.close()`;
  6. times all four sub-steps.

Run with:  uv sync --extra browse && uv run python -m tests.test_stagehand_shape
(without the package installed it skips instead of failing)
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from app.config import Settings
from app.tools import WebSearch

calls: list[str] = []


class FakeExtractData:
    """Mirrors stagehand.client_models.DefaultExtract (field `extraction: str`)."""

    extraction = "Main heading extracted from the test page."


class FakeExtractResult:
    """Mirrors stagehand.client_models.ExtractResult (fields `data`, `metadata`)."""

    data = FakeExtractData()
    metadata = {"tokens": 123}


class FakePage:
    async def goto(self, url: str) -> None:
        calls.append(f"page.goto({url})")


class FakeContext:
    async def pages(self) -> list[FakePage]:
        calls.append("await context.pages()")
        return [FakePage()]

    async def new_page(self) -> FakePage:
        calls.append("await context.new_page()")
        return FakePage()


class FakeBrowser:
    session_id = "sess_test_123"

    @property
    def context(self) -> FakeContext:
        calls.append("browser.context (property)")
        return FakeContext()

    def close(self) -> None:
        # SYNCHRONOUS on purpose: that is how the real package behaves.
        calls.append("browser.close() [sync]")


class FakeStagehand:
    async def extract(self, instruction: str, schema: Any = None) -> FakeExtractResult:
        calls.append(f"await sh.extract({instruction!r})")
        return FakeExtractResult()

    async def close(self) -> None:
        calls.append("await sh.close()")


async def fake_launch(*, api_key: str, timeout: float | None = None, **_: Any) -> FakeBrowser:
    calls.append(f"await launch(api_key=…, timeout={timeout})")
    return FakeBrowser()


async def fake_create(*, browser: Any, model: str | None = None, **_: Any) -> FakeStagehand:
    calls.append(f"await Stagehand.create(browser=…, model={model!r})")
    return FakeStagehand()


def install_doubles() -> None:
    import stagehand

    stagehand.browserbase.launch = fake_launch  # type: ignore[assignment]

    class _S:
        create = staticmethod(fake_create)

    stagehand.Stagehand = _S  # type: ignore[assignment]


async def main() -> int:
    try:
        install_doubles()
    except ImportError:
        print("SKIPPED - the stagehand package is not installed (uv sync --extra browse)")
        return 0
    s = Settings(
        _env_file=None,
        browserbase_api_key="bb_live_double",
        stagehand_model="openai/gpt-4.1",
        web_fetch_max_chars=6000,
    )
    w = WebSearch(s)
    r = await w.browse("https://example.test/article", "Extract the main heading.")
    await w.close()

    print("call sequence:")
    for c in calls:
        print("   ", c)
    print("\nresult:", r.content[:70])
    print("error :", r.error)
    print("meta  :", {k: v for k, v in r.meta.items() if k != "url"})

    failures = []
    if r.error:
        failures.append(f"returned an error: {r.error}")
    if "extracted" not in r.content:
        failures.append("did not read .data.extraction")
    for expected in (
        "browser.context (property)",
        "await context.pages()",
        "page.goto(https://example.test/article)",
        "await sh.close()",
        "browser.close() [sync]",
    ):
        if expected not in calls:
            failures.append(f"missing: {expected}")
    for step in ("launch_ms", "create_ms", "goto_ms", "extract_ms"):
        if step not in r.meta:
            failures.append(f"did not time {step}")
    if r.meta.get("session_id") != "sess_test_123":
        failures.append("did not record session_id")

    print()
    if failures:
        for f in failures:
            print("  FAIL:", f)
        return 1
    print("  OK - the integration shape matches the installed package's API")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

"""Price per model: what has to hold regardless of the day's catalogue."""

from __future__ import annotations

import csv

import pytest

from app.config import Settings
from app.llm import Usage
from app.pricing import CSV_PATH, PriceBook, get_price_book, reset_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_csv_exists_and_has_the_env_models(rows):
    """The CSV is a versioned artefact: if it disappears, every cost becomes a
    placeholder."""
    models = {r["model"] for r in rows}
    assert "gpt-5.6-terra" in models
    assert "azure/gpt-5.6-terra" in models
    assert "text-embedding-3-small" in models


def test_csv_has_no_empty_chat_price(rows):
    for r in rows:
        if r["mode"] == "chat":
            assert r["in_per_mtok"], r["model"]
            assert r["out_per_mtok"], r["model"]


def test_resolve_prefers_the_provider_prefix(rows):
    """Foundry/Azure pays the azure/* row. In a pinned region (azure/us/) the
    price is ~10% higher - matching the wrong row is a 10% error."""
    azure = PriceBook(rows, prefer_prefix="azure/").resolve("gpt-5.6-terra")
    pinned = PriceBook(rows, prefer_prefix="azure/us/").resolve("gpt-5.6-terra")
    assert azure.matched == "azure/gpt-5.6-terra"
    assert pinned.matched == "azure/us/gpt-5.6-terra"
    assert pinned.in_per_mtok > azure.in_per_mtok


def test_resolve_falls_back_to_the_bare_name(rows):
    b = PriceBook(rows, prefer_prefix="azure/")
    # STAGEHAND_MODEL arrives as "openai/gpt-4.1"; the catalogue stores "gpt-4.1"
    assert b.resolve("openai/gpt-4.1").matched in ("azure/gpt-4.1", "gpt-4.1")


def test_alias_resolves_a_custom_deployment_name(rows):
    b = PriceBook(rows, prefer_prefix="azure/", aliases={"rag-prod-01": "gpt-5.6-terra"})
    p = b.resolve("rag-prod-01")
    assert p.origin == "catalog"
    assert p.in_per_mtok == 2.0


def test_unknown_becomes_a_placeholder_not_an_approximate_price(rows):
    """Hard rule: there is NO fuzzy matching. Falling from 'gpt-5.6-terra' to
    'gpt-5.6' would produce a wrong cost with no warning."""
    p = PriceBook(rows, prefer_prefix="azure/").resolve("deployment-that-does-not-exist")
    assert p.origin == "fallback"
    assert p.is_placeholder


def test_override_beats_the_catalogue(rows):
    b = PriceBook(rows, prefer_prefix="azure/", override=(1.0, 0.1, 5.0))
    p = b.resolve("gpt-5.6-terra")
    assert p.origin == "override"
    assert (p.in_per_mtok, p.out_per_mtok) == (1.0, 5.0)


def test_unpublished_cached_price_charges_full_input(rows):
    """Err high, never low: an assumed discount would become prompt-cache savings
    that do not exist on the invoice."""
    p = PriceBook(rows, prefer_prefix="").resolve("text-embedding-3-small")
    assert not p.cached_published
    assert p.cached_in_per_mtok == p.in_per_mtok


def test_different_tiers_are_billed_at_different_prices():
    """The catalogue's reason to exist: one global in/out pair would bill both at
    the same price and the report would add up wrong."""
    s = Settings(
        llm_provider="mock",
        nano_model="gpt-4.1-nano",
        frontier_model="gpt-5.6-terra",
        price_in_per_mtok=None,
        price_out_per_mtok=None,
    )
    book = get_price_book(s)
    nano = book.resolve(s.nano_model)
    frontier = book.resolve(s.frontier_model)
    assert nano.origin == frontier.origin == "catalog"
    assert nano.out_per_mtok < frontier.out_per_mtok


def test_usage_cost_uses_the_model_that_ran():
    s = Settings(llm_provider="mock", price_in_per_mtok=None, price_out_per_mtok=None)
    expensive = Usage(input_tokens=1000, output_tokens=1000, model="gpt-5.6-terra").cost(s)
    cheap = Usage(input_tokens=1000, output_tokens=1000, model="gpt-4.1-nano").cost(s)
    assert expensive["cost_total"] > cheap["cost_total"]
    assert expensive["price_origin"] == "catalog"
    assert expensive["price_is_placeholder"] is False


def test_usage_cost_discounts_cached_input():
    s = Settings(llm_provider="mock", price_in_per_mtok=None, price_out_per_mtok=None)
    u = Usage(input_tokens=1_000_000, cached_input_tokens=900_000, model="gpt-5.6-terra")
    c = u.cost(s)
    # 100k fresh at $2/M + 900k cached at $0.20/M
    assert c["cost_input"] == pytest.approx(0.2)
    assert c["cost_cached_input"] == pytest.approx(0.18)

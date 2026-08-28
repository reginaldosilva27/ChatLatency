"""Deciding without the model: the gate for findings 02, 08, 11 and 15.

Four findings measured the same sequential model round trip under four names,
and the fixes for them share one property that a latency test cannot check:
they replace a model's judgement with a table's. So what has to be pinned is
not how fast the table is - it is what the table decides, and when it refuses
to decide at all.

The tests are grouped by the finding they answer to, and every one of them runs
offline. That is the point rather than a convenience: a heuristic router and a
backend-aware prompt are auditable precisely because checking them needs no
provider, where checking a classifier's judgement needs the classifier.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.graph import tool_system_prompt
from app.retrieval import GlossaryTable
from app.routing import HIGH, LOW, MEDIUM, heuristic_intent
from app.tools import WebSearch, tool_schemas

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus.json"


@pytest.fixture(scope="module")
def table() -> GlossaryTable:
    return GlossaryTable(json.loads(CORPUS.read_text(encoding="utf-8"))["glossary"])


# ---------------------------------------------------------------------------
# Finding 08 - the classifier got a trivial question wrong
# ---------------------------------------------------------------------------


def test_the_question_finding_08_got_wrong(table: GlossaryTable) -> None:
    """Finding 08: a pure attribute question was classified as a documentation
    question, and the tier was routed on the wrong label. The answer came out
    right anyway, which is what let it go unnoticed.

    The heuristic cannot make that mistake for this shape of question, and not
    because it is cleverer: the glossary either names the entity and the
    attribute or it does not, and here it does. This is the finding turned into
    an assertion.
    """
    for question in (
        "what is this metric's unit?",
        "what unit is TTFT measured in?",
        "what is a good target for the first token?",
        "what does end-to-end latency measure?",
    ):
        label, confidence = heuristic_intent(question, table)
        if table.canonical_key(question) is not None:
            assert (label, confidence) == ("metric_attribute", HIGH), question


def test_high_confidence_requires_a_canonical_key(table: GlossaryTable) -> None:
    """The invariant that keeps a wrong label from carrying authority. HIGH is
    reserved for the one branch backed by both tables; everything else is a
    guess and has to say so."""
    questions = [
        "what unit is TTFT measured in?",
        "why is my agent so slow?",
        "how does token streaming work over server sent events?",
        "is TTFT or ITL more important?",
        "what is the current price according to the OpenAI website?",
    ]
    for q in questions:
        label, confidence = heuristic_intent(q, table)
        assert label
        if confidence == HIGH:
            assert table.canonical_key(q) is not None, q


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # the seeded UI question - numbers of the user's own plus a sizing verb
        (
            "My TTFT is 1800 ms with 2 hops, 60 tokens/s and a 250 token answer. "
            "What does dropping one hop save?",
            "budget_calculation",
        ),
        ("why is my agent so slow?", "complaint"),
        ("this is too slow and it is frustrating", "complaint"),
        (
            "what is the current price of gpt-4.1-mini according to the OpenAI website?",
            "external_subject",
        ),
        ("how does token streaming work over server sent events?", "concept_question"),
        ("why is a semantic cache risky?", "concept_question"),
    ],
)
def test_the_remaining_labels(table: GlossaryTable, question: str, expected: str) -> None:
    assert heuristic_intent(question, table)[0] == expected


def test_a_complaint_that_mentions_a_metric_is_still_a_complaint(
    table: GlossaryTable,
) -> None:
    """Ordering matters here more than anywhere else in the module: `complaint`
    is the only label that reaches the frontier tier, so it is the only one
    whose miss costs money."""
    assert heuristic_intent("my TTFT is terrible", table)[0] == "complaint"


def test_an_unrecognised_question_declines_rather_than_guesses(
    table: GlossaryTable,
) -> None:
    label, confidence = heuristic_intent("what should I have for lunch?", table)
    assert (label, confidence) == ("concept_question", LOW)


def test_confidence_never_exceeds_what_the_branch_earned() -> None:
    assert HIGH > MEDIUM > LOW


# ---------------------------------------------------------------------------
# Finding 02 - the lever, and the compatibility it has to keep
# ---------------------------------------------------------------------------


def test_classify_intent_false_still_switches_the_mode_off() -> None:
    """`CLASSIFY_INTENT` predates `INTENT_MODE` and is named in `.env.example`,
    the README and the `ab-intent` scenario. It keeps its one job."""
    assert Settings(classify_intent=False, intent_mode="llm").effective_intent_mode == "off"
    assert Settings(classify_intent=False, intent_mode="heuristic").effective_intent_mode == "off"


def test_intent_mode_decides_when_classify_intent_does_not_object() -> None:
    for mode in ("llm", "heuristic", "async", "off"):
        assert Settings(classify_intent=True, intent_mode=mode).effective_intent_mode == mode


def test_the_default_costs_no_round_trip() -> None:
    """Finding 02's conclusion, adopted: the label is produced locally unless
    someone asks for the model. Only `llm` puts a call on the critical path."""
    assert Settings().effective_intent_mode == "heuristic"


# ---------------------------------------------------------------------------
# Finding 11 - the collapsed tiers
# ---------------------------------------------------------------------------


def test_tiers_collapsed_is_visible() -> None:
    """Finding 11: with one deployment serving all three tiers, intent
    classification cost 1,508-1,667 ms against 400 ms budgeted. Nothing fails
    and nothing logs, which is why the engine has to say it."""
    # `llm_provider` is pinned because `tier_model` consults the per-provider
    # overrides, and without it this test reads whatever .env the developer has.
    one = Settings(llm_provider="mock", nano_model="m", mini_model="m", frontier_model="m")
    assert one.tiers_collapsed is True
    three = Settings(llm_provider="mock", nano_model="a", mini_model="b", frontier_model="c")
    assert three.tiers_collapsed is False


def test_a_credential_never_reaches_a_traceback() -> None:
    """Found by this file: a failing assertion on a Settings object printed a
    live OpenAI key into the pytest output, because pydantic renders the whole
    model. In a public repository that output is a CI log."""
    s = Settings(llm_provider="mock", openai_api_key="sk-proj-not-a-real-key")
    assert "sk-proj-not-a-real-key" not in repr(s)
    assert s.openai_api_key == "sk-proj-not-a-real-key"


# ---------------------------------------------------------------------------
# Finding 15 - what web_search actually returns
# ---------------------------------------------------------------------------


def _web_search_description(has_snippets: bool) -> str:
    schemas = tool_schemas(["TTFT"], {"web_search", "web_fetch"}, has_snippets)
    return next(s["function"]["description"] for s in schemas if s["function"]["name"] == "web_search")


def test_the_schema_stops_claiming_something_false_on_snippet_backends() -> None:
    """The description used to say "Returns ONLY titles and URLs" whatever the
    backend. True of browserbase; false of the other four, including the
    keyless duckduckgo fallback the repository runs out of the box - so the
    engine was spending a hop it had already been given the answer for."""
    with_snip = _web_search_description(True)
    without = _web_search_description(False)
    assert with_snip != without
    assert "snippet" in with_snip.lower()
    assert "ONLY titles and URLs" in without


def test_the_no_snippet_description_still_orders_the_fetch() -> None:
    """The browserbase path must not lose the instruction that finding 15 says
    it needs: with no snippet the agent has to be told to follow with a fetch."""
    assert "web_fetch" in _web_search_description(False)


def test_the_prompt_states_the_hop_cost_the_backend_actually_has() -> None:
    """Finding 18 put the hop budget in the prompt because it changes what a
    correct first move is. Finding 15 is the other half: what the budget IS
    depends on whether search comes back with something readable."""
    with_snip = tool_system_prompt(has_snippets=True)
    without = tool_system_prompt(has_snippets=False)
    assert with_snip != without
    assert "costs two of them" in without
    assert "costs two of them" not in with_snip


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("duckduckgo", True),
        ("tavily", True),
        ("brave", True),
        ("serper", True),
        ("browserbase", False),
    ],
)
def test_which_backends_return_snippets(backend: str, expected: bool) -> None:
    assert (backend in WebSearch.SNIPPET_BACKENDS) is expected


def test_has_snippets_follows_the_effective_backend_not_the_requested_one() -> None:
    """Without a credential the backend falls back to duckduckgo, and a number
    measured with one backend cannot be read as another's - the same rule the
    trace already applies to `degraded`."""
    web = WebSearch(Settings(web_search_backend="browserbase", browserbase_api_key=None))
    assert web.backend == "duckduckgo"
    assert web.degraded is True
    assert web.has_snippets is True

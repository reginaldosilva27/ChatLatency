"""Deciding without asking the model.

Finding 02 measured that classifying intent on the critical path costs 828 ms
at p50 and 2,503 ms at p95, and that the cost is structural: it is not "a short
output", it is a full network round trip, so the size of the answer barely
matters. Finding 11 measured the same call at 1,508-1,667 ms when the frontier
model served the nano tier. Finding 13 found the same round trip a third time,
wearing the name "tool decision", and drew the rule that unifies them:

    count the model round trips that happen before the first word.

What makes that call so hard to justify here is what the label is actually
spent on. It feeds exactly ONE boolean in the tier decision (`n_route`):
`intent == "complaint"`. The other three inputs to that decision - a forced
tier, question length, weak retrieval context - are already heuristics
computed locally. The round trip is buying a boolean.

So this module buys it locally, following the heuristic finding 02 itself
prescribes: entity present, attribute recognised, question length. Two of those
three already exist as measured, tested code in `retrieval.py` - which is why
this is a small file and not a classifier.

**On being wrong.** A heuristic misroutes more often than a model does, and
this one will. The defence is the same one the canonical cache uses: it reports
a confidence, it declines to `concept_question` rather than inventing a label,
and the trace records `intent_source` so a heuristic label is never read as a
model's. Finding 08 is the reminder of why that matters - there, the LLM
classified a pure attribute question as documentation and routed the tier on
the wrong label. A wrong label that announces itself as a guess is a different
object from a wrong label that arrives with a model's authority.
"""

from __future__ import annotations

from .retrieval import GlossaryTable, tokenize

# Confidence is coarse on purpose. A finer scale would imply a calibration
# nobody has measured, and this module's whole argument is against claiming
# precision it did not pay for.
HIGH = 0.9
MEDIUM = 0.6
LOW = 0.2

# Long questions are the ones a heuristic reads worst and a model reads best,
# and finding 02 names question length as part of the heuristic for exactly
# that reason. Past this, the label goes out at LOW whatever the cues say.
_LONG_QUESTION_TOKENS = 28

# The cue lists below follow `_ATTRIBUTE_CUES` in retrieval.py: short, ordered,
# and readable in one sitting. That is the property being bought - a reviewer
# can audit why a question was routed the way it was, which is not true of a
# label a model produced.

# Sizing a system the user is describing: "if my TTFT is X, what does Y save?"
_ARITHMETIC_CUES = (
    "save", "saving", "faster", "slower", "reduce", "cut", "drop", "dropping",
    "how much", "how long", "worth it", "budget", "trade off", "tradeoff",
    "instead of", "compared to", "would it",
)

# Something outside the indexed documentation. These matter more than the
# others: finding 18 measured that reaching the internet costs two hops, so a
# question routed here changes what a correct first move is.
_EXTERNAL_CUES = (
    "according to", "website", "web site", "current price", "pricing page",
    "latest", "news", "release notes", "changelog", "documentation of",
    "announced", "today", "this week", "right now", "google", "search online",
    "search the internet", "on the internet",
)

# Dissatisfaction. The only label that reaches the frontier tier in n_route,
# and therefore the only one whose miss costs real money.
_COMPLAINT_CUES = (
    "too slow", "so slow", "very slow", "unacceptable", "terrible", "awful",
    "broken", "not working", "does not work", "doesn't work", "frustrat",
    "annoying", "useless", "waste of", "fed up", "complain",
)


def _has_cue(tokens_joined: str, cues: tuple[str, ...]) -> bool:
    return any(" ".join(tokenize(cue)) in tokens_joined for cue in cues)


def heuristic_intent(question: str, table: GlossaryTable) -> tuple[str, float]:
    """A label from `INTENT_LABELS` plus a confidence, with no network.

    Ordered from the strongest signal to the weakest, and every branch that is
    not certain says so in the second element. The caller decides what a low
    confidence is worth; this function's contract is only that it never dresses
    a guess as a fact.
    """
    tokens = tokenize(question)
    joined = " ".join(tokens)
    long_question = len(tokens) > _LONG_QUESTION_TOKENS

    # 1. A complete canonical key means the glossary can name both the entity
    #    and the attribute being asked for. This is the branch that settles
    #    finding 08: the question the LLM classifier got wrong is precisely the
    #    one this names with certainty, and it names it in microseconds.
    if table.canonical_key(question) is not None:
        return "metric_attribute", HIGH

    # 2. Dissatisfaction outranks the remaining topical cues: a complaint that
    #    also mentions a metric is still a complaint, and it is the only label
    #    that changes the tier.
    if _has_cue(joined, _COMPLAINT_CUES):
        return "complaint", MEDIUM

    # 3. The user supplying their own numbers is describing their system, not
    #    asking what an attribute is. Digits alone are too weak - a metric name
    #    with a number in it would trip it - so a sizing verb has to appear too.
    if any(tok.isdigit() for tok in tokens) and _has_cue(joined, _ARITHMETIC_CUES):
        return "budget_calculation", MEDIUM

    # 4. Outside the corpus. Deliberately below arithmetic: "what would 2 hops
    #    cost" is answerable locally even when it mentions a vendor.
    if _has_cue(joined, _EXTERNAL_CUES):
        return "external_subject", MEDIUM

    # 5. An entity with no recognised attribute. Enough to say the subject is a
    #    metric, not enough to say which attribute - so MEDIUM, and the tier
    #    router treats it as an ordinary question.
    if table.detect_metric(question) is not None and not long_question:
        return "metric_attribute", MEDIUM

    # 6. The honest default. `concept_question` rather than `other` because the
    #    corpus is documentation and an unrecognised question is far more often
    #    a concept than a nothing - but at LOW, which is the part that matters.
    return "concept_question", LOW

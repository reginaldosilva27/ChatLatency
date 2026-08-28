# Findings

Twenty findings that came out of measuring, not of assuming. Most of them hold for any
stack; findings 09 to 11 are specific to Azure AI Foundry, and 12 to 18 appear once the
agent gets tools.

**Measurement environment.** Local macOS. Two rounds — one against `api.openai.com`
(`gpt-4o-mini` / `gpt-4o`) and one against Azure AI Foundry in eastus2 (`gpt-5.6-terra`).
Local hybrid index, in-process cache, 8–40 requests per scenario. **These are not
production numbers**: they answer *where the time goes*, not *what it costs in your region
with provisioned throughput*.

**One note on naming.** These measurements were taken while the corpus was a fictional
product catalogue, before it was made topic-agnostic. Two tools were renamed in the process:
`plan_lookup` → `metric_lookup` (still an in-memory dict lookup) and `estimate_usage` →
`latency_budget` (still pure arithmetic). Their latency class is identical, so the numbers
carry over. Findings 06 and 07 were measured on the old corpus and their example question
pairs are kept **verbatim**, because that is what was actually measured — rewriting them
into the new domain would turn a measurement into a story.

---

## Summary

| # | finding | consequence |
|---|---|---|
| 01 | The instrumentation is accurate to **1 ms** against simulated targets; its own overhead is ~9 ms | what the bench attributes to the model is the model's |
| 02 | Classifying intent on the critical path costs **828 ms at p50 and 2,503 ms at p95** | the largest controllable slice, and it is avoidable |
| 03 | The exact cache (L1) answers in **10 ms p50** | 30x better than the typical 300 ms target |
| 04 | The semantic cache (L2) **makes average latency worse** | it charges 360 ms of embedding on every request |
| 05 | The model tier barely moves TTFT | tiering is a cost lever, not a first-token lever |
| 06 | **The semantic cache has no safe threshold** | "enable" vs "disable" scores above a legitimate paraphrase |
| 07 | A correctness defect, found and fixed | a semantic cache without per-entity partitioning serves another entity's data |
| 08 | An LLM intent classifier gets a trivial question wrong | it routes the tier on the wrong label |
| 09 | **A deployment may simply not stream incrementally** | this explains the classic "our agent takes 8 seconds" |
| 10 | Buffering tracks the deployment's **capacity**, not the model | it changes the argument for provisioned throughput |
| 11 | A frontier model used as a classifier costs **1.5 s per turn** | put the cheap tier on the cheap job |
| 12 | **The tool is never the cost — the hop is.** One tool cost 0.06 ms in a 3,797 ms turn | optimising tools is optimising 0.002% |
| 13 | The agent loop costs **+747 ms p50** against a fixed pipeline | for exactly the same reason as finding 02 |
| 14 | Local RAG (ChromaDB + in-process ONNX) does embedding **and** search in **23–45 ms** | against 360 ms for one remote embedding |
| 15 | Search without snippets forces a **third hop**: 7,123 ms against 3,460 ms | and 7.1 s is the floor, not the ceiling |
| 16 | A tool that runs a model internally has **invisible cost** — 497 tokens and 6.4 s off the books | every tool with an LLM inside must return its own `usage` |
| 17 | A tool reported **`ok=True` on an HTTP 403**, and the model truncated its own input to 1,000 chars | a green tool call next to a failed answer is the instrument lying |
| 18 | The **hop ceiling**, not the model, ends the loop — and it did not appear in the trace | "the tool never got called" was an exhausted budget |
| 19 | The canonical key needs no threshold: **7/7 adversarial pairs kept apart**, and a paraphrase served in **1.1 ms** | finding 06's way out, implemented — plus two defects it exposed |
| 20 | A table routes as well as the model here, for **no round trip** — and the prompt was charging a hop the backend did not cost | findings 02, 08, 11 and 15, applied |

Findings 01 to 16 are measurements; 17 to 20 are what happened when the engine was made
to act on them — 17 and 18 exposed by the payload capture, 19 and 20 by building the fixes
for 06 and for 02. The bug in the Stagehand documentation (`await` on a sync
method) is not in the list because it is not about latency — it is recorded in the tools
section of the README.

---

## The reference budget

Every measurement is compared against a declared budget. The default is the typical budget
of a streaming RAG agent (adjust `BUDGET` in `bench/load.py`):

| slice | target |
|---|---|
| network + gateway | ~40 ms |
| intent (nano model, short output) | ~400 ms |
| retrieval (remote index) | ~150 ms |
| model TTFT | ~1,050 ms |
| **first token, cache miss** | **~1,600 ms p50 · <= 3,000 ms p95** |
| **cache hit, complete answer** | **< 300 ms** |
| complete answer (~250 tok at ~170 tok/s) | ~3,000 ms |

Every row of that table is a span in the trace, with the same name — which is what makes a
measured number comparable to a promised one instead of comparable to nothing.

---

## 01 · The instrumentation is trustworthy before any conclusion is

With `LLM_PROVIDER=mock` (TTFT pinned at 1,050 ms, 170 tok/s) and `RETRIEVER=stub` (150 ms):

| slice | simulated target | measured | error |
|---|---|---|---|
| intent | 400 ms | 400 ms | 0 ms |
| retrieval | 150 ms | 151 ms | +1 ms |
| TTFT | 1,050 ms | 1,051 ms | +1 ms |
| tokens/s | 170 | 176 | +3% |
| transport (network + FastAPI + SSE) | — | **2–4 ms p50** | — |

**Total overhead: ~9 ms.** That also closes the gateway line of the budget: with no API
gateway in the path, FastAPI + SSE + loopback cost 2–4 ms. The 40 budgeted milliseconds are
almost entirely gateway and private network, not application.

Run this whenever you touch the graph:

```bash
# .env: LLM_PROVIDER=mock, RETRIEVER=stub
uv run python -m bench.load --scenario cold --requests 24 --concurrency 6
```

## 02 · The largest controllable slice of the budget does not need to exist

| configuration | 1st token p50 | 1st token p95 |
|---|---|---|
| intent on the critical path | 1,359 ms | 3,409 ms |
| intent **off** the critical path | **530 ms** | **906 ms** |

**−828 ms at p50 and −2,503 ms at p95.**

Classification cost 664–700 ms measured against 400 ms budgeted, and the reason is
structural: it is not "a short output", it is **a full network round trip to a model**, and
the output size barely matters. Worse: being a second sequential call to the provider, it
**doubles the exposure to the tail** — hence p95 collapsing from 3,409 to 906 ms.

Intent is still useful (it routes the tier, it feeds analytics). It just does not need to
**block the first token**:

1. A **heuristic router** on the critical path (entity present + attribute recognised +
   question length) with LLM classification as an **asynchronous event** after the answer
   starts. Analytics loses nothing: the events arrive seconds later.
2. If it has to be an LLM, run it **in parallel with the start of generation**, not before.
3. A genuinely nano model reduces it but does not remove it — the floor is the round trip.

> **Resolved.** Option 1 shipped: `INTENT_MODE=heuristic` is the default, and `app/routing.py`
> produces the label with no network. See [finding 20](#20--a-table-routes-as-well-as-the-model-here).

```bash
uv run python -m bench.load --scenario ab-intent --requests 20 --concurrency 4
```

## 03 · The exact cache beats its promise thirty times over

**10 ms p50** for a complete answer, zero tokens, against the typical < 300 ms. Aggressive
normalisation of the question — diacritics, punctuation, case, whitespace — is cheap and
raises the hit rate at no latency cost at all.

Caveat: measured with an in-process cache. With Redis, add 1–3 ms of RTT. Still an order of
magnitude under the target.

## 04 · The semantic cache makes average latency worse

| configuration | hit rate | p50 on hit | p50 on miss | **weighted average** |
|---|---|---|---|---|
| no cache | 0% | — | 1,317 ms | 1,317 ms |
| L1 exact | 10% | **10 ms** | 1,316 ms | **1,185 ms** |
| L1 + L2 semantic | 33% | 314 ms | 1,787 ms | 1,301 ms |

L2 raises the hit rate from 10% to 33% (paraphrases start hitting) — excellent for cost. But
it charges **360 ms p50 / 1,165 ms p95 of embedding on every request**, including the ones it
does not resolve. The miss degrades from 1,316 to 1,787 ms, and the L2 hit itself lands at
314 ms, over the 300 ms target.

On the weighted average, **L1+L2 is worse than L1 alone**. L2 pays for itself in tokens, not
in time. The first-order correction is to replace the remote embedding with a **local
in-process model** (~5 ms), which inverts the whole calculation. **But read finding 06
first.**

```bash
uv run python -m bench.load --scenario cache-curve --requests 30 --concurrency 5
```

## 05 · The tier barely moves TTFT

| tier | model TTFT p50 | 1st token p50 |
|---|---|---|
| nano | 537 ms | 1,428 ms |
| mini | 552 ms | 1,243 ms |
| frontier | 573 ms | 1,153 ms |

At the same prompt size, TTFT is dominated by the round trip and the prefill, and both are
similar across tiers. Tiering still matters — for cost per turn and for generation rate —
but it should not be sold as a first-token lever. **Small sample** (n=8 per tier):
directional.

## 06 · The semantic cache has no safe threshold — and the problem is correctness

The most serious finding, and it is not about time. `scripts/calibrate_l2.py` measures
embedding similarity (`text-embedding-3-small`) on real pairs. Measured on the previous
corpus of product plans, kept verbatim:

| question pair | similarity | we want |
|---|---|---|
| "what is retention on the **PRO** plan?" · "…on the **FREE** plan?" | **0.8256** | miss |
| "what is the **monthly** quota?" · "what is the **daily** quota?" | **0.7963** | miss |
| "how do I **enable** a webhook?" · "how do I **disable** a webhook?" | **0.7896** | miss |
| "what is the **read** limit?" · "…the **write** limit?" | **0.7164** | miss |
| "how do I rotate the **production** key?" · "…the **sandbox** key?" | **0.6880** | miss |
| "how do I configure a webhook?" · "what are the steps to create a webhook?" | 0.8592 | hit |
| "how long are logs kept?" · "what is the log retention period?" | 0.8020 | hit |
| "what is the plan's monthly quota?" · "how many requests do I get per month?" | **0.4821** | hit |

- lowest legitimate paraphrase (we want a hit): **0.482**
- highest opposite-answer neighbour (we want a miss): **0.826**

The measurement was repeated when the corpus domain changed, and the inversion **grew** —
which reinforces the conclusion rather than weakening it: the problem is not one domain's
vocabulary, it is how embeddings represent polarity.

The ranges are **inverted**. No cut-off is possible:

- cut at 0.83 → the cache answers PRO retention to a FREE user, and *"how to enable"* to
  someone who asked how to disable;
- cut at 0.48 → it loses most paraphrases, and L2 has no reason to exist.

The cause is known: **embeddings capture topic, not polarity.** Swapping "enable" for
"disable", "monthly" for "daily", one entity for another changes the entire answer and moves
the vector almost not at all. In a support channel that is not a cache miss — it is a wrong
limit and a wrong procedure, stated with confidence.

**That is why L2 ships off by default** (`CACHE_L2_ENABLED=false`), with the reason recorded
in `.env.example`. It can still be turned on for measurement; the `cache-curve` scenario
turns it on by itself.

> **Resolved.** The canonical key below is implemented and shipped on
> (`CACHE_CANONICAL_ENABLED=true`). Measurements, and the two defects building it exposed,
> are in [finding 19](#19--the-canonical-key-and-the-two-defects-it-exposed).

The way out is not a looser cut-off, it is **replacing similarity with a canonical key**:

1. Extract `(entity, attribute)` from the question — which `metric_lookup` already does —
   and cache on an **exact match over that key**. That is a semantic cache with the safety of
   an exact one: "TTFT unit" and "in what unit do you report time to first token" become the
   same key; "enable" and "disable" never do.
2. Strengthen L1 normalisation: cheap, 10 ms, zero risk.
3. If a similarity-based L2 stays, it needs a verifier after the hit — which costs a model
   call, i.e. exactly what the cache existed to avoid.

```bash
PYTHONPATH=. uv run python scripts/calibrate_l2.py
```

Run it before enabling L2 in any new language or domain — the answer changes with the
vocabulary.

## 07 · Correctness defect: semantic cache partitioning

In the first version, L2 filtered by language but **not by entity**. Since "what is the
monthly quota?" is the *same string* for any plan, two different entities landed on the same
neighbour with similarity 1.0 — and the second user received the first user's limit.

Fixed in `app/cache.py`: each `(locale, entity)` is its own namespace. Evidence measured
after the fix:

```
[PRO]   cache=miss -> The monthly quota of the Pro plan is 1,000,000 requests.
[SCALE] cache=miss -> The monthly quota of the Scale plan is 10,000,000 requests.
```

Worth recording for the failure mode: **latency looks great, the hit rate goes up, and the
wrong number arrives faster.** No performance test catches this.

Hence the process recommendation: **every cache hit needs a correctness gate, not only a
latency one** — a set of adversarial pairs (enable/disable, monthly/daily, entity A/entity B)
that must miss.

A second instance of the same class of bug was found later, in the in-process backend: the
vector index stored the bare locale where the partition key was expected, so the memory-backed
L2 could never hit. Also fixed. The lesson repeats: a cache that never hits and a cache that
hits wrongly both pass a latency test.

## 08 · The intent classifier gets a trivial question wrong

Observed in testing: a pure attribute question (`"what is this entity's capacity?"`)
classified as a documentation question instead of a structured attribute. The answer came out
right — the structured lookup works regardless of intent — but the **tier was routed on the
wrong label**.

Together with finding 02, this closes the argument: LLM classification is **slow and
imprecise** for routing. As a router, a heuristic is faster *and* more predictable. As an
**asynchronous analytics event** it remains valuable — there it has time, and an occasional
error does not change the user's answer.

> **Resolved.** Both halves: the heuristic router cannot make this particular mistake, and
> `INTENT_MODE=async` is the asynchronous event. Finding 20.

## 09 · A deployment may simply not stream incrementally

This finding explains the classic *"our agent takes 8 seconds to answer"*, and it is not
about the model being slow.

Measuring the arrival of **raw bytes** (`aiter_bytes`, no line buffering on the client) for a
~350 word answer, on a `gpt-5.6-terra` GlobalStandard deployment with 250 TPM:

```
1st byte      685 ms  ->     511 bytes   (just the SSE preamble)
   ... 6.2 seconds of silence ...
burst 2     6,863 ms  ->  16,966 bytes
burst 3     6,994 ms  ->  34,162 bytes
burst 4     7,122 ms  ->  42,562 bytes
burst 5     7,253 ms  ->  60,484 bytes
burst 6     7,380 ms  ->  21,961 bytes
```

The answer is generated in full and only then dispatched, in bursts spaced by one RTT. It
holds for both endpoint forms (`/openai/v1/` and `/openai/deployments/...?api-version=`), so
it is not the API — and it is not the client, because it was measured at the raw byte level.

**Consequence:** the most-cited lever in any agent design — *"always stream, no answer waits
to be complete before it starts appearing"* — **does not work on that deployment**. The user
waits for the whole answer.

The engine detects this by itself: `stream_buffered` in the trace, and an explicit warning in
the UI. It is also why it reports **two** token rates.

### The detector's denominator, and a false positive it produced

The flag first compared the stream window against the **whole turn**, and that was wrong in a
way only the agent loop exposes. On `gpt-4.1-mini`, a perfectly incremental answer — 86
tokens, 51 chunks, one chunk every 11 ms — was flagged as buffered, because 556 ms of
streaming against a 2,710 ms turn is 21%, and half that turn was hop 1 deciding to call a
tool. The denominator has to be the **hop that produced the text** (`answer_hop_ms`), where
the same answer scores 34%.

| case | share of the turn | share of the answering hop | streams? |
|---|---|---|---|
| `gpt-4.1-mini`, 2 hops, 86 tokens | 21% — *flagged* | **34%** | yes |
| `gpt-5.6-terra`, 2 hops, 92 tokens | 5% | **9%** | no |
| `gpt-5.6-terra` under the probe, 476 tokens | 5% | **5%** | no |
| single hop, 250 tokens | 63% | **64%** | yes |

Two other signals were tried and dropped, and both are instructive:

- **Chunk cadence does not separate them.** 6.5 ms per token streaming against 3.2 ms per
  token buffered — overlapping ranges, because a buffered answer still arrives in
  RTT-spaced bursts rather than instantaneously.
- **Delivery rate over generation rate is not a second opinion.** It reduces to
  `answer_hop_ms / stream_ms` — the same number as the share, wearing a different hat.

So the flag is one signal with a threshold, not a corroborated verdict, and it is documented
as such in `app/telemetry.py`. The raw-byte probe remains the only test that settles a
deployment. The cases above are pinned in `tests/test_stream_buffered.py`.

## 10 · Buffering tracks the deployment's capacity, not the model

Same resource, same region, same client, same prompt:

| deployment | SKU / TPM | 1st content | end | behaviour |
|---|---|---|---|---|
| `gpt-4.1-mini` | GlobalStandard **89,500** | 1,291 ms | 4,270 ms | incremental |
| `gpt-5.5` | GlobalStandard **2,000** | 1,763 ms | 10,528 ms | incremental |
| `gpt-5.6-terra` | GlobalStandard **250** | 6,569 ms | 7,056 ms | **one block** |
| `gpt-5.2` | GlobalStandard **50** | 11,301 ms | 12,077 ms | **one block** |
| `gpt-4.1` | Standard **50** | 4,787 ms | 5,398 ms | **one block** |

The two high-capacity deployments stream; the three low-capacity ones deliver in a block.
**That is a correlation over 5 samples, not a proof** — but the mechanism is plausible (a
deployment with a tight quota queues and returns in batches) and the hypothesis is testable
in an afternoon: raise the capacity of a buffering deployment and repeat the raw-byte
measurement.

If it holds, it changes the argument for provisioned throughput. Normally that is justified
by "predictable latency" and the p99 tail. What the measurement suggests is stronger:
**below a certain capacity there is no streaming**, and without streaming, perceived latency
is the time of the complete answer regardless of the model. It stops being a tail
optimisation and becomes a requirement of the conversational experience.

### Retested on the same resource, with a purpose-built probe

`scripts/probe_streaming.py` reads raw bytes off the socket and reports whether a deployment
actually streams. Run against six deployments of the same Foundry resource, same region, same
client, same ~350 word prompt:

| deployment | model | TPM | 1st content | content window | bursts | verdict |
|---|---|---|---|---|---|---|
| `gpt-5.2-chat-2` | gpt-chat-latest | 3,439 | **871 ms** | 88% of 7,902 ms | 47 | incremental |
| `gpt-5.4` | gpt-5.4 | 4,000 | **879–1,280 ms** | 83–90% | 38–96 | incremental |
| `gpt-4.1-mini` | gpt-4.1-mini | 89,500 | **1,140 ms** | 76% of 5,380 ms | 58 | incremental |
| `gpt-5.5` | gpt-5.5 | 2,000 | **1,475 ms** | 85% of 9,997 ms | 91 | incremental |
| `gpt-5.6-terra` | gpt-5.6-terra | 250 | **7,627 ms** | **5% of 8,187 ms** | 32 | **one block** |

The "content window" is the share of the total spent receiving text. At 5%, the answer was
finished before the first character left the server: 1,299 ms to the first byte, then 6.3
seconds of silence, then 476 tokens in 433 ms.

The split now sits at five deployments streaming and one not, and the one that does not is
the one with the lowest capacity by an order of magnitude. It remains a correlation - the
clean test is still to raise the TPM of the buffering deployment and re-probe - but the
practical conclusion no longer depends on settling the mechanism: **on this resource, the
newest model is also the one with the worst perceived latency, and moving one tier to a
higher-capacity deployment buys 6.3 seconds of first token.**

Two things the probe surfaced along the way:

- A model's family cannot be fully inferred from a deployment name. `gpt-5.2-chat-2` runs
  `gpt-chat-latest`, which takes `max_completion_tokens` and rejects `temperature != 1` like
  the rest of the gpt-5 family, but rejects `reasoning_effort: "none"` - it only accepts
  `"medium"`. A single global `REASONING_EFFORT` therefore cannot serve three tiers pointing
  at different models; the probe retries without the parameter, the engine would need a
  per-tier setting.
- `gpt-chat-latest` is not in the public price catalogue, so any tier pointed at it resolves
  to `origin: fallback` and every cost it produces is a placeholder. Fast, but not priceable
  without `PRICE_MODEL_ALIASES`.

## 11 · A frontier model as a classifier is expensive

With `gpt-5.6-terra` serving the nano tier, intent classification cost **1,508–1,667 ms** per
turn — against 400 ms budgeted and 664 ms with a mini model.

| configuration | 1st token |
|---|---|
| with intent classification | 2,673 ms |
| without intent classification | **1,407 ms** |

**1,266 ms** of difference — finding 02, amplified. Point `NANO_MODEL` at the cheapest model
available and keep the large model only on the tier that generates the answer.

> **Resolved**, as far as code can resolve a configuration: `/healthz` publishes
> `tiers_collapsed` and the server warns at startup when one deployment serves all three
> tiers. Finding 20.

## 12 · The tool is never the cost — the hop is

Three questions, three tools, measured end to end:

| question | tool | tool time | hop 1 | hop 2 | 1st token | tool / total |
|---|---|---|---|---|---|---|
| exact attribute | `metric_lookup` | **0.06 ms** | 2,246 ms | 1,544 ms | 3,797 ms | **0.002%** |
| documentation | `kb_search` | **45 ms** | 3,912 ms | 2,048 ms | 5,873 ms | **0.8%** |
| external subject | `web_search` | **784 ms** | 1,620 ms | 2,482 ms | 4,761 ms | **16%** |

Even the tool that goes to the internet — the slowest of the set, 784 ms — accounts for 16%
of the time. The other two add up to less than the rounding. **The two model hops account for
82 to 99%.**

The practical consequence runs against instinct: optimising the tool almost never pays.
Caching the tool's result does not pay (the hop happens anyway). What pays is **reducing the
number of hops** — which is finding 13.

## 13 · The agent loop costs a whole round trip, for the same reason as finding 02

Scenario `ab-agentic`, cache off, 14 requests, documentation questions only:

| topology | 1st token p50 | 1st token p95 | composition |
|---|---|---|---|
| fixed pipeline (always retrieves) | **2,743 ms** | 4,162 ms | intent 1,580 + TTFT 1,256 |
| agent loop (model picks tools) | **3,491 ms** | 4,403 ms | hop 1 1,682 + tool 38 + hop 2 1,780 |

**+747 ms at p50.** But the interesting number is not the difference — it is the similarity:

- in the fixed pipeline, the first model call is called *intent classification* and costs
  1,580 ms;
- in the agent loop, it is called *tool decision* and costs 1,682 ms.

**It is the same structural cost with two names.** Any architecture that makes a sequential
model call before it starts answering pays ~1.6 s, and the label on that call changes
nothing.

That unifies findings 02, 11 and 13 into one rule: **count the model round trips that happen
before the first word.** That is the number that determines perceived latency. Everything
else — tool, retrieval, cache, tiering — fits in the noise by comparison.

The uncomfortable corollary for agentic architectures: the fixed pipeline **can** eliminate
its first call (finding 02, heuristic router → 530 ms). The agent loop cannot, because the
tool decision *is* the call. An agent that picks the source heuristically when it can — and
only falls back to the model when the heuristic cannot decide — has the behaviour of an agent
with the latency of a fixed pipeline.

> **Implemented, not yet measured.** `SPECULATIVE_TOOLS=true` runs the lookup before hop 1.
> The saving cannot be measured with the mock provider, which never requests a tool — see
> the limits in finding 20.

## 14 · Local RAG solves the problem in finding 04

Finding 04 showed the semantic cache making average latency worse because the remote
embedding cost **360 ms p50** on every request. The recommendation was to swap it for a local
model. Now it is measured, in the same place:

| operation | where | measured |
|---|---|---|
| remote embedding (embedding only) | provider API | **360 ms p50 / 1,165 ms p95** |
| embedding **+ vector search** | ChromaDB + ONNX MiniLM in-process | **23–45 ms** |

An order of magnitude, search included. The startup cost — 63 ms to load the ONNX model and
379 ms to index 13 documents — happens once, at boot, and never on the hot path.

With that, finding 04 inverts: **a semantic cache with a local embedding would pay for its
own latency.** What still blocks L2 is not the time — it is finding 06.

## 15 · Search without snippets forces a third hop

Measured with the search backend in fallback, comparing a question that needs the internet
against one that resolves internally:

| question | tools fired | hops | 1st token |
|---|---|---|---|
| internal (documentation) | `kb_search` (32 ms) | 2 | **3,460 ms** |
| external (industry practice) | `web_search` (703 ms) + `web_fetch` | **3** | **7,123 ms** |

The agent searched, saw it had only titles, decided to open a URL, and only then answered.
The third hop cost more than the entire search.

In that measurement `web_fetch` returned in 0.03 ms because it failed instantly for lack of a
key — meaning **7.1 s is the floor**. With a real Fetch adding ~1–3 s, the same path lands at
8–10 s. It is the same number that appears in finding 09 for a different reason, and the
comparison is worth making: there the user waited 8 s because streaming did not exist; here
they wait 8 s because the architecture stacked three model round trips.

Two ways out, and both are design, not optimisation:

1. **A search backend that returns snippets** (DuckDuckGo, Tavily, Brave and Serper all do)
   lets the agent answer on the second hop. You trade source quality for a whole hop.
   > **Resolved.** The engine was not acting on this: the prompt and the `web_search` schema
   > both stated "titles and URLs only" on every backend. They are now conditional on
   > `has_snippets`. Finding 20.
2. **Search and Fetch in the same step**: search and open the first result in parallel,
   without consulting the model in between. You pay for a Fetch that may be discarded in
   order not to pay for a hop that is certain.

## 16 · A tool that runs a model internally has invisible cost

`web_browse` uses Stagehand, and Stagehand **runs a model inside the tool** to interpret the
instruction. Those tokens are paid for, but they do not pass through our provider's `usage` —
so they did not appear in `cost_total`. The turn looked cheaper than it was.

Measured on a trivial extraction (`https://example.com`, a one-sentence instruction):

| item | value |
|---|---|
| input tokens of the internal model | **497** |
| output tokens | 29 |
| internal inference time | **6,454 ms** |
| cost at `gpt-4.1` prices (Stagehand's model) | **$0.001226** |
| ~~cost at frontier prices, as it used to be billed~~ | ~~$0.001342~~ |
| Stagehand cache status | `DISABLED` |

On a real page (Hacker News) the internal prompt was **9,307 tokens** — the tool sends an
accessibility snapshot of the page to the model, and the snapshot's size is the page's size.

The engine now extracts that from Stagehand's `metadata.usage` and records it in the trace
(`llm_input_tokens`, `llm_output_tokens`, `llm_inference_ms`, `llm_cost`,
`llm_cache_status`). The UI states explicitly that there is cost outside the total.

The rule that falls out: **every tool that calls a model must return its own `usage`, or cost
per interaction is fiction.** It holds for third-party tools and it holds for sub-agents —
anything that spends tokens away from the main counter.

And a corollary that only appeared once price moved out of `.env`: **those tokens belong to
ANOTHER model.** Stagehand runs `gpt-4.1` (`$2`/`$8` per 1M), not the frontier
(`$2`/`$12`) — with a single global `in`/`out` pair in `.env`, those 29 output tokens were
billed 50% above what was owed. A small error in absolute value and invisible by
construction: the books balance, they just balance wrong.

---

## The revised budget, with the findings applied

| slice | reference | measured | with the findings |
|---|---|---|---|
| gateway + network | 40 ms | 2–4 ms (no gateway) | 40 ms |
| intent | 400 ms | 664 ms | **0 ms** (async / heuristic) |
| L2 embedding | not budgeted | 360 ms | **0 ms** (canonical key) |
| retrieval | 150 ms | 151 ms | ~0 ms (parallel) |
| model TTFT | 1,050 ms | 533 ms | 1,050 ms |
| **1st token, cache miss** | **1,600 ms** | 1,240 ms | **~1,240 ms p50** |
| **1st token, cache hit** | < 300 ms | **10 ms** | **< 50 ms** |

The reference budget is **conservative in structure and optimistic in the slices**: the
largest controllable slice is a sequential call that does not need to exist, and the exact
cache delivers two orders of magnitude better than promised.

The engine also knocked down a sentence that appears in nearly every design. "Three-layer
cache" gets listed as a latency lever; similarity-based L2, measured, is **a latency cost and
a correctness risk**. What is left is stronger: a 10 ms exact cache over a canonical key, and
a first token with no sequential model call in front of it.

---

## What is real and what is a stub

Transparency about what is actually measured. **Three layers are real** — orchestration, the
LLM and the instrumentation. The rest is an honest stub or absent, and that is deliberate:
latency is measured by isolating.

| layer | in this repo | status |
|---|---|---|
| Front end / channel | `static/index.html` | real (the ChatLatency UI) |
| API gateway | — | **absent** — no quota, metering or gateway cache |
| Runtime | FastAPI + uvicorn | real, 1 local process |
| Orchestration | `app/graph.py` (LangGraph) | **real** — speculative fan-out works |
| Guardrails | — | **absent** |
| Agent | 1 documentation agent | real, no registry |
| **LLM** | Foundry / Azure OpenAI / OpenAI | **real** — real tokens, real cost |
| Exact cache (L1) | `app/cache.py` | real; in-process `dict`, Redis optional |
| Canonical cache | `app/cache.py` + `app/retrieval.py` | **real**, on — `(entity, attribute)`, finding 19 |
| Semantic cache (L2) | `app/cache.py` | real, **off** (finding 06) |
| Tools | `app/tools.py` | **real** — 6 tools, individually timed |
| RAG | **ChromaDB + in-process ONNX** | **real** — synthetic data |
| Internet | **Browserbase** Search + Fetch (DuckDuckGo fallback) | **real** — measured with credentials |
| Real browser | Stagehand — opt-in (`uv sync --extra browse`) | **real** — session created, 12.3 s measured |
| Retrieval (fixed pipeline) | in-memory hybrid BM25 | real code, **synthetic data** |
| Serving table | `data/corpus.json` | real code, **synthetic data** |
| Catalogue / Postgres | — | **absent** |
| Telemetry | in-process ring buffer | partial (`/v1/traces/summary`) |

Direct consequences for the numbers: the fixed pipeline's **retrieval comes out at ~0 ms**
because the BM25 index is in memory (a remote index is ~150 ms); the **L1 cache comes out at
10 ms** because it is a `dict` (with Redis, +1–3 ms); and **the 40 ms of gateway were never
measured** — only what is left of them. `kb_search` (ChromaDB) and `web_search`, on the other
hand, are real and measure what they claim to measure.

---

## Limitations

1. **Network route.** The measurements were taken from Brazil to us-east/eastus2. A number
   that supports a contract has to come from the region the project actually uses, with
   provisioned throughput.
2. **No API gateway in the path.** The reference's 40 ms were never measured.
3. **In-process cache, not Redis.** The 10 ms L1 gains 1–3 ms of RTT.
4. **Small samples** (8–40 per scenario). p95 and p99 are directional. For a 1,000-concurrent
   gate, the driver has to run distributed — this one runs from one machine.
5. **A corpus of a few dozen synthetic documents.** Real retrieval over thousands of articles
   has different latency and different ranking quality.
6. **L2 calibration on 12 pairs.** The conclusion (inverted ranges) is robust because a
   single counter-example sustains it; the optimal cut-off for a canonical key is not.
7. **The capacity × buffering correlation rests on 5 samples.** A hypothesis, not a proof.
8. **List price, not contract price.** Price comes from `data/model_prices.csv`
   (`scripts/fetch_prices.py`), resolved per model — `azure/gpt-5.6-terra` = `$2.00` input /
   `$0.20` cached / `$12.00` output per 1M, checked on 2026-08-27. Cached input is no longer
   derived, but it is **still list price**: there is a more expensive long-context tier
   (`$4`/`$18`) above a threshold, a pinned region costs ~10% more, and an enterprise contract
   has its own price (use the `PRICE_*_PER_MTOK` override). The token counters are exact; only
   the multiplier depends on your contract. The CSV ages — `--check` in CI warns when the
   source diverges, and `/healthz` publishes `fetched_at`.
9. **Browserbase measured from a single location.** Search, Fetch and Browse were measured
   from Brazil with a small sample (4 searches, 2 fetches, 2 browses). The numbers are
   order-of-magnitude, not p95.

---

## Next steps

1. Repeat `ab-intent` and `cache-curve` in the target region and SKU — the two scenarios
   whose conclusions change the architecture.
2. ~~Implement the **heuristic router**~~ — done, finding 20. What is left is the half that
   needs traffic: run `INTENT_MODE=async` against a real provider and read `intent_agrees`,
   which is the accuracy loss this item asked for and the mock cannot supply.
3. ~~Prototype the **canonical-key cache** `(entity, attribute)` from finding 06~~ — done,
   finding 19. What is left is the recall half: the cue tables miss paraphrases like "how is
   throughput measured", and widening them needs its own safety pass, because a looser cue
   maps a question onto the wrong attribute.
4. Test the **capacity × buffering hypothesis** from finding 10 by raising a deployment's TPM.
5. ~~Implement the **hybrid agent** from finding 13~~ — done as `SPECULATIVE_TOOLS`, finding
   20, but unmeasured: the mock never asks for a tool, so the hop it saves needs a real
   provider to show. Either measure it there, or teach the mock to request a tool — which
   would make finding 13's whole comparison reproducible with no credentials.

## 17 · A tool reported success while failing, and a model truncated its own input

Two defects in the same turn, both invisible until the trace started carrying what the tools
actually returned. The question was *"what is the current price of gpt-4.1-mini according to
the OpenAI website?"* and the answer was a polite apology. The waterfall showed two green
tool calls.

**`web_fetch` returned `ok=True` on an HTTP 403.** There are two statuses in a fetch and only
one was being checked: `raise_for_status` covers the call to Browserbase, which answers 200
for a fetch it performed correctly, while the status of the *page* arrives inside the payload
as `statusCode`. A 403 there produced `error=None`, `ok=True` and the content `"(empty
page)"` — a green tool call sitting next to a model apologising that it could not read the
page. The instrument said the tool worked while the turn visibly failed, which is the one
thing a measurement must never do. An unreadable page is now an error, and an empty 200 —
almost always a client-rendered page — is reported as `empty_page` rather than as content.

**The model capped its own input below the useful content.** `web_fetch` exposed `max_chars`
to the model. Asked for a price, it called `web_fetch(url, max_chars=1000)`, received 1,045
characters of navigation boilerplate, and answered that *the page was truncated* and it could
not find the price. It created the truncation and then reported it as a property of the page.
The parameter was withdrawn: the cap is a cost lever that belongs to the operator
(`WEB_FETCH_MAX_CHARS`), and the trace already reports `chars_raw` against `chars_sent`.

The general rule, and the reason both survived so long: **`result_chars` is not a result.**
A trace that records how many characters a tool returned, but not what they were, cannot
distinguish a page from an error message of the same length. The engine now captures the raw
exchange — the messages sent on each hop, the text or `tool_calls` that came back, and each
tool's real output (`TRACE_PAYLOADS`, capped per field). Both defects were visible within one
turn of switching it on.

## 18 · The hop ceiling, not the model, ends the loop

`MAX_TOOL_HOPS=3` reads like a safety net and behaves like a budget. Reaching the internet
costs **two** hops — `web_search` returns titles and URLs with no snippet (finding 15), so
`web_fetch` has to follow — plus one to answer. That is the entire ceiling, so a single
wasted first hop makes the internet unreachable.

Measured on the same question, with the tool list unchanged:

| hop | before | after the prompt states the budget |
|---|---|---|
| 1 | `metric_lookup` | **`web_search`** |
| 2 | `kb_search` | `web_fetch` |
| 3 | answer: *"you would need to check the OpenAI website"* | answer from the page |

Before, the model looked for an external price in documentation that only covers this
engine's own measurements, spent both tool hops there, and produced a fluent answer with the
internet never touched. Read from the outside that is indistinguishable from a broken tool —
and it is what "the web search stopped working" actually was.

Two changes, and the second matters more than the first:

- The prompt now states the budget and the cost of reaching the internet, because **what a
  correct first move is depends on how many moves there are.** A tool list without a step
  count is an incomplete brief.
- The trace records `hops_exhausted` when the tools were withdrawn on the last hop to force
  an answer. "The model was finished" and "the model ran out of steps" produce the same shape
  of answer — fluent, plausible, missing whatever the next tool would have found — and only
  the trace can separate them. **A cap that does not appear in the trace is a cap that gets
  blamed on the tool it silenced.**

## 19 · The canonical key, and the two defects it exposed

Finding 06 ended in a recommendation rather than a fix: replace similarity with a canonical
key over `(entity, attribute)`. This is that key, implemented and measured — and the more
useful half of the result is what building it found.

**The tier.** `app/retrieval.py` already had the parts: `detect_metric` names the entity,
`_ATTRIBUTE_CUES` names the attribute. The key is those two plus the locale (`lc_key`), so the
question's *wording* never reaches it. Storage is L1's, because an exact lookup over a derived
key needs no vector and no embedding — which is why a paraphrase costs a dict access here and
cost 360 ms on every request in finding 04.

Measured with `LLM_PROVIDER=mock` (which pins the model's contribution, so the cache tier is
the only thing varying), at `MOCK_TTFT_MS=300` and `MOCK_TOKENS_PER_S=400`, `RETRIEVER=stub`,
agent loop, in-process cache backend. The miss column is therefore a *simulated* model, quoted
only so the tiers can be read against something; what is measured here is the cache:

| turn | tier | complete |
|---|---|---|
| "what unit is TTFT measured in?" | miss | 877 ms |
| the same string, again | `l1` | **1.5 ms** |
| "in which unit do you report time to first token?" | `canonical` | **1.1 ms** |
| "what unit is inter-token latency measured in?" | miss | 890 ms |

Row three is the finding: a paraphrase nobody had asked before, answered at L1's price. Row
four is the safety property sitting in the same table — same attribute, different entity,
correctly a miss.

**Safety, on the pairs that defeated similarity.** `scripts/calibrate_canonical.py` runs
`calibrate_l2.py`'s own pair lists — imported, not copied, so the comparison cannot drift —
through the canonical key: **7 of 7 adversarial pairs kept apart, zero collisions.** Recall is
partial on purpose: 2 of the 3 paraphrase pairs that name a glossary entity share a key, and
the other 2 of 5 name no entity, so they produce no key at all.

That asymmetry *is* the mechanism. Similarity always answers — it returns its nearest
neighbour whatever the distance — so its only available defence is a threshold, and finding 06
measured that no threshold exists. A canonical key can **decline**. "How do I enable
streaming?" and "how do I disable streaming?", the pair that scored 0.79 and broke the
semantic cache, name no entity in this glossary: both produce no key, so neither can serve the
other. The safety is not a tuned cut-off, it is a smaller mouth.

And what is left is auditable. The script needs no embedding, no provider, no credential and
no network — it runs offline because a canonical key is a property of two tables you can read,
where similarity is a property of a model you can only sample.

### The two defects building it exposed

Both are the same shape — a cache claiming something it did not have — and **neither would
have failed a latency test.**

**`p50` and `p99` were aliases of the `P95` entry**, whose `measures` reads *"the value below
which 95 percent of requests fall"*. So "what does p50 measure?" resolved to `(P95, measures)`,
and the canonical cache would have served the p95 definition for p50 — in 1 ms, with
confidence. That is finding 07's failure mode reproduced inside the tier built to prevent it.
The fix was to stop claiming the aliases: the entry models one percentile, so `p50` and `p99`
now decline. Note where the bug lived and what fixing it took — one row of one table, and
reading the row. This is the property being bought, more than the milliseconds.

**The post-cache edge enumerated tier names.** `if cache_tier in ("l1", "l2")`, written twice,
once per topology. A third tier satisfied neither branch, so its hits fell through to the full
pipeline: the trace reported `cache_tier=canonical` while the answer was quietly regenerated,
and the "hit" cost **1,043 ms instead of 1 ms**. The instrument reported a hit the engine never
took, which is precisely what finding 17 established a measurement must never do. The
condition now asks whether there is an answer to serve rather than listing which tiers are
allowed to have one, so a fourth tier cannot forget to register itself. The UI carried the
same list, also twice, also fixed.

The rule worth keeping: **a new cache tier needs a correctness gate before it needs a latency
number.** `tests/test_canonical_cache.py` is that gate — adversarial pairs that must miss,
paraphrases that must hit, and the known recall gaps pinned, so that editing the cue tables
shows up here as a diff instead of passing unnoticed.

## 20 · A table routes as well as the model here

Findings 02, 11 and 13 measured the same sequential model round trip three times under three
names, and finding 13 drew the rule out of them: **count the model round trips that happen
before the first word.** This is what happened when the engine was made to act on that.

**What the round trip was buying.** The intent label feeds exactly one boolean in the tier
decision (`n_route`): `intent == "complaint"`. The other three inputs to that decision — a
forced tier, question length, weak retrieval context — were already local heuristics. So the
call on the critical path was a network round trip for one boolean, and that is the fact that
made the fix small rather than clever.

`app/routing.py` produces the label from the tables finding 19 had already built:
`canonical_key` for the strong case, `detect_metric` for the weak one, plus short cue lists
for arithmetic, externality and dissatisfaction. It is the heuristic finding 02 itself
prescribed — entity present, attribute recognised, question length — and two thirds of it
already existed as tested code.

| `INTENT_MODE` | 1st token p50 | 1st token p95 |
|---|---|---|
| `llm` — a nano call on the critical path | 1,389 ms | 1,396 ms |
| `heuristic` — `app/routing.py`, no network | **1,163 ms** | **1,169 ms** |

**−227 ms p50**, on `LLM_PROVIDER=mock` at `MOCK_TTFT_MS=1000`, `RETRIEVER=stub`, speculative
retrieval on, 20 requests at concurrency 4 (`--scenario ab-router`).

Two things about that number, and the second is the finding.

It is **not** finding 02's 828 ms, and it is not supposed to be. The mock charges 380 ms for
a `complete()` call, and speculative retrieval already hides 150 ms of it behind the stub
index, so 227 ms is the part that was not already overlapped. What transfers is the shape,
not the magnitude: against a real deployment the same subtraction is worth whatever that
deployment charges for a round trip, which finding 02 measured at 828 ms and finding 11 at
1,508–1,667 ms.

And **`ab-intent` (llm against off) returned 227 ms too** — the same number as `ab-router`
(llm against heuristic), to within a millisecond. That equality is the result worth keeping:
producing the label locally is indistinguishable, in latency, from not producing one at all.
The round trip was the whole cost. There was never a computation to speed up.

### What is not settled

**Whether the table routes as *well*, only that it routes as *fast*.** A heuristic misroutes
more often than a model, and this one will. Three defences, none of which is "it is
accurate": it reports a confidence, `HIGH` is reserved for the branch where both tables agree
(a test pins that invariant), and the trace carries `intent_source` so a heuristic label is
never read as a model's. `INTENT_MODE=async` runs both and records `intent_agrees` — that
agreement rate is the missing number, and it needs real traffic, not a mock whose classifier
returns noise.

Finding 08 is the reason this matters more than it sounds. There, the LLM classifier put a
pure attribute question in the wrong bucket and routed the tier on it. The heuristic cannot
make that particular mistake — `canonical_key` names that exact shape of question with
certainty — but it will make others, and a wrong label that announces itself as a guess is a
different object from a wrong label carrying a model's authority.

### Finding 11, made observable

Finding 11 is not a bug, it is a configuration that looks harmless: one deployment serving all
three tiers, so `intent` runs on the frontier model at 1,508–1,667 ms. Nothing fails and
nothing logs. `Settings.tiers_collapsed` now says so at startup and in `/healthz`. It is the
smallest change here and possibly the highest-yield, because the failure it names is silent
and the fix is one line of `.env`.

### Finding 15: the prompt was charging a hop the backend did not cost

`web_search` returns titles and URLs with no snippet **on browserbase**. On duckduckgo,
tavily, brave and serper it returns a summary, and every one of those already reported
`has_snippets: True` in its meta. The engine was not reading its own flag: both the tool
schema and the agent's system prompt stated *"returns ONLY titles and URLs"* unconditionally,
which instructed the model to spend a `web_fetch` hop on content it had already been handed.

That was live on the **keyless duckduckgo fallback** — the path the repository takes when
someone clones it with no credentials, which is to say the path most readers were on. Both are
now conditional on the effective backend (after the fallback, not the configured value), and
`tool_system_prompt` is resolved once at startup so the provider's prompt cache still sees a
stable prefix.

Unmeasured here for the honest reason: costing it needs a real provider and a question that
actually reaches the internet. Finding 15 priced the hop it removes at 3,663 ms.

### The two defects building this exposed

**The canonical cache was keying on questions that only mentioned a metric.** The seeded UI
question —

> "My TTFT is 1800 ms with 2 hops, 60 tokens/s and a 250 token answer. What does dropping one
> hop save?"

— resolved to `(TPS, measures)`, because "tokens/s" names TPS and "what does" cues `measures`.
So the canonical cache would have stored a budget calculation under the key for *what tokens
per second means*, and served each in answer to the other. Finding 07's failure mode, a third
time, in the tier built to prevent it.

It survived finding 19's correctness gate because every adversarial pair there was a short
attribute question, and this is a long compound one. `canonical_key` now declines a question
that names more than one metric, supplies digits of its own, or runs past 14 tokens — the
same "decline rather than guess" that makes the tier safe at all, applied to a shape the
first pass did not imagine. The gate has the class now.

**A credential reached a test traceback.** A failing assertion on a `Settings` object printed
a live OpenAI key into the pytest output, because pydantic renders the whole model. This
repository is public and that output is a CI log. Every credential field is now
`Field(repr=False)`; the values still work, they just stop travelling inside error messages.
Not a latency finding, and recorded here anyway, because the discipline is the same one
findings 17 and 18 are about: what the instrument emits is part of what the instrument is.

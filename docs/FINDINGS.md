# Findings

Sixteen findings that came out of measuring, not of assuming. Most of them hold for any
stack; findings 09 to 11 are specific to Azure AI Foundry, and 12 to 16 appear once the
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

Findings 01 to 15 are measurements. The bug in the Stagehand documentation (`await` on a sync
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

## 11 · A frontier model as a classifier is expensive

With `gpt-5.6-terra` serving the nano tier, intent classification cost **1,508–1,667 ms** per
turn — against 400 ms budgeted and 664 ms with a mini model.

| configuration | 1st token |
|---|---|
| with intent classification | 2,673 ms |
| without intent classification | **1,407 ms** |

**1,266 ms** of difference — finding 02, amplified. Point `NANO_MODEL` at the cheapest model
available and keep the large model only on the tier that generates the answer.

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
2. Implement the **heuristic router** and measure the accuracy loss against the 828 ms.
3. Prototype the **canonical-key cache** `(entity, attribute)` from finding 06 and compare hit
   rate and correctness against similarity-based L2.
4. Test the **capacity × buffering hypothesis** from finding 10 by raising a deployment's TPM.
5. Implement the **hybrid agent** from finding 13: a heuristic picks the tool when it can, and
   the model decides only when the heuristic cannot.

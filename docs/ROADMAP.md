# What ChatLatency teaches, and what it should become

A review of the whole surface — the six tools, the configuration, the UI — against one
question: **does this help someone learn where the time goes in an LLM agent, or is it here
because it was interesting to build?**

Everything below is a proposal. The numbers cited are this repository's own measurements
(`docs/FINDINGS.md`), not estimates.

---

## 1. The message

One sentence, and every part of the project should be arguing for it:

> **The wait before the first word is a property of your topology, not of your model — and
> you only know your topology by measuring it.**

That is what the findings actually show. Changing the model tier moves TTFT by ~40 ms
(finding 05). Removing one classification hop moves it by 828 ms at p50 and 2,503 ms at p95
(finding 02). The most expensive tool in the box costs 0.23 ms in a 3,797 ms turn (finding
12). The lever is never where people reach first.

A second message emerged while building it, and it deserves equal billing because three
findings and two defects belong to it:

> **An instrument that flatters itself is worse than no instrument.**

A buffering detector that flagged a healthy deployment (finding 09's addendum). A tool that
reported `ok=True` on an HTTP 403 (finding 17). A hop ceiling that ended the loop and left no
trace (finding 18). Each one produced a *confident wrong conclusion*, which is worse than an
absent one. Teaching people to distrust their own dashboards is a rarer and more valuable
lesson than teaching them to read a waterfall.

### The syllabus

Eight concepts, in the order someone should meet them:

| # | concept | where it already lives | where the UI teaches it |
|---|---|---|---|
| 1 | TTFT is what the user feels; E2E and tok/s are not | `doc-ttft-why`, finding 01 | the chronograph — **strong** |
| 2 | Everything before the model is paid in full | finding 02, `doc-latency-budget` | the waterfall — **strong** |
| 3 | Hops multiply; the tool is never the cost | findings 12, 13, 15, 18 | the note under the chart — **strong** |
| 4 | Streaming can be a lie | findings 09, 10 | `stream_buffered` flag — **strong** |
| 5 | Exact cache is nearly free; semantic trades correctness for hit rate | findings 03, 04, 06, 07 | a toggle with no visible consequence — **weak** |
| 6 | Cost comes from provider usage; tools with an LLM inside break the books | finding 16 | one line in the card — **weak** |
| 7 | Percentiles, never averages | `doc-percentiles`, the bench | **the UI violates this** — see 4.1 |
| 8 | Declare a budget, then compare measured against promised | `bench/load.py` | **absent from the UI** — see 4.2 |

The gap is clear and it is not in the engine. Concepts 1–4 are taught well by the interface.
Concepts 5–8 are only taught by reading `docs/` or running the bench, which most visitors to
an open-source repository will never do.

---

## 2. The tools, one by one

Each tool is here to occupy a **latency decade**. That is the actual curriculum: six orders
of magnitude between the fastest and the slowest, so the waterfall has something to show.
The question for each is whether it earns its maintenance.

| tool | measured | what it is the only teacher of | keys / deps | verdict |
|---|---|---|---|---|
| `metric_lookup` | 0.03–0.08 ms | the floor — makes "the tool is never the cost" visible | none | **keep** |
| `latency_budget` | ~0.01 ms | perceived-latency arithmetic; its *output* is a lesson | none | **keep, promote** |
| `kb_search` | 23–45 ms | local RAG; in-process embeddings beat a remote call 15:1 | chromadb (heavy) | **keep** |
| `web_search` | 283–1,062 ms | the network round trip; no snippets forces a third hop | key (falls back) | **keep** |
| `web_fetch` | 430–1,978 ms | payload size → input tokens → cost of the *next* hop | key | **keep** |
| `web_browse` | ~12.3 s | the 10-second decade; hidden LLM cost inside a tool | key + extra dep + Stagehand churn | **keep, demote** |

### Notes on the two at the edges

**`latency_budget` is the most underused asset in the repository.** It is the only tool whose
answer teaches the subject rather than reporting a fact — you hand it a TTFT, a rate and a hop
count and it tells you what the user will feel. It should be the first seed question, and its
output should render as a small chart in the card rather than as prose the model paraphrases.

**`web_browse` has the worst lesson-to-maintenance ratio.** It teaches two real things (the
10-second decade, and finding 16's invisible token cost) but costs an optional dependency, a
Browserbase account, and exposure to Stagehand's API — the README already documents a bug in
their own docs. Keep it, because finding 16 is one of the best in the set, but stop treating
it as a first-class tool: it should be a documented experiment, not a row in the default
table. **Proposal B below reproduces its main lesson for free.**

### What is missing

Three tools would each teach something no current tool teaches, and two of them need no
credentials at all — which matters more than it sounds for an open-source project, because
today the most interesting lessons are gated behind a Browserbase key.

**A. `simulate_tool(ms, label)` — a tool that takes exactly as long as you tell it.**
The highest immersion-per-line change available. It makes every topology lesson reproducible
with zero credentials: set it to 3,000 ms and watch the first token move by 3,000 ms; request
it twice in one hop and watch the waterfall show one 3,000 ms bar instead of 6,000. This is
honest in exactly the way `LLM_PROVIDER=mock` and `RETRIEVER=stub` are honest — a declared
simulator, labelled as one in the trace, which the repository already relies on to prove its
own instrumentation is accurate to 1 ms (finding 01).

**B. `summarize(text)` — a tool that calls the nano model internally.**
Reproduces finding 16 (a tool with an LLM inside spends tokens that never reach the turn's
`usage`) using the model you already have configured, instead of a 12-second browser session.
It turns the repository's most subtle cost lesson into something anyone can trigger in one
turn.

**C. A tool that fails on demand.**
Timeouts and retries on the critical path are a whole class of latency nobody measures until
production. Lower priority than A and B, but it is the natural home for the "what does a retry
cost you" lesson, which the project currently does not teach at all.

### What to cut

Not because it is bad, but because configuration surface is a barrier and this project has
**51 environment variables**.

| surface | why it should go or shrink |
|---|---|
| `redis` in core dependencies | the import is already lazy; it belongs in an extra, not in everyone's install |
| `RETRIEVER=search` (Azure AI Search) | needs an Azure resource almost no reader has; its lesson — remote index vs. local — is already made by finding 14's numbers. Demote to a documented adapter or drop |
| `tavily` / `brave` / `serper` backends | three extra key fields for the same lesson `browserbase` and `duckduckgo` already teach: a remote round trip is a remote round trip |
| `.env.example` as one flat list | split into **"the eight you will actually touch"** and an appendix. The knob count is itself intimidating |

---

## 3. The single biggest gap: the UI does not have a curriculum

The README contains **"Four experiments, in about ten minutes"** — the best teaching asset in
the project. The interface does not know it exists. A visitor lands on a chat box, asks
something, gets a beautiful trace, and has no idea what to do next.

### 3.1 The Lab — guided experiments in the interface

A panel next to Config with the four experiments already written in the README, each as a
runnable card:

```
┌─ EXPERIMENT 2 · THE HOP ─────────────────────────────┐
│ Claim: the tool is not the cost — the hop is.        │
│                                                      │
│ Runs the same question twice: once where the model   │
│ answers directly, once where it must call a tool.    │
│                                                      │
│ Expect: +700 to +1,200 ms on the first token,        │
│ and a tool bar under 1 ms.                           │
│                                          [ run it ]  │
└──────────────────────────────────────────────────────┘
                        ↓ after the run
        measured: +1,081 ms · tool 0.14 ms · claim holds
```

Each experiment sets its own toggles, runs the turns, and states whether the measured result
matched the claim. **A claim that fails should say so** — that is the whole ethos of the
repository, and an experiment harness that can only confirm is the flattering instrument
finding 17 warns about.

This is the change that turns a demo into a course, and it needs no new engine capability:
every lever is already per-request in the POST body.

### 3.2 Baseline pinning — make every toggle self-teaching

Pin any turn as the baseline. Every later turn then shows Δ per span against it:

```
first token   2,154 ms   ▼ −1,266 ms vs. baseline
hop 1         1,060 ms   ▼ −1,508 ms   (intent classification removed)
```

Today, comparing two configurations means remembering four numbers from a card that scrolled
away. This is the difference between "the toggle changed something" and "the toggle cost me
1.2 seconds", and it makes the Caching section (syllabus concept 5, currently weak) teach
itself for the first time.

### 3.3 The budget, drawn on the waterfall

`bench/load.py` declares a reference budget per span — ~40 ms gateway, ~400 ms intent,
~150 ms retrieval, ~1,050 ms TTFT. The UI never mentions it. Drawing a target tick on each bar
turns the waterfall from *descriptive* ("here is where the time went") into *evaluative*
("here is where it went over"), which is the form an engineer can act on. This is syllabus
concept 8, currently absent from the interface entirely.

### 3.4 Point at the corpus

Every note in the card — the hop lesson, the buffering warning, the cache floor — has a
document in the indexed corpus that explains it in full. The UI never links to them. A
`why?` link on each note, opening the corresponding doc, connects the measurement to the
concept at the exact moment the reader is curious. The retrieval is already there; only the
link is missing.

---

## 4. Honesty defects found during this review

Both are the same species as findings 17 and 18: the instrument stating more than it measured.

### 4.1 The header reports a p95 it does not have

`renderMeters` computes `p95` from the first turn onward. With `n=1`, p95 equals p50 and the
header prints two identical numbers as if they were two measurements. With `n=4` it is the
maximum wearing a percentile's name.

The repository ships a document called *"Report percentiles, never averages"* and a bench that
handles this correctly. The interface should not violate its own curriculum: show `n`,
suppress p95 below ~20 samples, and say why it is suppressed.

### 4.2 A "New chat" that leaves the session statistics running

Deliberate, and defensible — percentiles improve with samples, and Config has a separate
control. But the header keeps showing accumulated `TURNS` and `SPEND` above an empty
transcript, which reads as a bug even when it is a decision. It should be stated in the UI,
not just in a code comment.

---

## 5. Sequence

Ordered by lesson delivered per hour of work.

**Phase 1 — focus and honesty** (small, no new concepts)
1. ✅ Fix the p95 report; show `n` (4.1) — suppressed below 20 samples, with the reason
2. ✅ Draw the budget on the waterfall (3.3) — `app/budget.py`, served at `/v1/budget`
3. ✅ `why?` links from each note to its corpus document (3.4) — `/v1/corpus/doc/{id}`
4. ✅ Split `.env.example`; move `redis` to an extra (2)

**Phase 2 — immersion** (the actual product change)
5. ✅ The Lab: five experiments, runnable, with verdicts computed from the run (3.1)
6. ✅ Baseline pinning and per-span deltas (3.2)

**Phase 3 — the tool box**
7. ✅ `simulate_tool` — every topology lesson, no credentials (A)
8. ✅ `summarize` — finding 16 reproducible for free (B)
9. ◻ Demote `web_browse` to a documented experiment; decide on Azure AI Search and the three
   extra search backends (2)

### On item 9, and why it is still open

The documentation demotion is done; the deletions are not, and they should not be made by
whoever happens to be editing. `RETRIEVER=search` (Azure AI Search) and the `tavily` /
`brave` / `serper` backends are working code that someone may be running. The argument for
removing them is that they cost configuration surface and teach nothing a cheaper option does
not already teach — which is a judgement about who this repository is for, not a defect.

Deleting a working integration to reduce a variable count is the kind of change that should
be argued in a pull request, not slipped into a cleanup.

**Not planned, and worth saying why:** more model providers, more vector stores, more
frameworks. Each one adds configuration surface and teaches nothing new — the lesson is that
topology dominates, and a second vector store is not a second topology.

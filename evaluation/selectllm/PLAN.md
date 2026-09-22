# Choosing the answer-generation model by measurement

`CHAT_PROVIDER=anthropic` + `claude-opus-5` + `effort=low` was never measured
against an alternative. `core/llm/anthropic.py` says so in a comment — "Raise
this if answer quality turns out to need it — measure with
`python -m evaluation.run_eval` rather than guessing." Nobody measured.

This experiment measures five arms on quality, latency, tokens and cost, and
pre-registers the rule that picks the winner.

**Status: plan agreed, not yet run. Nothing below is a result.**

## Arms — 5

Anthropic only, by decision. Fable 5.1 is excluded as a *candidate* so it can
judge without grading itself.

| arm id | model | effort | in $/MTok | out $/MTok |
| --- | --- | --- | --- | --- |
| `opus5-low` | `claude-opus-5` | low | $5 | $25 |
| `opus5-med` | `claude-opus-5` | medium | $5 | $25 |
| `sonnet5-low` | `claude-sonnet-5` | low | $2 | $10 |
| `sonnet5-med` | `claude-sonnet-5` | medium | $2 | $10 |
| `haiku45` | `claude-haiku-4-5` | — | $1 | $5 |

`opus5-low` is the incumbent and the baseline every arm is read against.

**Haiku 4.5 takes no `effort`** — a hard 400. `supports_effort()` already drops
it by matching the Claude-5 family prefix, so the arm needs no special case, but
Haiku cannot be swept on effort and its single row is not comparable to a
low/medium pair.

## Held fixed

Everything except the generator. `evaluation/live.py:pipeline()` exists for
this and its docstring says so: index, embedder, retriever, prompt and floor are
shared, so a difference between arms is a difference between models.

- **`top_k = 11` per namespace** (22 chunks/request) — the pending `exp-topk`
  recommendation, not the live `DEFAULT_TOP_K = 8`, so cost numbers stay valid
  after it merges. **This measures a config that is not live yet**; if
  `exp-topk` is abandoned, every number here needs rerunning at k=8.
- `MIN_RETRIEVAL_SCORE` at 0.35, which CLAUDE.md establishes is inert.
- `SYSTEM_INSTRUCTION` byte-identical across arms.

## The blended path, measured for the first time

`run_eval` and the `exp-topk` runner both score one namespace per case, so — as
CLAUDE.md flags — nothing measures the blended `/chat` path. Generation *is*
blended: the model gets `people_k + courses_k` chunks in one prompt and must
choose between corpora.

Cases here carry **no `namespace` field**. Every case runs the real blended path
at 22 chunks. That is the one structural difference from
`evaluation/topk/questions.jsonl`.

## Judging: blind ranking against ground truth

Fable 5.1 ranks the five answers to each question **blind**, by fidelity to
ground truth. One call per (case, ordering). Not claim-by-claim faithfulness —
that was a proxy for "which answer is better", and this measures it directly.

Why ranking rather than absolute scores: forced choice discriminates where
absolute rubric scores compress. CLAUDE.md already has the evidence —
`context_utilization` read **0.00 on every list-style answer**, this product's
main question shape, which is why it was dropped.

### Two kinds of ground truth, because one does not exist for broad questions

| stratum | n | ground truth | judge sees | $/call |
| --- | --- | --- | --- | --- |
| `grounded` | 12 | prose reference, complete | ref + 5 answers | $0.039 |
| `blended` | 8 | prose reference, complete | ref + 5 answers | $0.039 |
| `trap` | 4 | prose reference ("declines, because…") | ref + 5 answers | $0.039 |
| `list` | 8 | **acceptance criteria**, not an answer | criteria + 22 chunks + 5 answers | $0.091 |
| `noans` | 8 | "declines" — deterministic | nothing, measured | $0 |

**A complete reference is the oracle, so those calls carry no chunks** — which
is what makes them 2.3x cheaper than a grounded ranking call.

**`list` cases cannot have a prose reference, and this is settled, not an open
question.** CLAUDE.md: a reference naming 3 of 30 valid people scores the other
27 as precision errors. `exp-topk` then measured it — `people-broad-hci`
retrieved 12 genuinely relevant HCI people while `expected_slugs` named a
different subset of 7, so the set is *incomplete*, not wrong. Its finding #3 is
that recall is untrustworthy on broad questions for exactly this reason.

So a `list` case ships **acceptance criteria** instead: "names >= 5 people; every
person named must appear in the retrieved chunks; no duplicates; nobody whose
profile lacks HCI content." Checkable and complete, where an exhaustive list of
46 people is neither. These are the only calls that still need the chunks —
verifying a named person exists requires seeing what was retrievable.

### Three guardrails

1. **Order is randomized and counterbalanced** — every case is ranked twice with
   the arms in reversed order. LLM judges favour earlier options; without this
   the experiment measures position, not quality. Disagreement between the two
   orderings is reported, not averaged away.
2. **An absolute `acceptable: yes/no` per answer**, in the same call at no extra
   cost. Ranking alone names the least bad of five bad answers and the decision
   rule would fire regardless.
3. **Arms are anonymised** as A–E in the prompt, remapped after. The judge must
   not know which answer is Opus.

### No framework

`client.messages.parse()` with a schema, `shared/retry.py:with_backoff`, and a
disk cache keyed on (rubric version, model, effort, inputs). That is the whole
judge — `topk/judge.py` is 266 lines and made **0 errors over 900 pairs**.

DeepEval is not used, not even as a cross-check: a 1-in-35 unparseable-JSON
oracle cannot validate a 0-in-900 one, its numbers would not be comparable
(different judge, rubric, metric set — CLAUDE.md records this for the Ragas port),
and it is 1,341 lines plus 1,004 of tests pulling opentelemetry and posthog.
Structured outputs removed its reason to exist: the schema is enforced
server-side, so there is no JSON left to fail to parse.

**Judge validation is a prerequisite.** 15 cases ranked by hand, scored against
the judge with Cohen's kappa, before the grid runs. If kappa is poor the rubric
is rewritten before any arm is measured. `exp-topk` listed this as an unmet
limit; here it gates the run.

## Dataset — `cases.jsonl`, 40 cases

Expectations and ground truth harvested **offline from the raw GCS corpus**,
never from retriever or model output — harvesting from model output grades the
models on their own answers. Stratum is read from the case id (`{stratum}-{slug}`).

Fields: `id`, `question`, `reference` **or** `criteria`, `expected_slugs`
(kept as a cheap deterministic sanity check, explicitly marked incomplete on
`list` cases), `notes`.

- **`blended` is new** — "Who teaches Compilers, and what does that person
  research?" cannot be answered from one namespace. A weaker model is expected
  to answer half the question or cite across corpora incorrectly.
- **`trap` is new and is the highest-signal stratum for model choice.** A false
  premise ("which Khoury professor came from MIT to work on quantum
  networking?") invites fabrication: stronger models decline the premise, weaker
  ones synthesise a plausible person from nearby chunks. Only 4 cases because
  each needs hand-verification that the premise really is false.
- `noans` is 8 rather than `golden.jsonl`'s 7 — the existing 7 plus one
  in-domain-but-absent question, harder than a fully out-of-scope one.
- **Both colleges.** The 783 CoS profiles added 2026-09-21 have no golden
  coverage; `exp-topk` added 6 CoS questions. Drawing only from Khoury would
  measure half the corpus.

## Metrics — 5 groups

1. **Rank (judged)** — mean rank 1–5 per arm, plus order-disagreement rate.
2. **Acceptable-rate (judged)** — fraction of answers passing the absolute gate.
3. **Refusal correctness (measured)** — the 8 `noans` cases via
   `CaseOutcome.declined`, which matches the prompt's phrasing rather than the
   exact `NO_ANSWER` string; CLAUDE.md records that exact equality scored
   caveated refusals 0/7 instead of 7/7.
4. **Latency (measured)** — `RAGResult.timings_ms["generate"]`, p50 and p95,
   against `REQUEST_BUDGET_SECONDS = 45`.
5. **Tokens and cost (measured, then derived)** — input, output, and the
   **thinking-inflation ratio** (output ÷ visible answer tokens). Thinking bills
   as output under every `display` setting, so an arm can cost 3x what its
   answer length suggests. No existing harness records this.

Recorded per call because each silently corrupts a mean: `stop_reason`
(`max_tokens` means a truncated answer scored as a bad one), and refusals, which
arrive as HTTP 200 with `stop_reason == "refusal"`. `generate()` maps those to
`""`, which the pipeline turns into the no-answer string — **so a safety refusal
and an honest "I don't have that" are indistinguishable today**, which inflates
the `noans` score. The experiment must separate them.

## Decision rule — pre-registered, before any data exists

> The **cheapest** arm whose **mean rank is within 0.5** of the best arm, whose
> **acceptable-rate is >= 0.95**, whose **refusal score is 8/8**, and whose **p95
> generate latency is under 20s**. Ties break toward lower cost.

If no arm clears the refusal gate, the incumbent stays and the result is "no arm
is safe enough to switch to" — a result, not a failure. `exp-topk`'s rule
declined to fire and that was reported as the finding.

## Prerequisite code change: the generator must report usage

`Generator.generate()` returns `str`, and `core/llm/anthropic.py` has
`response.usage` in hand and discards it. Groups 4 and 5 are unmeasurable until
that changes.

Widen the ABC to return `Generation(text, input_tokens, output_tokens,
stop_reason)`. `topk/judge.py` set the precedent — its `Verdict` already carries
`input_tokens`/`output_tokens`.

**Not** an instance attribute such as `last_usage`. `serving/api/app.py` builds
one generator in `lifespan` and dispatches `pipeline.answer` through
`run_in_threadpool` at Cloud Run concurrency 80, so a per-instance metric is
read by the wrong request. CLAUDE.md documents exactly this failure for DeepEval
metrics holding results on the instance.

`live.py:pipeline()` also needs an `effort` passthrough; `build_generator`
already forwards `**kwargs`, so that is one parameter, not a refactor.

## Cost — input measured, output still projected

40 cases x 5 arms x 2 repeats = 400 generations. Repeats exist because Claude 5
rejects `temperature` entirely, so sampling noise cannot be turned off.

**Input is measured: 5,272 tokens** per blended request at k=11 — mean over 8
real cases via `messages.count_tokens` (free), median 4,918, range 3,955–7,426.
Rendered through the real `PromptBuilder` over actual chunk text from the
`exp-topk` dumps, with the real `SYSTEM_INSTRUCTION`.

**`exp-topk`'s README assumed ~7,700 and is overstated 1.5x.** Its "$0.04 per
query on Opus 5" should read ~$0.026 input. Worth correcting there.

| component | cost |
| --- | --- |
| generation, 400 calls | $13.07 |
| ranking, 128 calls over 2 repeats | $6.62 |
| kappa validation, 30 calls | $1.16 |
| embedding / Pinecone / `count_tokens` | $0 — free tier |
| **end to end** | **$20.86** |

Range on the output/thinking unknown: **$18–27**. That band is the very thing
the experiment measures, and it changes no decision.

Two properties of this shape:

- **Generation is now the majority of the bill** ($13.07 of $20.86), inverted
  from the claim-level design where judging was 60%. 128 ranking calls replaced
  2,000 claim calls.
- **Haiku's entire arm is $0.59.** Cost discipline belongs in the judge and the
  repeat count, never in the candidate set.

**Generations must be dumped to `samples.jsonl`** as `exp-topk` does, so a
rubric change re-ranks from the dump and never re-pays for generation. Without
that, every judge iteration costs another $13.

`--dry-run` gates the spend: projected cost per arm from `count_tokens`, zero
generation calls, approved before anything is spent. Output tokens stay
projected until the first arm runs; a ~$0.50 probe of 3 generations per arm
would pin them.

## Known limits, stated up front

- **n = 40 across 5 strata** is 4–12 cases per stratum. Paired comparison across
  arms (same cases, same chunks, same prompt) is what makes it readable at this
  n; absolute levels are soft, and `trap` at n=4 can only show a large effect.
- **Ranking is ordinal and relative.** It cannot say how good the best arm is,
  only that it is best — which is what `acceptable-rate` is for. Ranks also do
  not transfer to a later run that adds a sixth arm.
- **Judge validation gates the run** (see above), so this is a prerequisite
  rather than a limit — unlike `exp-topk`.
- **Retrieval is held at k=11, which is not live.**
- **Two arms cannot be swept on effort** — Haiku 4.5 rejects the parameter.
- **`list` ground truth is criteria, not an answer**, so those 8 cases measure
  compliance with a rubric rather than agreement with a reference. That is a
  weaker claim than the other 24 cases support, and is reported separately
  rather than pooled into one mean.

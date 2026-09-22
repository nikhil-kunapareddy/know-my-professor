# Choosing the answer-generation model by measurement

`CHAT_PROVIDER=anthropic` + `claude-opus-5` + `effort=low` was never measured
against an alternative. `core/llm/anthropic.py` even says so in a comment:
"Raise this if answer quality turns out to need it — measure with
`python -m evaluation.run_eval` rather than guessing." Nobody measured.

This experiment measures it across five arms on quality, latency, tokens and
cost, and pre-registers the rule that picks the winner.

**Status: plan agreed, not yet run. Nothing below is a result.**

## Arms — 5

Anthropic only, by decision. Fable 5.1 is excluded as a *candidate* so it can
serve as the judge without grading itself.

| arm id | model | effort | in $/MTok | out $/MTok |
| --- | --- | --- | --- | --- |
| `opus5-low` | `claude-opus-5` | low | $5 | $25 |
| `opus5-med` | `claude-opus-5` | medium | $5 | $25 |
| `sonnet5-low` | `claude-sonnet-5` | low | $2 | $10 |
| `sonnet5-med` | `claude-sonnet-5` | medium | $2 | $10 |
| `haiku45` | `claude-haiku-4-5` | — | $1 | $5 |

`opus5-low` is the incumbent, so it is the baseline every other arm is read
against.

**Haiku 4.5 takes no `effort`.** It is a hard 400 ("This model does not support
the effort parameter"). `supports_effort()` in `core/llm/anthropic.py` already
drops it by matching on the Claude-5 family prefix, so the arm needs no special
case — but it also means Haiku cannot be swept on effort, and its single row is
not directly comparable to a low/medium pair.

## Held fixed

Everything except the generator. `evaluation/live.py:pipeline()` already exists
for this and its docstring says so: index, embedder, retriever, prompt and floor
are shared, so a difference between arms is a difference between models.

- **`top_k = 11` per namespace**, i.e. 22 chunks per request — the pending
  `exp-topk` recommendation, not the current `DEFAULT_TOP_K = 8`. Chosen so the
  cost numbers stay valid after `exp-topk` merges. **This experiment therefore
  measures a configuration that is not live yet**; if `exp-topk` is abandoned,
  every cost number here needs rerunning at k=8.
- `MIN_RETRIEVAL_SCORE` left at 0.35, which CLAUDE.md establishes is inert.
- `SYSTEM_INSTRUCTION` byte-identical across arms.

## The blended path, measured for the first time

Every existing harness scores one namespace per case (`run_eval` and the
`exp-topk` runner both do). CLAUDE.md flags the consequence: nothing measures
the blended `/chat` path at all. Generation *is* blended — the model receives
`people_k + courses_k` chunks in one prompt and has to choose between corpora.

So cases here carry **no `namespace` field**. Every case runs the real blended
path at 22 chunks. This is the one structural difference from
`evaluation/topk/questions.jsonl`.

## Dataset — `cases.jsonl`, target 40 cases

Built the same way as `topk/questions.jsonl` and `golden.jsonl`: expectations
harvested **offline from the raw GCS corpus**, never from retriever or model
output. Harvesting from model output would grade the models on their own
answers.

Five strata, because what separates a weak generator from a strong one is not
the same as what separates retrieval settings. Stratum is read from the case id
(`{stratum}-{slug}`).

| stratum | n | `reference`? | what it isolates |
| --- | --- | --- | --- |
| `grounded` | 12 | yes | ordinary correctness on one verifiable entity |
| `list` | 8 | **no** | faithfulness when many answers are valid |
| `noans` | 8 | n/a | refusal discipline on out-of-scope questions |
| `blended` | 8 | yes | synthesis across both corpora in one answer |
| `trap` | 4 | yes | fabrication resistance under a false premise |

- **`list` cases carry no `reference` on purpose.** CLAUDE.md already records
  why for `golden.jsonl`: a reference naming 3 of 30 valid people scores the
  other 27 as errors. `exp-topk` then quantified it — 137 valid courses for one
  ML question, 46 valid people for one neuroscience question. These cases are
  scored on faithfulness (is every claim supported by a retrieved chunk?), never
  on correctness.
- **`blended` is new.** "Who teaches Compilers, and what does that person
  research?" cannot be answered from one namespace. A weaker model is expected
  to answer half the question or cite across corpora incorrectly.
- **`trap` is new and is the highest-signal stratum for model choice.** A false
  premise ("which Khoury professor came from MIT to work on quantum
  networking?") invites fabrication. Stronger models decline the premise; weaker
  ones synthesise a plausible person from nearby chunks. Only 4 cases because
  each needs hand-verification that the premise really is false.
- `noans` is 8 rather than the 7 in `golden.jsonl`: the existing 7 plus one
  in-domain-but-absent question, which is harder than a fully out-of-scope one.

## Metrics — 5 groups

Measured, not judged, wherever possible.

1. **Quality (judged)** — faithfulness against the 22 retrieved chunks;
   correctness against `reference` where one exists.
2. **Refusal correctness (measured)** — the 8 `noans` cases. Scored with
   `CaseOutcome.declined`, which matches the prompt's phrasing rather than the
   exact `NO_ANSWER` string — CLAUDE.md records that exact equality scored
   caveated refusals 0/7 instead of 7/7.
3. **Latency (measured)** — `RAGResult.timings_ms["generate"]`, p50 and p95,
   against `REQUEST_BUDGET_SECONDS = 45`.
4. **Tokens (measured)** — input, output, and the **thinking-inflation ratio**
   (output tokens ÷ visible answer tokens). Thinking tokens bill as output under
   every `display` setting, so an arm can cost 3x what its answer length
   suggests. This ratio is the main thing the experiment exposes and no existing
   harness records it.
5. **Cost (derived)** — $/query and $/1k queries from measured tokens × the
   table above. Not estimated from chunk counts.

Also recorded per call, because each silently corrupts a mean: `stop_reason`
(`max_tokens` means a truncated answer scored as a bad one), and refusals, which
arrive as HTTP 200 with `stop_reason == "refusal"` — `generate()` currently maps
those to `""`, which the pipeline turns into the no-answer string. **A safety
refusal and an honest "I don't have that" are therefore indistinguishable
today**, and on the `noans` strata that inflates the refusal score. The
experiment must separate them.

## Decision rule — pre-registered, written before any data exists

> The **cheapest** arm whose faithfulness is within **0.05** of the best arm,
> whose refusal score is **8/8**, and whose **p95** generate latency is under
> **20s**. Ties break toward lower cost. Correctness is a tiebreaker within
> that set, not a gate, because it is unmeasurable on `list` cases.

If no arm clears the refusal gate, the incumbent stays and the result is "no
arm is safe enough to switch to" — a result, not a failure. `exp-topk`'s rule
declined to fire and that was reported as the finding.

## Prerequisite code change: the generator must report usage

`Generator.generate()` returns `str`, and `core/llm/anthropic.py` has
`response.usage` in hand and discards it. Groups 4 and 5 above are unmeasurable
until that changes.

Widen the ABC to return a `Generation(text, input_tokens, output_tokens,
stop_reason)`. `exp-topk/judge.py` set the precedent — it already captures
`input_tokens`/`output_tokens` on its `Verdict`.

**Not** an instance attribute such as `last_usage`. `serving/api/app.py` builds
one generator in `lifespan` and dispatches `pipeline.answer` through
`run_in_threadpool` at Cloud Run concurrency 80, so a per-instance metric is
read by the wrong request. CLAUDE.md documents exactly this failure for DeepEval
metrics holding results on the instance.

`live.py:pipeline()` also needs an `effort` passthrough; `build_generator`
already forwards `**kwargs`, so the change is one parameter, not a refactor.

## Judging: Fable 5.1, but not through DeepEval

`claude-fable-5-1` is the judge — it is the more capable model and it is not a
candidate, so nothing grades itself.

**The cost driver is judge call count, not judge price.** DeepEval fires ~21
judge calls per (case, metric-set), each carrying the full context. At 40 cases
× 5 arms × 21 calls ≈ 4,200 calls of ~8K input, Fable's $10/$50 rates put the
judged portion near **$380**. That is not a Fable problem; it is a
calls-per-case problem.

`exp-topk` measured the affordable shape: **one small pointwise call per unit**,
`effort: low`, ~50 output tokens — 900 pairs for **$9.62**, $0.0107 each, zero
judge errors. Same judge model.

So the harness follows `exp-topk`, not DeepEval: decompose each answer into
claims, verify each claim against the chunk it cites, one small call per claim.
Reuse `judge.py`'s disk cache keyed on (rubric version, model, effort, inputs),
so a re-score is free and a rubric edit invalidates correctly.

**No framework, and DeepEval is not used at all here** — not even as a
cross-check. Three reasons, in order of weight:

- A 1-in-35 oracle cannot validate a 0-in-900 one. DeepEval's cheap judge
  returned unparseable JSON on 1 pair in 35 on the 2026-09-20 live run; the
  pointwise judge made zero errors over 900 pairs.
- Its numbers would not be comparable anyway — different judge, rubric and
  metric set. CLAUDE.md already records this for the Ragas -> DeepEval port,
  where stored runs became non-comparable for exactly this reason.
- It is 1,341 lines of wiring plus 1,004 lines of tests, against 266 for
  `topk/judge.py`, and pulls opentelemetry + posthog (which is why it is kept
  out of requirements.txt).

What replaces the cross-check is a **hand-labelled sample**: ~30 claims labelled
by hand, scored against the judge with Cohen's kappa, before the grid runs. That
validates the judge against the only ground truth that counts, and it also
settles `effort: low` vs `medium` for the judge itself. `exp-topk` listed this
as an unmet limit; here it is a prerequisite.

The judge needs no abstraction beyond what the repo already has: a rubric
string, `client.messages.parse()` with a schema, and `shared/retry.py`'s
`with_backoff`. Structured outputs are what remove the framework's reason to
exist — `output_format` is enforced server-side, so a validated object comes
back and there is no JSON to fail to parse.

## Projected cost — to be replaced by a measured `--dry-run`

40 cases × 5 arms × 2 repeats = 400 generations. Repeats exist because Claude 5
rejects `temperature` entirely, so sampling noise cannot be turned off; Llama's
`temperature=0.0` has no equivalent here.

| component | estimate |
| --- | --- |
| generation, 400 calls @ 22 chunks | ~$14 |
| judging, pointwise Fable, ~1.2k claims | ~$15 |
| **total** | **~$30** |

Input is ~7.8K tokens per generation (22 chunks ≈ 7.7K, system 116). **Prompt
caching does not help**: `SYSTEM_INSTRUCTION` is 465 chars ≈ 116 tokens against
a 512–4096 minimum cacheable prefix, so the stable prefix silently will not
cache, and the retrieved chunks — 95%+ of input — change every query.

`--dry-run` must print the projected number per arm from `count_tokens` with
zero generation calls, and that number gets approved before anything is spent.
This table is arithmetic, not measurement.

## Known limits, stated up front

- **n = 40 across 5 strata** is 4–12 cases per stratum. Paired comparison across
  arms (same cases, same chunks) is what makes it readable at this n; absolute
  levels are soft, and `trap` at n=4 can only show a large effect.
- **Judge validation is a prerequisite, not a limit.** Unlike `exp-topk`, this
  run does not start until the ~30-claim hand-labelled sample is scored. If
  kappa is poor, the rubric is rewritten before any arm is measured.
- **Retrieval is held at k=11, which is not live.** See above.
- **Two arms cannot be swept on effort** — Haiku 4.5 rejects the parameter.
- **CoS coverage.** 783 College of Science profiles were added 2026-09-21 with
  no golden coverage; `exp-topk` added 6 CoS people questions. This set must
  draw from both colleges or it measures Khoury only.

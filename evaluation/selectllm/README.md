# Choosing the answer-generation model by measurement

`claude-opus-5` at `effort=low` was the default because it was the default.
This measured it against three alternatives on the blended `/chat` path.

**Headline: the pre-registered rule declined to fire, and the reason is a
finding about the product rather than about the models.** No arm reached the
0.95 acceptable-rate gate — every arm sat at 0.77–0.81, so roughly **one answer
in five is judged not good enough regardless of which model writes it**. That
is a retrieval and prompt problem, and no model choice fixes it.

`PLAN.md` is the pre-registered design. It was written, and the decision rule
fixed, before any data existed.

## Arms — 4

| arm | model | effort | in $/MTok | out $/MTok |
| --- | --- | --- | --- | --- |
| `opus5-low` | `claude-opus-5` | low | $5 | $25 |
| `opus5-med` | `claude-opus-5` | medium | $5 | $25 |
| `sonnet5-med` | `claude-sonnet-5` | medium | $2 | $10 |
| `haiku45` | `claude-haiku-4-5` | — (rejects it) | $1 | $5 |

Judge: `claude-fable-5-1`, deliberately not an arm so it cannot rank itself.

## Results — 50 cases x 4 arms x 2 repeats = 400 answers, 180 rankings, $16.41

| arm | mean rank | 95% CI | accept | refusals | p50 | p95 | in tok | out tok | $/1k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `opus5-low` | **1.43** | [1.34, 1.53] | 0.81 | 10/10 | 3.4s | 6.4s | 4,241 | 182 | $25.74 |
| `opus5-med` | 1.49 | [1.40, 1.59] | 0.79 | **8/10** | 3.8s | 7.5s | 4,241 | 217 | $26.64 |
| `haiku45` | 1.66 | [1.54, 1.78] | 0.77 | 10/10 | **1.9s** | **3.3s** | **2,854** | **122** | **$3.46** |
| `sonnet5-med` | 1.67 | [1.55, 1.78] | 0.78 | 10/10 | 2.1s | 4.3s | 4,241 | 128 | $9.76 |

CIs are 10,000-sample bootstraps over 180 rank observations per arm. They split
the arms into two groups that do not overlap:

**{`opus5-low`, `opus5-med`} > {`haiku45`, `sonnet5-med`}**

Within each group the difference is noise. In particular **medium effort bought
nothing**: `opus5-med` is not better than `opus5-low` on rank, is worse on
refusals, costs 4% more and is 17% slower. The `DEFAULT_EFFORT = "low"` comment
in `core/llm/anthropic.py` guessed right.

## Mean rank by stratum — the most useful table here

| arm | lookup (72) | broad (48) | blended (40) | trap (20) |
| --- | --- | --- | --- | --- |
| `opus5-low` | **1.47** | **1.35** | 1.43 | 1.50 |
| `opus5-med` | 1.53 | 1.44 | 1.40 | 1.70 |
| `sonnet5-med` | 1.85 | 1.69 | 1.35 | 1.60 |
| `haiku45` | 1.89 | 1.81 | **1.32** | **1.10** |

The aggregate ranking hides an inversion. **Haiku is the best arm at refusing a
false premise (1.10) and at the blended course+person question (1.32), and the
worst at everything else.** Opus wins where coverage and detail decide the
answer — broad "who works on X" and one-entity lookups.

The reading: Opus's advantage is *elaboration*, and elaboration is a liability
on a trap question. A `trap` case rewards saying less.

## What the numbers do NOT support

**Per-case rankings are noisy, and this bounds every claim above.** Measured
three ways:

| comparison | pairwise agreement |
| --- | --- |
| Fable vs itself, arm order reversed | 0.783 (kappa 0.65) |
| Fable vs itself, repeat 1 vs repeat 2 | 0.622 |
| Claude Opus 5 vs Fable, blind | 0.511 (kappa 0.32) |

The judge disagrees with **itself** on 22% of pairwise preferences when the
candidates are merely reordered, and on 38% when shown a second sample from the
same arm. So 0.51 agreement from an independent ranker is close to the
achievable ceiling, not evidence of a broken rubric — and the rubric is not
ranking by position, or reversal would flip far more.

This is why the conclusion is stated as two groups rather than a strict order of
four. A gap of 0.06 mean rank (`opus5-low` vs `opus5-med`) is well inside the
noise; a gap of 0.23 (`opus5-low` vs `haiku45`) is not.

Arm answers are genuinely different (median pairwise text similarity 0.35), so
the noise is disagreement about quality, not indistinguishable candidates.

## Findings that were not the question asked

1. **Haiku 4.5 tokenises this prompt 1.48x more efficiently than Opus 5.**
   Same 22 chunks, byte-identical prompt, verified per case: 2,854 tokens
   against 4,241. A different tokenizer, not a different context. Its cost
   advantage therefore compounds — cheaper per token *and* fewer tokens — and
   it is **7.4x cheaper per 1,000 queries**, not the 5x the price list implies.

2. **`opus5-med` failed 2 of 10 refusal cases.** It is the only arm that
   answered a question the corpus cannot support. Higher effort made the model
   more willing to construct an answer, which is the opposite of the intended
   effect and the single clearest quality difference the experiment found.

3. **Thinking is cheap at these effort levels.** Billed output runs ~1.5-1.7x
   the visible answer on the Claude 5 arms (182 billed / ~112 visible at low,
   217 / ~129 at medium). The feared silent 3x thinking tax did not appear on
   this task. Haiku, which does no thinking, writes the **longest** answers
   (528 chars) from the **fewest** billed tokens (122).

4. **`evaluation/live.py` was searching an empty namespace.** It never passed
   `namespaces` to `RAGPipeline`, so it fell back to the unnamed default
   partition, empty since the people vectors moved to `people`. Latent because
   `run_eval` always passes an explicit `namespace=`; the first caller of the
   blended path retrieved 0 chunks and nothing raised. The free dry-run gate
   caught it before any money was spent. Fixed, and pinned by a test.

5. **`golden.jsonl`'s `no-answer-organic-chemistry` is now answerable.** 14
   College of Science chemists match it; the CoS rollout silently converted a
   refusal case into an answerable one. The other six refusal cases survive.

## Recommendation — a judgment, not the rule's output

The rule declined to fire, so this is an argument from the tables and is
labelled as such.

**Keep `opus5-low`.** It has the best mean rank, ties for the best refusal
record, and its p95 of 6.4s sits comfortably inside the 45s request budget.
Nothing here justifies a change, and `opus5-med` is strictly worse.

**But the interesting option is `haiku45` at 7.4x less cost**, and the honest
statement is that this experiment cannot approve it: it is significantly worse
on the two strata that make up 120 of the 180 rankings. If cost ever becomes
binding, the route is Haiku for course/schedule questions and Opus for people
questions — which the stratum table supports and which no single-model choice
can capture.

**The accept-rate result outranks the model choice.** One answer in five is
judged unacceptable on every arm. Work on retrieval and the prompt has more
headroom than any swap between these four.

## Method, in brief

- 50 cases harvested offline from the raw GCS corpus, never from retriever or
  model output. 18 lookup / 12 broad / 10 blended / 5 trap / 5 noans, spanning
  Khoury and the College of Science.
- Ground truth authored by Fable 5.1 from harvested `source_facts`, because the
  code's author is Opus 5 and two arms are Opus 5. Its filtering pass dropped
  15 of 31 robotics keyword matches and 9 of 22 accessibility matches as false
  positives.
- Generation at `top_k=11` per namespace (22 chunks), blended across both
  namespaces — the path `/chat` serves and no prior harness measured.
- Ranking is blind (arms shown as A-D), counterbalanced (every case ranked
  twice with the order reversed), ties allowed, with an absolute
  `acceptable: yes/no` alongside the rank.
- Refusals scored deterministically, never judged.
- No evaluation framework: `client.messages.parse()` with a schema, the repo's
  own `with_backoff`, and a disk cache. 0 judge errors in 225 calls.

## Reproducing

```bash
python -m evaluation.selectllm dry-run    # free; gates the spend
python -m evaluation.selectllm generate   # 400 answers -> samples.jsonl
python -m evaluation.selectllm rank       # 180 rankings -> ranks.jsonl
python -m evaluation.selectllm report     # applies the pre-registered rule
```

`rank` re-reads `samples.jsonl`, so a rubric change re-judges for ~$8 and never
re-pays the $6.56 for generation. Both commands are resumable.

## Known limits

- **Human validation is still open.** The 0.511 figure is Opus-vs-Fable, which
  cannot detect a bias the two share. A 15-case human ranking remains the one
  check that would settle whether the judge tracks what a user wants.
- **n = 50**, so `trap` (5 cases) and `noans` (5) can only show large effects.
  The refusal difference on `opus5-med` is 2 cases out of 10.
- **Broad ground truth is a filtered keyword harvest**, not a verified census.
  It scores all four arms identically, so it cancels in a ranking, but the
  absolute coverage numbers are soft.
- **`top_k=11` is not live** — `DEFAULT_TOP_K` is still 8. Cost figures assume
  the `exp-topk` recommendation ships.
- **One term of Banner data.** Every blended case resolves to Fall 2026.

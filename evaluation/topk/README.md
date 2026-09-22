# Choosing `top_k` by measurement

`DEFAULT_TOP_K = 8` was never measured. This experiment measured it, per
namespace, with an LLM-as-judge over the retrieved chunks.

**Headline: there is no relevance-based optimum.** For broad questions the count
of genuinely useful chunks is still climbing at k=25, at 76–89% precision. The
knee the experiment went looking for does not exist, so k is a cost/quality
decision and the tables below say what each k buys. This is the same shape as
the `MIN_RETRIEVAL_SCORE` finding in CLAUDE.md: the intuitive knob turns out not
to be the one that decides anything.

## Running it

```bash
python -m evaluation.topk --namespace people --dry-run     # retrieve + cost estimate, no judge
python -m evaluation.topk --namespace people               # the real run
python -m evaluation.topk --namespace courses
python -m evaluation.topk --namespace people --stratum broad --ks 3,5,7,9,11,15,20,25
python -m evaluation.topk --namespace people --from-dump   # re-score, no API calls, free
```

Env: `PINECONE_API_KEY` + the embedding key for retrieval, `ANTHROPIC_API_KEY`
for the judge. Verdicts cache in `.topk_cache/`; recorded runs land in
`evaluation/results/2026-09-21-topk/` and replay with `--from-dump`.

## Method

- **One namespace per question.** The knob is per-namespace — `/chat` gives each
  its own `top_k` — so mixing corpora in one query would measure something the
  product never does.
- **Retrieve once at `--k-max`, judge once, slice prefixes.** Retrieval is
  ranked and deterministic, so the chunks at k=3 are the first three at k=11.
  Five k values cost one retrieval and one judging pass.
- **Pointwise binary judging** with `claude-fable-5-1` at `effort: low`. One
  call per (question, chunk): judging all k at once is ~3× cheaper and makes a
  chunk's verdict depend on its neighbours, which is the variable under test.
- **The rubric is "usable", not "on-topic"** (see `judge.py`). A
  formal-verification syllabus is not a relevant answer to "who works on formal
  verification" — it names no person.
- **The decision rule was fixed before the data existed** (`report.py`):
  smallest k where a +2 step buys < 0.5 relevant chunks, provided recall has not
  been given up. It declined to fire, which is a result, not a failure.

## Question set — `questions.jsonl`, 50 cases

14 people + 36 courses, the 27/73 split of the namespaces' vector counts (4,105
and 10,948). Expectations harvested **offline from the raw GCS corpus** on
2026-09-21, never from retriever output.

|  | broad | narrow | no-answer |
| --- | --- | --- | --- |
| `people` (14) | 7 | 5 | 2 |
| `courses` (36) | 18 | 12 | 6 |

People = 6 Khoury (carried from `golden.jsonl`) + 6 College of Science (new —
CoS had **zero** golden coverage) + 2 no-answer. Courses span 20+ subjects, not
just CS.

**The broad/narrow split is the most important design choice here, and it was
learned the hard way.** A narrow question has exactly one correct entity, so its
relevant-chunk count pins at ~1 and is flat at every k *by construction*. A set
of only narrow questions answers "k=3" and measures nothing. The stratum is read
from the case id (`{namespace}-{broad|narrow|noans}-{slug}`).

## Results — 900 judged pairs, $9.62, 0 judge errors

### `people`

| k | relevant | marginal | prec | recall | MRR | broad prec | narrow prec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 2.58 | — | 86.1% | 91.7% | 0.764 | 100.0% | 66.7% |
| 5 | 4.25 | +1.67 | 85.0% | 91.7% | 0.764 | 100.0% | 64.0% |
| 7 | 5.67 | +1.42 | 81.0% | 91.7% | 0.764 | 95.9% | 60.0% |
| 9 | 7.00 | +1.33 | 77.8% | 91.7% | 0.764 | 95.2% | 53.3% |
| 11 | 8.50 | +1.50 | 77.3% | 100.0% | 0.774 | 96.1% | 50.9% |
| 15 | — | — | — | — | — | 92.4% | — |
| 20 | — | — | — | — | — | 89.3% | — |
| 25 | — | — | — | — | — | 89.1% | — |

Broad, extended: 3.00 → 5.00 → 6.71 → 8.57 → 10.57 → 13.86 → 17.86 → **22.29**
relevant chunks at k=3…25. Still gaining **+4.43** on the last step.

### `courses`

| k | relevant | marginal | prec | recall | MRR | broad prec | narrow prec |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 2.37 | — | 78.9% | 100.0% | 0.833 | 100.0% | 47.2% |
| 5 | 3.63 | +1.27 | 72.7% | 100.0% | 0.833 | 98.9% | 33.3% |
| 7 | 4.77 | +1.13 | 68.1% | 100.0% | 0.833 | 97.6% | 23.8% |
| 9 | 5.77 | +1.00 | 64.1% | 100.0% | 0.833 | 94.4% | 18.5% |
| 11 | 6.73 | +0.97 | 61.2% | 100.0% | 0.833 | 91.9% | 15.2% |
| 15 | — | — | — | — | — | 87.8% | — |
| 20 | — | — | — | — | — | 81.1% | — |
| 25 | — | — | — | — | — | 75.8% | — |

Broad, extended: 3.00 → … → 10.11 → 13.17 → 16.22 → **18.94**. Still gaining
**+2.72** on the last step.

## What the data actually says

1. **No knee exists for broad questions.** Relevance does not run out: "which
   courses cover machine learning?" has 137 valid courses, "who researches
   neuroscience?" has 46 valid people. The binding constraint is cost and
   latency, not relevance — so no threshold on relevance can pick k.

2. **Narrow questions are settled at k=3.** `courses/narrow` gains **+0.00**
   from k=5 onward while precision falls 47% → 15%. Everything past rank 3 is
   noise for a course lookup.

3. **`recall@k` is not trustworthy on broad questions.** `people-broad-hci`
   retrieves 12 HCI people that the judge scores relevant, but `expected_slugs`
   names a different subset of 7; the "recall jump" at k=11 is one coincidental
   overlap, not a retrieval improvement. CLAUDE.md already notes this for
   `reference`; it applies to `expected_slugs` on broad questions too. Recall is
   sound on narrow questions, where it is 100% at k=3 in both namespaces.

4. **A larger k does not endanger the no-answer path.** All 8 no-answer
   questions were judged **100% clean at every k up to 11** — not one plausible
   chunk crept in. Raising k costs precision, not safety.

5. **Cheap.** 900 judged pairs, $9.62, $0.011/pair, zero judge errors.
   `effort: low` produced ~50 output tokens per call.

## Recommendation, and it is a judgment call

The pre-registered rule declined to fire, so what follows is **not** what the
rule chose — it is an argument from the tables, labelled as such.

**k = 11 for both namespaces**, i.e. raise `DEFAULT_TOP_K` from 8 to 11 and
leave it shared. Reasoning:

- k=11 is the largest k at which **both** broad strata keep ≥90% precision
  (people 96.1%, courses 91.9%). At k=15 courses falls to 87.8%.
- Broad questions gain real, judged-useful context all the way there.
- Narrow questions are unharmed: the no-answer result shows the model is not
  misled by extra chunks, and narrow recall is saturated far below 11.
- Cost: 22 chunks per `/chat` request instead of 16 — roughly 7.7K input tokens,
  about $0.04 per query on Opus 5.

**`TOP_K_BY_NAMESPACE` is not needed.** Both namespaces land on the same number,
so the per-namespace config change this experiment was expected to justify is
unnecessary — one constant still does the job.

## Known limits

- **n is small**: 7 broad people questions, 18 broad course questions. The
  marginal-gain comparison is paired across k, which is what makes it readable
  at this n; the absolute levels are softer.
- **The judge is unvalidated against human labels.** A hand-labelled sample
  (~30 pairs, Cohen's κ) would fix that and would also settle `effort: low` vs
  `medium`. Not done.
- **The absolute relevance counts are rubric-dependent.** Differences between k
  are not — every k is scored over the same judged chunks, so a rubric bias is a
  constant that cancels.
- **Blended `/chat` is not measured here.** Every question is scored against one
  namespace. A request actually receives `people_k + courses_k` chunks.

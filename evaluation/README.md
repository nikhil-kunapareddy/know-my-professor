# evaluation/

Two harnesses over one golden set. They answer different questions, and the
second one costs money, so start with the first.

| | `run_eval.py` | `run_deepeval.py` |
| --- | --- | --- |
| Asks | was the right professor retrieved, and cited? | is the answer grounded, relevant, correct? |
| Scores with | set arithmetic over slugs | an LLM judge (DeepEval) |
| Cost | one embed + one search per case | ~21 judge calls per case |
| Needs | `PINECONE_API_KEY`, `MISTRAL_API_KEY` | + `ANTHROPIC_API_KEY`, the `eval` extra |

```bash
pip install -e ".[eval]"                                  # deepeval is not in requirements.txt

python -m evaluation.run_eval                             # recall@k / MRR
python -m evaluation.run_deepeval --stage retrieval       # judged, no generation
python -m evaluation.run_deepeval                         # all three stages
python -m evaluation.run_deepeval --dump evaluation/runs/$(date +%F).jsonl
python -m evaluation.run_deepeval --from-dump evaluation/runs/2026-09-17.jsonl
python -m evaluation.run_deepeval --min faithfulness=0.8 --min context_recall=0.7

# a stored experiment: a reproducible random sample, saved with its settings
python -m evaluation.run_deepeval --sample 10 --seed 7 \
  --dump evaluation/results/<dir>/samples.jsonl \
  --report-json evaluation/results/<dir>/report.json \
  --verbose | tee evaluation/results/<dir>/report.txt
```

Use `--sample N --seed S` rather than `--limit N`: the golden file is grouped by
what each block probes, so a prefix measures one block and never reaches the
no-answer cases at the end. Stored runs live in `evaluation/results/` (tracked
in git); see its README for the layout and for how to re-judge one for free.

## The three stages, and why they are separate

A single "quality" number tells you nothing about where to spend your time, so
metrics are grouped by the point in the pipeline they probe. Read them in this
order; the first stage that looks wrong is the one to fix.

**`retrieval`** — embed + search only, no generator, so it is the cheap one.
Diagnoses the index, the embedding model, `top_k`, and `MIN_RETRIEVAL_SCORE`.

- `context_relevance` — are the retrieved chunks about the question at all?
- `context_precision` — are the useful ones ranked above the useless ones?
- `context_recall` — does the context cover what the reference answer claims?

**`generation`** — runs the real pipeline and judges the answer against the
context it was actually given.

- `faithfulness` — does the answer assert anything the context does not support?
- `answer_relevancy` — does it address the question, or pad around it?

**`end_to_end`** — compares the answer to the golden reference.

- `answer_correctness` — does it agree with the reference on the facts? A
  `GEval` metric with evaluation steps written for this corpus; the steps are
  spelled out in `deepeval_eval/metrics.py` rather than generated per run, so
  the yardstick does not move between runs.
- `semantic_similarity` — is it saying the same thing? (no judge, embeddings
  only — the one number in the report that is not an LLM's opinion)

**Four metrics from the Ragas era are gone**, dropped rather than reinvented
when this moved to DeepEval: `context_entity_recall` and `noise_sensitivity`
(no equivalent), `response_groundedness` (faithfulness already covers it), and
`context_utilization` (measured 2026-09-17 as reading 0.00 on any list-style
answer, which is this product's main question shape). Numbers are therefore not
comparable to runs stored before the port.

Localising a regression is the point of the split: faithfulness down with
`context_recall` flat is the generator's fault, both down together is the
retriever's, and prompt work in the second case is wasted work.

## What a run actually costs (measured, 2026-09-20)

One real case (`computing-on-encrypted-data`, top_k=8) scored by all seven
metrics, counted by wrapping both clients:

```
metric                   judge  embed     secs
context_relevance            9      0      4.7
context_precision            2      0      6.6
context_recall               2      0      3.3
faithfulness                 4      0      5.1
answer_relevancy             3      0      4.1
answer_correctness           1      0      2.2
semantic_similarity          0      1      1.8
TOTAL per case              21      1     27.7
```

For scale, the same case under the previous Ragas harness cost **50 judge calls,
6 embeds and 134 seconds** across eleven metrics. Most of the saving is
structural rather than clever: `noise_sensitivity` alone was 19 judge calls and
62 seconds, and `context_utilization` another 8.

**Judge calls scale with `top_k`.** `context_relevance` makes one call per
retrieved chunk plus one, so halving `--top-k` roughly halves it. The others are
flat: `faithfulness` decomposes the answer into claims in one call and judges
them in a second, `answer_correctness` is a single G-Eval call, and
`semantic_similarity` never touches the judge at all.

Extrapolating the 60-case set across all three stages: **~1,260 judge requests,
~120 Mistral embedding requests** (60 similarity + 60 query embeds), **60 Opus
generations**, and roughly **7-10 minutes** wall clock at `--concurrency 4`.

Against the limits the providers themselves report:

| Provider | Reported limit | This run | Verdict |
| --- | --- | --- | --- |
| Anthropic (judge + generation) | 20,000 req/min, 10M in / 2M out tokens per min | ~1,320 requests total | not close |
| Pinecone serverless free | generous for reads | 60 queries | not close |
| Mistral free tier | 60 req/min (`x-ratelimit-limit-req-minute`) | ~120 requests | comfortable, but paced anyway |

`build_judge_embedder` still paces the embedder at
`EMBED_RATE_LIMIT_SLEEP_SECONDS` (1s), the free tier's own ceiling and the same
knob ingest uses. It is no longer the binding constraint the way it was under
Ragas — dropping from 6 embeds per case to 1 took ~420 requests down to ~120 —
but pacing 120 requests costs two minutes and removes a whole class of flake.

Two things worth knowing before you spend it:

- `--stage retrieval` is the cheap one: 13 of the 21 judge calls, no generator,
  no embeddings, and it answers "is the right context even reaching the model?"
- `--stage end_to_end` is the only stage that needs the embedder at all.

## Keeping it cheap

The zero-cost constraint applies here too, and a judged run is the one part of
this repo that spends real quota:

- `--stage retrieval` skips the generator entirely.
- Judge responses are cached on disk in `.deepeval_cache/` (gitignored), keyed
  on the model, the prompt and the schema — an unchanged re-run is nearly free,
  a changed pipeline re-judges. DeepEval has no cache on this path (its own is
  for `deepeval test run`), so `CachingJudge` in `judge.py` adds one; a hit
  reports a cost of 0.0, because it is one.
- `--dump` records the retrieved contexts and answers; `--from-dump` re-judges
  that recording without touching Pinecone or the chat model. Iterate on
  metrics, thresholds, and reference answers this way.
- A dump is re-joined to `golden.jsonl` by case id, so you can write reference
  answers **after** a run and score `end_to_end` on it for the price of judging.
- `--limit N` bounds everything.
- The judge defaults to a cheap model (`claude-haiku-4-5-*`). `--judge-model`
  overrides it, but absolute numbers shift between judges, so do not compare
  runs scored by different ones.
- **A failed judge call is not cached, so re-running retries only the
  failures.** Observed 2026-09-20: one metric-case pair in 35 died with
  `DeepEvalError: Evaluation LLM outputted an invalid JSON`, failed again on an
  immediate replay, and scored normally on a later attempt with the same cheap
  judge. It is intermittent rather than a property of the case. The run is not
  lost when it happens — the pair is recorded as an error, excluded from the
  mean, and the shrunken denominator is printed next to it (`n=3`, not `n=4`) —
  and `--from-dump` re-judges it for the price of that one call.

## The golden set

60 cases: 53 answerable, 7 no-answer, 105 distinct people, all 11 registered
section types covered (a test enforces that last one).

Expectations were derived from the **raw corpus** — 867 profiles and 252
weblinks records read out of GCS and keyword-searched offline — not from what
the retriever returns. That distinction is the whole ballgame: expectations
harvested from retrieval output guarantee high recall and measure nothing.
Every claim in every `reference` is traceable to text in the named person's
profile or weblinks record.

Three deliberate choices, explained in the file's own header:

- **Broad questions carry no reference.** For "who works on machine learning?"
  dozens of people are right, and a reference naming three would score every
  other valid professor as a context-precision error. Those cases get a wide
  `expected_slugs` list (any match is a hit) and are judged only by the metrics
  that need no reference.
- **`expect_no_answer` cases name no slugs.** Without them an evaluation can
  only reward retrieving more, so raising `MIN_RETRIEVAL_SCORE` always looks
  free. Each no-answer topic was checked to have zero support in the corpus.
- **A refusal is recognised by the prompt's phrasing, not by string identity.**
  Asked something out of scope, the model opens with the requested refusal and
  then adds an honest caveat about the nearest material it saw. That is the
  behaviour worth having; exact equality with `NO_ANSWER` scored it as failure.

## Reading the numbers honestly

- **Recall@k and MRR cover answerable cases only.** Averaging refusals into them
  would let a system that retrieves nothing score well on the cases that want
  nothing. Refusals get their own line (`Declined right`).
- **A no-answer case is decided by the generator, not by retrieval.** Judge them
  with `--generate`; a retrieval-only run says so rather than reporting 0%.
- **A missing input is a skip, not a zero.** A case with no `reference`, or one
  where the pipeline declined to answer, is excluded and counted in the skip
  list rather than averaged in as a failure.
- **Errors are reported, never fatal.** A judge failure on one case does not
  discard the rest of the run, so check the error block before believing a `--`.
- **`--min` gates fail on unmeasured metrics.** A gate that passes because
  nothing could be scored would be worse than a red build.

## Adding cases and metrics

A case is one JSON line in `golden.jsonl`; `reference` is optional but unlocks
half the metrics. A metric is one row in `evaluation/deepeval_eval/metrics.py`
naming a DeepEval class as `"module:ClassName"` — nothing else changes, and
`tests/test_deepeval_eval.py` then checks that the row's declared inputs match
what that metric actually asks for.

For something DeepEval does not ship, write a `BaseMetric` subclass and name it
the same way; `similarity.py` is the worked example, and it costs no judge
calls at all.

## Layout

```
golden.jsonl        the questions, expected slugs, and reference answers
harness.py          pure scoring: recall@k, MRR, section recall, citation precision
live.py             builds the embedder/retriever/pipeline both runners measure
run_eval.py         CLI: the non-LLM harness against the live index
run_deepeval.py     CLI: the judged harness, by stage
plots.py            CLI: figures for a stored run (needs the [eval] extra)
results/            stored runs, tracked in git — see results/README.md
deepeval_eval/
  samples.py        pure: golden case + pipeline output -> a judged record
  metrics.py        the stage -> metric table (lazy, by name)
  judge.py          the judge model and its disk cache — the only module needing keys
  similarity.py     the one metric DeepEval lacks, on the corpus embedder
  runner.py         scores samples x metrics concurrently; records, never raises
  report.py         pure: aggregate, format, and gate
```

Only `judge.py`, `similarity.py` and `metrics.build_metric` import DeepEval, so
everything that decides *what* gets scored is unit-tested offline with no SDK
and no network — 52 of the 68 tests run with the extra uninstalled.

## Three DeepEval wiring facts (hard-won; all three are silent or late failures)

1. **Never pass `temperature` to the judge.** Anthropic's `Messages.create()`
   rejects sampling parameters outright, and `AnthropicModel` forwards one if
   it finds it — including from DeepEval's own `TEMPERATURE` env var.
   `assert_no_temperature` fails at construction instead of on every call.
2. **A metric built without `model=` is an OpenAI metric.** DeepEval resolves a
   missing or string-valued model to `OpenAIModel` and fails deep inside the
   metric on a missing `OPENAI_API_KEY`. There is no OpenAI key here by design,
   so `build_metric` refuses up front.
3. **`GEval._required_params` is a bare annotation, never assigned.** Reading it
   yields a `typing` object rather than parameters; G-Eval's real inputs are the
   `evaluation_params` it was built with. `required_fields` checks both, and a
   test pins it.

One more that is a design constraint rather than a bug: G-Eval scores through
log probabilities when the judge is an OpenAI model and falls back to schema
extraction otherwise, which is the path taken here — Anthropic returns no log
probabilities. The fallback is supported upstream; the practical effect is that
`answer_correctness` lands on coarser values than a GPT judge would give.

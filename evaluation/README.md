# evaluation/

Two harnesses over one golden set. They answer different questions, and the
second one costs money, so start with the first.

| | `run_eval.py` | `run_ragas.py` |
| --- | --- | --- |
| Asks | was the right professor retrieved, and cited? | is the answer grounded, relevant, correct? |
| Scores with | set arithmetic over slugs | an LLM judge (Ragas) |
| Cost | one embed + one search per case | dozens of judge calls per case |
| Needs | `PINECONE_API_KEY`, `MISTRAL_API_KEY` | + `ANTHROPIC_API_KEY`, the `eval` extra |

```bash
pip install -e ".[eval]"                                  # ragas is not in requirements.txt

python -m evaluation.run_eval                             # recall@k / MRR
python -m evaluation.run_ragas --stage retrieval          # judged, no generation
python -m evaluation.run_ragas                            # all three stages
python -m evaluation.run_ragas --dump evaluation/runs/$(date +%F).jsonl
python -m evaluation.run_ragas --from-dump evaluation/runs/2026-09-17.jsonl
python -m evaluation.run_ragas --min faithfulness=0.8 --min context_recall=0.7

# a stored experiment: a reproducible random sample, saved with its settings
python -m evaluation.run_ragas --sample 10 --seed 7 \
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
- `context_entity_recall` — did the people/labs the answer needs get retrieved?

**`generation`** — runs the real pipeline and judges the answer against the
context it was actually given.

- `faithfulness` — does the answer assert anything the context does not support?
- `response_groundedness` — is the answer traceable to the context?
- `context_utilization` — did the answer use the top-ranked chunks, or ignore them?
- `answer_relevancy` — does it address the question, or pad around it?

**`end_to_end`** — compares the answer to the golden reference.

- `answer_correctness` — does it agree with the reference on the facts?
- `semantic_similarity` — is it saying the same thing? (no judge, embeddings only)
- `noise_sensitivity` — **lower is better**: how much do irrelevant retrieved
  chunks corrupt an otherwise correct answer?

Localising a regression is the point of the split: faithfulness down with
`context_recall` flat is the generator's fault, both down together is the
retriever's, and prompt work in the second case is wasted work.

## What a run actually costs (measured, 2026-09-17)

One real case (`crypto-secure-computation`, top_k=8) scored by all ten metrics,
counted by wrapping both clients:

```
metric                      judge  embed   secs
context_relevance               2      0    1.3
context_precision               8      0   13.3
context_recall                  1      0    2.4
context_entity_recall           2      0    5.1
faithfulness                    2      0    4.9
response_groundedness           2      0    1.3
context_utilization             8      0   13.9
answer_relevancy                3      2    6.4
answer_correctness              3      2   19.3
semantic_similarity             0      2    4.0
noise_sensitivity              19      0   62.4
TOTAL per case                 50      6   134
```

**Judge calls scale with `top_k`.** `context_precision` and
`context_utilization` make one call *per retrieved chunk*
(`for context in retrieved_contexts`), and `noise_sensitivity` runs
faithfulness-style verdicts per chunk on top of a statement decomposition. The
same case measured at 3 chunks cost 30 judge calls, not 50, so halving `--top-k`
roughly halves the two precision metrics. Embedding calls do **not** scale with
chunks: they come from question generation (`strictness=3`) and similarity, and
stay at 6 per case.

The rest of the count is claim decomposition (faithfulness splits the answer into
atomic statements in one call, then judges them all in a second) and dual-judge
averaging (`context_relevance`, `response_groundedness` each rate twice and mean
the result).

So the 60-case set, all three stages: **~3,000 judge requests, ~420 Mistral
embedding requests** (360 judging + 60 query embeds), **60 Opus generations**,
and roughly **40-50 minutes** wall clock at `--concurrency 4`.

Against the limits the providers themselves report:

| Provider | Reported limit | This run | Verdict |
| --- | --- | --- | --- |
| Anthropic (judge + generation) | 20,000 req/min, 10M in / 2M out tokens per min | ~3,060 requests total | not close |
| Pinecone serverless free | generous for reads | 60 queries | not close |
| **Mistral free tier** | **60 req/min** (`x-ratelimit-limit-req-minute`) | **~420 requests** | **the binding constraint** |

Which is why `build_judge_embeddings` paces the judge embedder at
`EMBED_RATE_LIMIT_SLEEP_SECONDS` (1s), the free tier's own ceiling — the same
knob ingest uses. Unpaced, the embedding metrics collapse into 429 backoff.

Two things worth knowing before you spend it:

- `noise_sensitivity` alone is 19 of the 50 judge calls and 62 of the 134
  seconds. Dropping `--stage end_to_end` removes it plus `answer_correctness`
  and `semantic_similarity`: 28 judge calls and 2 embeds per case instead of 50
  and 6, which cuts Mistral traffic from ~420 to ~180 requests.
- `--stage retrieval` makes no embedding calls at all (4 metrics, 13 judge
  calls per case) and never invokes the generator.

## Keeping it cheap

The zero-cost constraint applies here too, and a judged run is the one part of
this repo that spends real quota:

- `--stage retrieval` skips the generator entirely.
- Judge responses are cached on disk in `.ragas_cache/` (gitignored), keyed on
  the prompt — an unchanged re-run is nearly free, a changed pipeline re-judges.
- `--dump` records the retrieved contexts and answers; `--from-dump` re-judges
  that recording without touching Pinecone or the chat model. Iterate on
  metrics, thresholds, and reference answers this way.
- A dump is re-joined to `golden.jsonl` by case id, so you can write reference
  answers **after** a run and score `end_to_end` on it for the price of judging.
- `--limit N` bounds everything.
- The judge defaults to a cheap model (`claude-haiku-4-5-*`). `--judge-model`
  overrides it, but absolute numbers shift between judges, so do not compare
  runs scored by different ones.

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
half the metrics. A metric is one row in `evaluation/ragas_eval/metrics.py`
naming a Ragas class as `"module:ClassName"` — nothing else changes, and
`tests/test_ragas_eval.py` then checks that the row's declared inputs match what
that metric actually asks for.

## Layout

```
golden.jsonl        the questions, expected slugs, and reference answers
harness.py          pure scoring: recall@k, MRR, section recall, citation precision
live.py             builds the embedder/retriever/pipeline both runners measure
run_eval.py         CLI: the non-LLM harness against the live index
run_ragas.py        CLI: the judged harness, by stage
plots.py            CLI: figures for a stored run (needs the [eval] extra)
results/            stored runs, tracked in git — see results/README.md
ragas_eval/
  samples.py        pure: golden case + pipeline output -> a judged record
  metrics.py        the stage -> metric table (lazy, by name)
  judge.py          the judge LLM and embeddings — the only module needing keys
  runner.py         scores samples x metrics concurrently; records, never raises
  report.py         pure: aggregate, format, and gate
```

Only `judge.py` imports Ragas at module scope, so everything that decides *what*
gets scored is unit-tested offline with no SDK and no network.

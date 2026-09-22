# Rerank experiment — the question set

> **This lives on `origin/exp-reranker`**, following the same convention as
> `exp-topk` and `exp-select-llm`: the experiment and its recorded run stay on
> their own branch, while the thing they justified ships on `main`.
>
> On `main` you will find `core/rerank/` (the production reranker, ON by
> default) and its unit tests. You will not find this harness, the 60-question
> set, or `evaluation/results/2026-09-22-rerank/` — check this branch out when
> you need to re-run or re-read the measurement.
>
> `evaluation/topk/` is vendored here too, byte-identical to `origin/exp-topk`,
> because this experiment imports its judge and reuses its 900 cached verdicts.

60 cases in `questions.jsonl`. **None were written for this experiment.** Both
sources predate the reranker, so no question can have been shaped, even
unconsciously, by what the reranker happens to do well.

## Result (run 2026-09-22)

Full write-up: [`evaluation/results/2026-09-22-rerank/notes.md`](../results/2026-09-22-rerank/notes.md).
Replay it free with `python -m evaluation.rerank --from-dump`.

**Reranking helps, modestly. The cutoff does not — ship none.**

- **MRR 0.852 -> 0.900** at k=11; precision@8 **59.6% -> 62.7%**. Recall
  identical at full depth, as it must be.
- The gain is concentrated: **`courses-narrow` +0.152 MRR**, the stratum with
  the worst precision in the corpus (15.2%). Pooling hides this.
- `people-narrow` regressed -0.100, from a starting 1.000 on n=5.
- The blended control came out **flat**, as pre-registered.
- **No cutoff survives.** Relevant chunks median 0.121 vs irrelevant 0.003 —
  40x apart, far better than cosine ever gave — but the tails overlap and the
  cheapest nonzero cutoff already discards 36% of relevant chunks. The
  originally proposed **0.5 would discard 83%**.

`RERANK_MIN_SCORE` therefore stays `0.0`. Reranking is worth having for the
ORDER it produces, not for a number to threshold on.

## Composition

| source | cases | namespace | stratum |
| --- | --- | --- | --- |
| `origin/exp-topk` | 18 | `courses` | broad |
| | 12 | `courses` | narrow |
| | 6 | `courses` | noans |
| | 7 | `people` | broad |
| | 5 | `people` | narrow |
| | 2 | `people` | noans |
| `origin/exp-select-llm` | 10 | **null** (all) | blended |

**14 people / 36 courses = 72% courses**, against a corpus that is 73% course
vectors (4,105 people and 10,948 courses). Ids and notes are carried over
unchanged, so every case traces to where it came from.

## Why not `golden.jsonl`

It is **60 people cases against 3 course cases**. Measuring a reranker on it
would score almost entirely the smaller quarter of the corpus and say nothing
about the other 10,948 vectors. `tests/test_eval.py` pins the split here so a
later edit cannot quietly unbalance it again.

## Why the strata matter

Taken from `evaluation/topk/README.md`, measured over 900 judged pairs:

| | precision at k=11 |
| --- | --- |
| people, broad | 96.1% |
| people, narrow | 50.9% |
| courses, broad | 91.9% |
| courses, narrow | **15.2%** |

Broad questions are already well served — the corpus simply holds more valid
answers than any `top_k`. **Narrow questions are where the noise is**, and a
cutoff is exactly the tool for them. A set without narrow cases would show a
reranker doing almost nothing and the conclusion would be an artefact of the
questions.

The 6 + 2 `noans` cases guard the other direction: reranking and a cutoff must
not start refusing questions the system can answer.

## The 10 blended cases are a CONTROL, not a target

They are the only questions anywhere that need both corpora in one query — the
path `/chat` actually serves, and one CLAUDE.md records as unmeasurable because
every harness scores a case against a single namespace. That gap is worth
closing.

But **reranking cannot fix this question shape**, and that is already measured.
From `core/retrieval/HYBRID-SEARCH-NOTES.md`, on `MATH 7233`:

```
"Gabor Lippner research interests"        -> 4 of 4 his chunks retrieved
"Who teaches MATH 7233 ... and research?" -> 0 of his chunks retrieved
```

His profile never names the course code, so the person chunk is not retrieved
at all — and a reranker only reorders what retrieval returned. The miss is
multi-hop, needing a second retrieval hop or ingest-time denormalisation.

So: expect these 10 to be **flat**. Movement here means the measurement is
wrong, which is what makes them useful. Do not read a flat result as the
reranker failing.

A blended shape reranking *could* help looks different — both corpora holding
genuinely relevant chunks that retrieval finds, where only the ordering is in
question ("which courses and which faculty cover cryptography?"). CLAUDE.md
records three such cases where a course chunk outscored the correct person
chunk. None are in this set yet.

## Reranking is now ON by default

Changed 2026-09-22 on the strength of the result below. Two consequences worth
knowing:

- **`run_eval` now reranks**, because `evaluation/live.py`'s `connect()` builds
  whatever the settings name. For a true baseline, run
  `RERANK_PROVIDER=none python -m evaluation.run_eval`. Leaving it on also
  spends one rerank request per case against the 500/month tier.
- **This experiment is unaffected.** It builds its own reranker explicitly and
  retrieves through `live.retriever`, bypassing the configured one — otherwise
  the baseline arm would be reranked too.

## Running it

```bash
python -m evaluation.rerank --dry-run     # retrieve + rerank, no judge
python -m evaluation.rerank               # the full run
python -m evaluation.rerank --corpus courses --stratum narrow
python -m evaluation.rerank --from-dump   # re-score a recording, no API calls
```

Both arms hold the **same chunks** — reranking reorders a set, it does not
change it — so each chunk is judged once and scores both. `evaluation.topk`'s
cache key is (rubric, model, effort, question, chunk_id, text) and contains
none of the ordering, so the 50 carried-over questions mostly hit its 900
existing verdicts. The 10 blended questions are new and are what the judging
actually costs.

`evaluation/topk` is **imported, never modified**. Its judge and its per-k
arithmetic are the measuring instrument; changing the instrument between
experiments would make the two sets of numbers incomparable. `blended` is
absent from its `STRATA`, so this package carries its own four-value tuple
rather than editing that one.

Budget: **one rerank request per question**, against 500/month. A full run
spends 60. `--from-dump` spends nothing.

## Reading recall in the output

At the full retrieved depth the two arms **must** show identical recall —
reordering cannot add a chunk retrieval never returned, so a difference there
is a bug, not a result.

At any smaller k recall can move **both ways**, because the prefix is then a
genuinely different set of chunks. That is the entire point of reranking, and
equally it is how a cutoff loses recall. MRR and precision are where a working
reranker shows up first.

## Loading

`namespace: null` is new: an **absent** key still defaults to `people` (the 60
original golden cases omit it), while an **explicit null** means every
namespace. `EvalCase.from_dict` tells them apart by key presence, since
`raw.get()` collapses the two.

`evaluation/topk`'s `STRATA` does not include `blended`; the rerank harness
needs its own tuple, or `stratum_of` returns `""` for those 10.

# evaluation/results/

Stored evaluation runs. One directory per run, named `YYYY-MM-DD-<what>`.

This folder is **tracked in git**, unlike `evaluation/runs/` (gitignored), which
is for throwaway dumps. The difference is intent: a result kept here is evidence
for a decision — "did this chunking change help?" — and that only works if the
run it came from is still readable months later, alongside the settings that
produced it.

## What each run directory holds

| file | what it is |
| --- | --- |
| `notes.md` | what the run was for, and what it found |
| `samples.jsonl` | the recorded run: question, retrieved contexts, answer, reference, retrieved ids. Replayable with `--from-dump`, so metrics can be re-judged without re-paying for retrieval or generation |
| `report.json` | every per-case score with the judge's reason, per-metric summaries, and the full run metadata (stages, seed, top_k, min_score, index, judge model) |
| `report.txt` | the console report as printed, for reading without a JSON viewer |
| `plots/` | figures rendered from the JSON by `python -m evaluation.plots <dir>` |

The two probe experiments (`-score-floor`, `-judge-call-cost`) store `data.json`
instead of a report, since they measure the system rather than score it.

## Comparing runs

A directory whose subdirectories each hold a finished run is treated as a
comparison — `python -m evaluation.plots <parent>` then draws the by-model
figures instead of the single-run ones. `2026-09-17-llm-selection` is the
worked example: one subdirectory per model, everything but the generator held
fixed.

## Plots

```bash
python -m evaluation.plots evaluation/results/<dir>      # needs the [eval] extra
```

Static PNGs on purpose: they live in git beside the numbers they came from and
have to render in a diff, on GitHub, and in an editor, where no chart runtime
exists. `report.json` is the interactive layer — every score carries the judge's
own reason. Colour is assigned by the job (sequential for magnitude, two
categorical slots for identity) using validated defaults, and a skipped cell is
drawn in a neutral so it never reads as a zero.

## Reproducing a run

`report.json`'s `meta` block carries everything needed. A run recorded with
`--sample N --seed S` selects the same cases every time, so two runs are
comparable:

```bash
python -m evaluation.run_ragas --sample 10 --seed 7 \
  --dump evaluation/results/<dir>/samples.jsonl \
  --report-json evaluation/results/<dir>/report.json \
  --verbose | tee evaluation/results/<dir>/report.txt
```

To re-judge a stored run without touching Pinecone or the chat model (free
except for judge calls, and those are cached in `.ragas_cache/`):

```bash
python -m evaluation.run_ragas --from-dump evaluation/results/<dir>/samples.jsonl
```

## Why the sample is random, not the first N

`--limit` takes a prefix, and `golden.jsonl` is grouped by what each block
probes: narrow topics, then broad ones, then section-targeted, then role
filters, then the seven no-answer cases at the very end. A prefix would measure
one block and never see a refusal — a test in `tests/test_eval.py` asserts
exactly that about the first ten cases. `--sample` draws across the whole file
and `--seed` makes the draw repeatable.

## Reading a result without fooling yourself

- **Check the denominator.** `report.json`'s per-metric `scored` count is how
  many cases actually produced a number. A mean over three cases is not a
  measurement, and skips are common and legitimate (a broad case carries no
  reference; a refusal case has no answer to judge).
- **`noise_sensitivity` is inverted.** Lower is better. It is flagged in both
  the JSON (`lower_is_better`) and the printed table.
- **Compare like with like.** Scores move with `top_k`, `min_score`, the judge
  model, and the corpus. All four are in `meta`; if they differ between two
  runs, the difference in scores is not a quality change.
- **Read `detail`.** Every score carries the judge's own reason, which is
  usually the fastest route to understanding a surprising number.

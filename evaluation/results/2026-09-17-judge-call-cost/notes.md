# 2026-09-17 — what does one judged case cost?

## Why

Before committing to the full 60-case sweep: how many provider calls does one
case make, which metrics dominate, and what scales?

## Protocol

Wrapped `judge.agenerate` and the embedder's `embed_texts` with counters, then
scored a single case with every metric, cache disabled so every call is real.
Run twice: once on a 3-context sample, once on a real 8-context case
(`crypto-secure-computation` at the serving `top_k=8`). Counts in `data.json`,
figure in `plots/01-judge-calls.png`.

## Result

**50 judge calls and 6 embedding calls per case at `top_k=8`** (30 and 6 at
`top_k=3`), about 134 seconds if run serially.

Three metrics scale with the number of retrieved chunks, because they ask the
judge about each chunk separately (`for context in retrieved_contexts`):

| metric | 3 chunks | 8 chunks |
| --- | --- | --- |
| `context_precision` | 3 | 8 |
| `context_utilization` | 3 | 8 |
| `noise_sensitivity` | 9 | 19 |

Everything else is flat: claim decomposition costs 2 calls (split the answer
into atomic statements, then judge them all at once), dual-judge metrics cost 2
(two ratings averaged), `answer_relevancy` costs 3 (`strictness=3` generates
three questions from the answer), and `semantic_similarity` costs no judge call
at all. Embedding calls do not scale with chunks — they come from question
generation and similarity.

## What it changed

- **The judge embedder is now paced.** Mistral's free tier reports
  `x-ratelimit-limit-req-minute: 60`; a 60-case run wants ~420 embedding
  requests (360 judging + 60 query embeds). `build_judge_embeddings` sets
  `pace_seconds=EMBED_RATE_LIMIT_SLEEP_SECONDS`, the same knob ingest uses.
  Unpaced, the embedding metrics collapse into 429 backoff.
- **Anthropic is not a constraint.** This account reports 20,000 requests/min
  and 10M input tokens/min on both the judge and generation models; the full
  sweep needs ~3,000 requests.
- **`noise_sensitivity` is the expensive one** — 19 of 50 calls and 62 of 134
  seconds. Dropping `--stage end_to_end` removes it plus the two
  embedding-heavy correctness metrics, roughly halving a run.

# Does a cross-encoder reranker help, and where should the cutoff go?

Run 2026-09-22. `python -m evaluation.rerank` over
`evaluation/rerank/questions.jsonl`, 60 questions, k=11 per namespace against
`know-my-professor-m1024`.

- reranker: Pinecone hosted `bge-reranker-v2-m3`, 60 requests, **0 degraded**
- judge: `claude-fable-5-1` effort=low, 770 pairs, **550 cached**, 0 errors,
  185,148 in + 11,191 out = **$2.41**
- replay: `python -m evaluation.rerank --from-dump` (free)

**Two answers: reranking helps, modestly. The cutoff does not — ship none.**

## 1. Reranking improves ordering

| k | arm | relevant | prec | recall | MRR |
| --- | --- | --- | --- | --- | --- |
| 3 | before | 2.17 | 72.4% | 98.1% | 0.849 |
| 3 | after | 2.27 | **75.6%** | 96.2% | **0.888** |
| 5 | before | 3.31 | 66.2% | 98.1% | 0.849 |
| 5 | after | 3.40 | **68.1%** | 98.1% | **0.897** |
| 8 | before | 4.77 | 59.6% | 98.1% | 0.849 |
| 8 | after | 5.02 | **62.7%** | 98.1% | **0.897** |
| 11 | before | 6.08 | 55.2% | 100.0% | 0.852 |
| 11 | after | 6.19 | **56.3%** | 100.0% | **0.900** |

**MRR +0.051.** Modest, real, and in the direction predicted.

Recall is **identical in both arms at k=11**, which it must be: k=11 is the
full retrieved set for a single-namespace case, and reordering cannot add a
chunk retrieval never returned. A difference there would have meant a bug in
the harness, not a result.

At k=3 recall *fell* 98.1% → 96.2%. That is not a contradiction — the prefix is
a different set of chunks, so reordering moves things in and out of it. It is
also the mechanism by which a cutoff loses recall, visible here for free.

No-answer cases stayed **100% clean at every k in both arms**. Reranking does
not endanger the refusal path.

## 2. The gain is concentrated, and pooling hides it

At k=11:

| corpus | stratum | n | prec before | prec after | MRR before | MRR after |
| --- | --- | --- | --- | --- | --- | --- |
| blended | blended | 10 | 10.9% | 16.4% | 1.000 | 1.000 |
| courses | broad | 18 | 91.9% | 91.9% | 0.944 | **0.972** |
| courses | narrow | 12 | 15.2% | 15.2% | 0.667 | **0.819** |
| courses | noans | 6 | 0.0% | 0.0% | 0.000 | 0.000 |
| people | broad | 7 | 96.1% | 96.1% | 0.613 | **0.711** |
| people | narrow | 5 | 50.9% | 50.9% | 1.000 | **0.900** |
| people | noans | 2 | 0.0% | 0.0% | 0.000 | 0.000 |

- **`courses-narrow` gains +0.152 MRR**, the largest move in the run and
  exactly the stratum predicted: `exp-topk` measured its precision at k=11 as
  15.2%, the worst in the corpus. A pooled mean would have shown +0.051 and
  said nothing about where it came from.
- **`people-narrow` regressed −0.100.** It was already perfect at 1.000, so the
  reranker had nothing to gain and one question to lose. n=5, so this is a
  single question moving; worth watching, not worth acting on.
- **The blended control came out flat at 1.000**, as pre-registered. Those
  questions are multi-hop ("who teaches MATH 7233, and what do they research?")
  and the professor's chunk is never retrieved, so no reordering can surface
  it. Flat was the prediction and flat is what happened.

### An internal consistency check that passed

Precision at k=11 is **identical before and after for every single-namespace
stratum**. It has to be: precision over a fixed set is order-independent, and
k=11 is that whole set. Only `blended` moved (10.9% → 16.4%), because a blended
case retrieves 22 chunks (11 per namespace) so k=11 is a genuine prefix there.
Had a single-namespace precision moved, the harness would have been wrong.

## 3. The cutoff: no value works

| | n | min | p10 | p25 | p50 | p75 | p90 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| relevant | 323 | 0.000 | 0.007 | 0.026 | **0.121** | 0.381 | 0.758 | 0.995 |
| irrelevant | 447 | 0.000 | 0.000 | 0.001 | **0.003** | 0.018 | 0.168 | 0.939 |

The medians are 40x apart — far better separation than cosine ever gave, where
answerable (0.706–0.866) and out-of-scope (0.726–0.780) overlapped completely.
**But the tails overlap hard.** The lowest relevant chunk scores 0.000 and the
highest irrelevant scores 0.939, and 10% of relevant chunks fall below 0.007.

| cutoff | keep rel | keep irr | drop rel | drop irr | prec | rel lost |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 323 | 447 | 0 | 0 | 41.9% | 0.0% |
| 0.05 | 207 | 83 | 116 | 364 | 71.4% | 35.9% |
| 0.10 | 174 | 56 | 149 | 391 | 75.7% | 46.1% |
| 0.50 | 56 | 14 | 267 | 433 | 80.0% | **82.7%** |
| 0.80 | 30 | 4 | 293 | 443 | 88.2% | 90.7% |

The pre-registered rule — highest cutoff losing under 5% of relevant chunks —
**declined to fire**. The cheapest nonzero cutoff already costs 36%.

**`RERANK_MIN_SCORE` stays 0.0.** And the 0.5 originally proposed would have
been actively harmful: it discards 83% of the relevant chunks to buy 38 points
of precision.

This is the second time an absolute threshold has failed on this corpus, for
the same underlying reason both times: the signal ranks well and calibrates
badly. Reranking is worth having for the ORDER it produces, not for a number
to threshold on.

## 4. Known limitation of the cutoff metric

`relevant_loss` above is **chunk-level, not question-level**. Losing 36% of
relevant chunks at cutoff 0.05 does not mean 36% of questions lose their
answer: a question holding six relevant chunks can shed two and still be
answerable. The rule is therefore stricter than the real cost, and "no cutoff"
is the conservative reading rather than a proven one.

The honest follow-up is *how many questions retain at least one relevant chunk
at each cutoff*. It costs nothing — the recorded run replays with
`--from-dump`. Until that is measured, the 0.0 default stands, but it stands on
a metric that overstates the harm.

## 5. What this does not measure

- **Answer quality.** Every number here is retrieval-side. Whether a better
  ordering produces a better answer from Opus 5 is a separate question, and
  `exp-select-llm` found an acceptable-rate of 0.77–0.81 across four models,
  suggesting generation is not the binding constraint.
- **Latency.** One extra round trip per request, unmeasured here.
- **Sustained cost.** 500 rerank requests/month is one per `/chat` question.
  This run spent 60 (65 including a 5-question probe).
- **`people-narrow` at n=5** is too small to act on either way.

# 2026-09-17 — first judged run, 10-case random sample

## Why

First real use of the Ragas harness against the live system. The goal was not a
quality verdict — ten cases cannot give one — but to answer three operational
questions before paying for the full 60-case sweep:

1. Does the whole pipeline (retrieve → generate → judge → report) hold up on
   real data, end to end?
2. What does a run actually cost?
3. Do the metrics say anything useful about *this* product, or do some of them
   break on the shape of question it answers?

Question 3 turned out to be the interesting one.

## Protocol

```bash
python -m evaluation.run_ragas --sample 10 --seed 7 \
  --dump evaluation/results/2026-09-17-sample10/samples.jsonl \
  --report-json evaluation/results/2026-09-17-sample10/report.json \
  --verbose | tee evaluation/results/2026-09-17-sample10/report.txt
```

All three stages. `top_k=8`, `min_score=0.35`, index
`know-my-professor-m1024`, answers from `claude-opus-5`, judge
`claude-haiku-4-5-20251001`, concurrency 4. Seed 7 selects:

```
computing-on-encrypted-data     secure-compilation
formal-verification-systems     accessible-computing
robotics-broad                  games-broad
paraphrase-image-understanding  projects-clinical-simulation
web-summary-scalable-data       speech-technology
```

Cost: **453 judge calls**, 10 Opus generations, ~60 Mistral embeds, **~6 minutes**
wall clock (01:41:39 → 01:47:46), of which ~2 minutes was the sequential
generation pass.

## Results

Figures: `plots/01-all-scores.png` (every score), `02-metric-means.png`,
`03-utilization-vs-faithfulness.png` (finding 2), `04-correctness-vs-similarity.png`
(finding 3).


```
NON-LLM BASELINE          recall@8 100.0%   MRR 0.950   citation precision 53.3%

RETRIEVAL                   mean    n  skip
  context_relevance        1.000   10     0
  context_precision        0.562    8     2
  context_recall           0.677    8     2
  context_entity_recall    0.510    8     2

GENERATION
  faithfulness             0.929   10     0
  response_groundedness    0.900   10     0
  context_utilization      0.425   10     0
  answer_relevancy         0.795   10     0

END TO END
  answer_correctness       0.517    8     2
  semantic_similarity      0.928    8     2
  noise_sensitivity        0.293    8     2   (lower is better)
```

All 12 skips are the two broad cases (`robotics-broad`, `games-broad`) against
the six reference-based metrics — by design, and confirmation that the
skip-not-zero rule works.

## What it found

**1. Retrieval and grounding are healthy.** recall@8 100%, MRR 0.950 (the right
professor is almost always rank 1), `context_relevance` 1.000, `faithfulness`
0.929, `response_groundedness` 0.900. On this sample the system retrieves
relevant material and does not invent claims.

**2. `context_utilization` is meaningless for this product, and its printed hint
is actively wrong.** It scored exactly 0.00 on four cases — `accessible-computing`,
`robotics-broad`, `games-broad`, `speech-technology` — while the same cases
scored 0.87–1.00 on faithfulness and 0.75–1.00 on groundedness. The four are
precisely the cases whose answer is a **list of several people**; the cases it
scored well (`secure-compilation` 1.00, `projects-clinical-simulation` 1.00,
`web-summary-scalable-data` 0.83) all have single-person answers.

The mechanism: `ContextPrecisionWithoutReference` asks, per chunk, "was this
context useful in arriving at the given answer?" For "Maitraye Das researches
accessible computing [1]. Kashif Imteyaz works on accessibility tools [3].
Rudaiba Adnin studies accessibility of GenAI tools [4]." no single chunk yields
the whole answer, and the judge says no to each. `context_precision` on that
same case is 1.00, so the chunks are demonstrably useful.

"Who works on X?" → a list **is** this product's dominant question shape, so the
0.425 mean is an artifact, not a diagnosis. The report's legend line ("the answer
ignored the top-ranked chunks") is misleading here and has been amended.
Recommendation: drop the metric, or restrict it to single-answer cases.

**3. `answer_correctness` is measuring my reference-writing, not the answers.**
0.517 mean while `semantic_similarity` is 0.928 on the same cases. Semantic
similarity says the answer and the reference are about the same thing; the
claim-level F1 disagrees because the answers correctly name *more* people and
detail than my deliberately short 2–3 sentence references, and every extra true
statement counts as a false positive. Lesson for the golden set: a reference has
to be as complete as the ideal answer, or thoroughness is penalised. The same
effect inflates `noise_sensitivity` (0.293).

**4. This sample says nothing about refusals.** Seed 7 drew zero of the seven
`expect_no_answer` cases (expected ~1.2). They were measured separately with
`run_eval --generate`: 7/7 declined correctly. A future sample should either be
larger or stratified.

**5. Citation precision 53.3%** is partly the same golden-set artifact: on the
broad cases the answer cites valid professors who are not in `expected_slugs`.
Worth re-checking once references are widened.

## What to change before the full 60-case run

- Drop `context_utilization`, or gate it to single-answer cases (finding 2).
- Rewrite references to name everyone the ideal answer would, so
  `answer_correctness` and `noise_sensitivity` measure correctness rather than
  brevity (finding 3).
- Consider `--top-k 5`: recall@8 is 100% and MRR 0.950, so chunks 6–8 are
  carrying little, and the two per-chunk metrics cost one judge call each per
  chunk.

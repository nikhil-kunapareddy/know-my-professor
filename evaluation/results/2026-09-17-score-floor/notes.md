# 2026-09-17 — what relevance floor separates in-scope from out-of-scope?

## Why

`MIN_RETRIEVAL_SCORE = 0.35` floors retrieval: anything below it is dropped, and
if nothing survives the pipeline returns the no-answer string. `CLAUDE.md` has
carried a note since the refactor that 0.35 "was chosen as a starting point, not
measured." This measures it.

## Protocol

Every one of the 60 golden cases embedded and searched at `top_k=8` with **no
floor applied**, recording:

- for the 53 answerable cases, the score of the first chunk belonging to an
  expected professor — the score a floor must stay below to keep a right answer;
- for the 7 no-answer cases, the score of the top chunk — the score a floor must
  stay above to trigger a refusal.

Raw scores in `data.json`. `plots/01-score-floor-overlap.png` is the result.

## Result

```
answerable   first correct chunk    0.705 – 0.866   (median 0.773, n=49)
out of scope top chunk              0.726 – 0.780   (n=7)
```

**The ranges overlap completely.** The out-of-scope band sits inside the
answerable band, around its median. No threshold can separate them: a floor high
enough to refuse "who works on organic chemistry?" (needs > 0.780) would discard
the correct answer on most answerable questions.

Two consequences:

1. **0.35 is inert.** Nothing in this corpus scores below ~0.70 under
   `mistral-embed-2312`, so the floor never fires and the no-answer path is
   unreachable through it. Tuning the constant is wasted effort — this is not a
   mistuning, it is the wrong mechanism for this embedding model, whose cosine
   scores occupy a narrow band because every chunk is English prose about a
   person at a university.
2. **The refusal guarantee rests entirely on the prompt and the model** — and it
   holds. Measured separately with `run_eval --golden <negatives> --generate`:
   **7/7 declined correctly**, each opening with the requested refusal and then
   honestly naming the nearest material ("I don't have that information in my
   data. The closest is Benjamin Gyori, who works on computational systems
   biology [4].").

## What to do with this

Leave `MIN_RETRIEVAL_SCORE` where it is — it costs nothing and is harmless — but
do not treat it as a safety mechanism, and do not spend time tuning it. If a
retrieval-level guard is ever wanted, it needs a **relative** signal: the
top-1 score's margin over the rest of the top-k, or a rerank step, or a
cross-encoder. An absolute cosine threshold cannot work here.

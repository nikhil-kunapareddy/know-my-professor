# Hybrid search — design notes, not yet implemented

Status: **design only, paused 2026-09-21.** Nothing in this branch changes
retrieval. These are the measurements and the argument, written down so
resuming does not mean re-deriving them.

The `Retriever` ABC docstring already anticipates this: "lets a future
hybrid/BM25 strategy slot in behind the same interface."

## Why bother — the gap, measured

Dense retrieval cannot see a token *inside* a chunk when the chunk's overall
topic is something else. The clearest case is the one CLAUDE.md already flags as
a real retrieval gap:

```
QUERY: "Which professors earned a PhD from MIT?"
  chunks retrieved (top_k=11 x 2 namespaces): 22
  chunks that actually mention MIT:            2
  people in the corpus with MIT in education:  34
```

So roughly **6% recall on a purely lexical question**, and the true positives do
not stand out: `cos-elizabeth-wilson#education` scores **0.767** while an
unrelated `cameron-moy#biography` ranks above it at **0.778**. Education chunks
are degree lists, so every degree list embeds near every other one — the query
retrieves *education chunks generically* rather than the ones containing "MIT".

This is the same pathology as the inert score floor: absolute cosine values
carry almost no signal on this corpus (CLAUDE.md, `MIN_RETRIEVAL_SCORE`).

Supporting evidence that retrieval is where the headroom is, not generation:
the model-selection experiment (`origin/exp-select-llm`) found an
acceptable-rate of **0.77–0.81 across all four arms** — about one answer in five
judged not good enough *regardless of which model writes it*.

### Two motivations that were tested and RULED OUT

Do not re-propose these; they were measured.

1. **Course codes do not need hybrid.** Mistral handles identifiers well.
   ```
   "CY 6760 Wireless and Mobile Systems Security" -> course-cy6760#course_description  0.937  rank 1
   "CS 5700"                                      -> course-cs5700#course_description  0.845  rank 1
   ```
2. **The "who teaches X and what do they research" miss is multi-hop, not
   lexical.** Measured on `MATH 7233`:
   ```
   "Gabor Lippner research interests"        -> 4 of 4 his chunks retrieved
   "Who teaches MATH 7233 ... and research?" -> 0 of his chunks retrieved
   ```
   His profile contains no "MATH 7233", so **BM25 would not find it either.**
   Fixing this needs a second retrieval hop (retrieve, extract instructor
   names, retrieve again) or ingest-time denormalisation (write the
   instructor's research into the schedule chunk). Different feature, separate
   branch.

## The blocker, and the fact that de-risks it

- The live index `know-my-professor-m1024` is **metric=cosine**, dim 1024,
  serverless aws/us-east-1. Pinecone's single-index sparse-dense hybrid
  requires **metric=dotproduct**, and **metric is immutable** — there is no
  ALTER. Single-index hybrid therefore means a new index.
- **Mistral embeddings are unit-normalised** (measured: L2 = 1.000004,
  0.999996, 1.000004). So cosine and dotproduct produce **identical rankings**
  for the dense side. A metric migration carries *zero* dense ranking risk.
- Migration would be fetch + upsert of the existing 15,053 vectors — **no
  re-embedding cost**, the vectors already exist. Note that CLAUDE.md's
  self-emptying incident was a namespace move *within* one index; across two
  indexes the source survives, so the old index remains a real rollback.
- Two indexes exist on the account already: `know-my-professor-m1024` (live)
  and `know-my-professor` (3072-dim, retained for rollback only, effectively
  dead weight). **The Starter plan's index limit was never verified** — if it
  binds, deleting the 3072-dim index frees a slot.

## How this is done in production

**Separate retrievers fused by Reciprocal Rank Fusion is the dominant modern
pattern.** Elasticsearch and OpenSearch ship an `rrf` retriever natively; Vespa
and Weaviate do the same; most RAG stacks default to it. Pinecone's own guidance
has moved this way too — single-index sparse-dense is the *older*
Pinecone-specific pattern.

**RRF fuses on ranks, not scores**, which is the property that matters here:
BM25 and cosine scores are on incomparable scales, so any weighted-score blend
needs a normalisation that drifts. This repo has already been burned by trusting
absolute cosine values.

**The other production answer is to skip hybrid and rerank instead**:
over-retrieve with dense, then run a cross-encoder over the candidates.
`evaluation/topk`'s README (on `origin/exp-topk`) independently reached for
this — it concluded the
system needs "a *relative* signal (top-1 margin over top-2..k, or a rerank
step), not a cosine threshold." Reranking often beats hybrid per unit of effort,
at the cost of per-request latency and money, where BM25 is free. **Worth
benchmarking against hybrid rather than assuming hybrid wins.**

## The three architectures

| | new dotproduct index | separate sparse index + RRF | local BM25 + RRF |
| --- | --- | --- | --- |
| migrate 15,053 vectors | yes | no | no |
| cutover risk | a real cutover | none | none |
| score-scale problem | must tune an alpha | avoided (ranks) | avoided (ranks) |
| Pinecone queries per `/chat` | 2 | 4 (parallelisable) | 2 |
| index-limit question | new index | new index | none |
| image / memory cost | none | none | ~51 MB corpus + in-memory index |
| extra staleness path | no | no | yes — corpus copy must track ingest |

Latency is not the deciding factor: Pinecone queries run ~50–100 ms against a
measured p50 generation of 3.4 s.

## Open question raised while discussing this: does two-index fusion break at scale?

**It is a real disadvantage, and the concern is correct.** Fusing two top-k
lists means a document that is, say, rank 30 in dense *and* rank 30 in sparse —
decent on both signals, top-k on neither — is **invisible to fusion**, whereas a
single sparse-dense index scores every document on the combined representation
and can surface it. The gap widens with N, because more documents fall into that
band.

Mitigation, and it is standard: **over-fetch**. Pull top-50–100 from each index
for the candidate pass with metadata excluded (cheap), fuse, then truncate to
the 11 actually shown. That recovers nearly all of those documents.

At **15,053 vectors this is not the binding constraint** — Pinecone serverless
targets millions, and the pathology needs a corpus dense with near-duplicates,
which faculty profiles and course descriptions are not. Revisit the single-index
design if the corpus reaches the millions.

## The scaling problem that actually bites: BM25 IDF is global state

Independent of the one-index-vs-two question, and a direct consequence of
choosing self-computed BM25 over a hosted sparse model.

Every term's IDF depends on how many documents contain it. So when the corpus
grows — and it went **867 → 1,650 profiles in one month** — every document
sparse vector computed under the old IDF becomes inconsistent with a query
vector computed under the new one. Document and query weights silently drift
apart; nothing raises.

| approach | correctness | cost per ingest |
| --- | --- | --- |
| Recompute IDF, re-upsert all sparse vectors | exact | full sparse re-upsert |
| Freeze IDF in a versioned snapshot | drifts as corpus grows | incremental |
| Corpus-independent weighting (no IDF) | no drift by construction | incremental |

**This breaks the `content_hash` incremental model**, which exists precisely to
avoid re-processing unchanged chunks. At 15k documents a full sparse re-upsert
takes minutes and is acceptable; it is what stops being acceptable at scale.

If IDF is frozen, it must be **versioned alongside `content_hash`** — a query
embedded under IDF v2 against documents written under v1 is a silent ranking
bug, in the same family as the namespace mismatch that returns nothing rather
than erroring.

## Decisions already taken

- **Sparse vectors: self-computed BM25**, not Pinecone's hosted
  `pinecone-sparse-english-v0`. Accepts the IDF maintenance burden above in
  exchange for control and no hosted-model dependency.
- **Scope: flag-gated, off by default.** `HybridRetriever` behind the existing
  `Retriever` ABC, a `RETRIEVAL_MODE` env var defaulting to `dense`, plus
  lexical golden cases so `run_eval` can measure the change. No production
  behaviour shifts until the numbers justify it — same discipline as the
  `top_k` and model-selection experiments.

## Still undecided

1. **Which architecture.** Recommendation on the table was *separate sparse
   index + RRF with over-fetch, IDF recomputed per ingest*, on the grounds that
   it leaves the dense index untouched so rollback is flipping a flag. Not
   accepted yet.
2. **Whether to benchmark reranking first.** It may be the larger win and does
   not touch the index at all.
3. **Whether to quantify the prize before building.** A throwaway local BM25
   over the corpus would size the gain on the `education-*` cases in an
   afternoon, with no index work and no migration.

## When resuming

1. Re-run the measurements in this document; the corpus changes monthly.
2. Decide item 1 above.
3. Add lexical golden cases — `education-*` are the canaries, and
   `golden.jsonl` currently has only two.
4. Note that `evaluation/live.py`'s namespace bug (CLAUDE.md TODO item 5, fixed
   on `origin/exp-select-llm` commit `5a66827`) must be fixed before any
   blended-path retrieval eval, or it measures nothing and says so silently.

### Reproducing the measurements

```bash
# the MIT lexical gap
MISTRAL_API_KEY=... PINECONE_API_KEY=... .venv/bin/python - <<'PY'
import re
from evaluation.live import connect
s = connect()
PAT = re.compile(r"\b(MIT|Massachusetts Institute of Technology)\b", re.I)
v = s.embedder.embed_query("Which professors earned a PhD from MIT?")
hits = [h for ns in ("people", "courses") for h in s.retriever.retrieve(v, 11, namespace=ns)]
print(len(hits), "retrieved;", sum(bool(PAT.search(h.metadata.get("text", ""))) for h in hits), "mention MIT")
PY

# how many people actually qualify (needs the raw corpus pulled locally)
# gs://know-my-professor-raw/profiles -> .corpus/profiles, then grep education for MIT

# unit-norm check that makes a metric migration safe
.venv/bin/python -c "
import math; from shared.embeddings import build_embedder
print([round(math.sqrt(sum(x*x for x in v)), 6) for v in build_embedder('mistral').embed_texts(['a','b'])])"
```

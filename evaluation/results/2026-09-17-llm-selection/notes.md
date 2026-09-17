# 2026-09-17 — which model should generate the answers?

## Why

`claude-opus-5` has been the default generator since the provider switch, chosen
rather than measured. `/chat` runs under a 45-second request budget and the task
is short extractive summary over eight retrieved chunks — not obviously a job
that needs the largest model. This asks what the alternatives actually cost and
buy.

## Protocol

Four models over the **same 10 cases** (`--sample 10 --seed 7`), with everything
except the generator held fixed: same index, same embedder, same `top_k=8`, same
`min_score=0.35`, same prompt, same judge (`claude-haiku-4-5-20251001`).

```bash
for m in claude-opus-5 claude-sonnet-5 claude-haiku-4-5-20251001 claude-fable-5-1; do
  python -m evaluation.run_ragas --sample 10 --seed 7 \
    --stage generation --stage end_to_end --chat-model $m \
    --dump   evaluation/results/2026-09-17-llm-selection/$m/samples.jsonl \
    --report-json evaluation/results/2026-09-17-llm-selection/$m/report.json --verbose
done
```

**The retrieval stage is deliberately omitted.** `context_relevance`,
`context_precision`, `context_recall` and `context_entity_recall` score the
question, the contexts and the reference — none of which change with the
generator — so running them four times would have bought nothing. The non-LLM
baseline confirms it: recall@8 is 100% in all four runs.

Latency is the pipeline's own `generate` stage timing (`RAGResult.timings_ms`),
so embed and retrieve are excluded.

**Two candidates could not be included.** `claude-haiku-4-5` initially failed
with `400 — "This model does not support the effort parameter"`, because
`AnthropicGenerator` sent `output_config={"effort": ...}` unconditionally; that
is now fixed and is a bug that would have broken `CHAT_MODEL` for any non-Claude-5
model in production. The **Llama provider could not be tested at all**: the API
rejects the registered default model id and `models.list()` returns an empty
catalogue for this key, so the `CHAT_PROVIDER=llama` rollback path documented in
`CLAUDE.md` does not currently work.

## Results

```
metric                   fable-5-1   haiku-4-5    opus-5   sonnet-5
faithfulness                 0.941       0.884     0.927      0.857
response_groundedness        0.850       1.000     0.900      0.925
answer_relevancy             0.870       0.855     0.819      0.823
answer_correctness           0.520       0.564     0.522      0.484
semantic_similarity          0.929       0.915     0.927      0.924
noise_sensitivity ↓          0.297       0.254     0.186      0.229
context_utilization          0.258       0.470     0.258      0.488   (unreliable, see below)

median generate latency       5.7s        1.8s      3.1s       2.2s
citation precision           50.8%       60.0%     53.3%      58.7%
median answer length      —           538 ch    422 ch     284 ch
```

Figures: `plots/01-quality-by-model.png`, `02-latency-by-model.png`,
`03-quality-vs-latency.png`.

## What it says

**1. The quality differences are small; the latency differences are not.**
Faithfulness spans 0.857–0.941 and semantic similarity 0.915–0.929 — ranges that
10 cases cannot resolve. Latency spans **1.8s to 5.7s, a 3.2× spread**, measured
on the same ten questions. When one axis separates cleanly and the other does
not, the one that separates should decide.

**2. `claude-fable-5-1` is the most faithful and the worst choice here.** It
leads faithfulness (0.941) and relevancy (0.870) but takes 5.7s — three times
Haiku — for a gain inside the noise. For a chat UI answering "who works on X",
that trade is bad.

**3. `claude-haiku-4-5` is the surprise.** Fastest at 1.8s, and it leads
`response_groundedness` (1.000), `answer_relevancy`, `answer_correctness` and
citation precision (60.0%). Its faithfulness (0.884) is the weak spot — below
Opus (0.927) and Fable (0.941) — which is exactly the metric you least want to
lose on a citation product.

**4. Opus's real edge is noise sensitivity** (0.186, best of the four): it is
least likely to let an irrelevant retrieved chunk corrupt a correct answer.
Combined with its faithfulness, that is a coherent "least likely to make things
up" profile, bought at 1.7× Haiku's latency.

**5. Ignore the `context_utilization` row.** The
[2026-09-17-sample10](../2026-09-17-sample10/notes.md) run established that this
metric reads 0 on list-style answers regardless of quality; the variation here
tracks answer shape, not model quality.

## The confound worth naming

**The judge is `claude-haiku-4-5`, and so is one of the candidates.** LLM judges
show self-preference, so Haiku's strong showing on the judged metrics is exactly
where a bias would surface. Two of its wins are judge-independent — latency, and
citation precision, which is computed by string parsing in
`evaluation/harness.py` — and both still favour it. But before acting on the
judged metrics, re-run with `--judge-model claude-opus-5` and check the ranking
holds.

## Recommendation

Do not switch on this run alone: n=10, single-shot latencies from one machine,
and the self-preference confound above. What to do next, in order:

1. Re-run these four with `--judge-model claude-opus-5` (10 cases, no
   regeneration needed — `--from-dump` re-judges the stored answers for the
   price of judge calls alone).
2. If the ranking holds, run Haiku and Opus over all 60 cases.
3. If Haiku's faithfulness gap closes at n=60, switch the default: the 1.3s
   saved per request is worth more to this product than a difference in
   faithfulness that 60 cases still cannot resolve.

Switching is a one-line env change on the service (`CHAT_MODEL`), now that the
`effort` bug is fixed — no rebuild, and reversible.

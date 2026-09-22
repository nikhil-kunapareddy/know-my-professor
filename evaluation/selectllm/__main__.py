"""CLI for the model-selection experiment.

    python -m evaluation.selectllm dry-run      # free: projected cost per arm
    python -m evaluation.selectllm generate     # spends: 4 arms x 50 x repeats
    python -m evaluation.selectllm rank         # spends: blind counterbalanced
    python -m evaluation.selectllm report       # free: reads the dumps

``generate`` and ``rank`` are resumable and append to their dumps, so an
interrupted run resumes instead of re-paying.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from shared.config import MIN_RETRIEVAL_SCORE

from . import generate as gen
from . import rank as rk
from . import report as rep
from .arms import ARMS, JUDGE_MODEL, PRICING
from .cases import load

RESULTS = Path(__file__).resolve().parents[2] / "evaluation" / "results" / "2026-09-21-selectllm"
SAMPLES = RESULTS / "samples.jsonl"
RANKS = RESULTS / "ranks.jsonl"
#: The exp-topk recommendation, not the live DEFAULT_TOP_K of 8. See PLAN.md.
TOP_K = 11


def _live():
    from evaluation.live import connect

    return connect()


def cmd_dry_run(args) -> None:
    """Projected cost per arm from count_tokens. Zero generation calls."""
    from .client import client

    cases = load()
    system = _live()
    embedder = gen.CachingEmbedder(system.embedder)
    pipeline = system.pipeline(top_k=TOP_K, min_score=MIN_RETRIEVAL_SCORE)

    from core.llm.prompts import SYSTEM_INSTRUCTION

    counts = []
    sample_ids = [c.id for c in cases[:: max(1, len(cases) // args.probe)]][: args.probe]
    for case in [c for c in cases if c.id in set(sample_ids)]:
        results = []
        query = embedder.embed_query(case.question)
        for ns in pipeline.namespaces:
            results.extend(system.retriever.retrieve(query, TOP_K, namespace=ns))
        results.sort(key=lambda r: r.score, reverse=True)
        message = pipeline.prompt_builder.build_user_message(case.question, results)
        counted = client().messages.count_tokens(
            model="claude-opus-5",
            system=SYSTEM_INSTRUCTION,
            messages=[{"role": "user", "content": message}],
        )
        counts.append(counted.input_tokens)
        print(f"  {case.id:34s} {len(results):3d} chunks  {counted.input_tokens:6,d} in")

    mean_in = statistics.mean(counts)
    print(f"\nmeasured input: mean {mean_in:,.0f} tokens over {len(counts)} cases "
          f"(median {statistics.median(counts):,.0f})")

    calls = len(cases) * args.repeats
    print(f"\nprojected generation: {len(ARMS)} arms x {len(cases)} cases x "
          f"{args.repeats} repeats = {len(ARMS) * calls} calls")
    total = 0.0
    for name, arm in ARMS.items():
        for label, out in (("lean", 500), ("mid", 1000), ("high", 2200)):
            cost = calls * arm.cost(int(mean_in), out)
            if label == "mid":
                total += cost
                print(f"  {name:13s} {arm.model:18s} out~{out:5d} -> ${cost:7.2f}")
    judged = [c for c in cases if c.judged]
    rank_calls = len(judged) * args.repeats * 2
    rank_each = PRICING[JUDGE_MODEL]
    rank_cost = rank_calls * ((2400 * rank_each.input + 400 * rank_each.output) / 1e6)
    print(f"\nprojected ranking: {rank_calls} calls ({len(judged)} judged cases "
          f"x {args.repeats} repeats x 2 orderings) -> ${rank_cost:.2f}")
    print(f"\nPROJECTED TOTAL (mid): ${total + rank_cost:.2f}")


def cmd_generate(args) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = load()
    samples = gen.run(
        _live(), cases, SAMPLES, top_k=TOP_K,
        min_score=MIN_RETRIEVAL_SCORE, repeats=args.repeats,
    )
    spent = sum(s.cost_usd for s in samples if s.error is None)
    errors = [s for s in samples if s.error]
    print(f"{len(samples)} samples in {SAMPLES.name}; spent ${spent:.2f}")
    if errors:
        print(f"{len(errors)} errored:")
        for s in errors[:5]:
            print(f"  {s.case_id} {s.arm}: {s.error}")


def cmd_rank(args) -> None:
    from .client import Spend

    cases = load()
    samples = list(gen.existing(SAMPLES).values())
    if not samples:
        raise SystemExit(f"no samples at {SAMPLES}; run generate first")
    spend = Spend(cap_usd=args.cap)
    only = set(args.only.split(",")) if args.only else None
    results = rk.run(cases, samples, RANKS, spend, only=only)
    print(f"{len(results)} new rankings in {RANKS.name}; {spend}")
    bad = [r for r in results if r.error]
    if bad:
        print(f"{len(bad)} judge errors:")
        for r in bad[:5]:
            print(f"  {r.case_id} r{r.repeat} {r.ordering}: {r.error}")


def cmd_report(args) -> None:
    cases = load()
    samples = list(gen.existing(SAMPLES).values())
    ranks = rk.load_ranks(RANKS)
    reports, diag = rep.build(cases, samples, ranks)
    winner, why = rep.decide(reports)

    print(f"\n{'arm':14s}{'rank':>7}{'n':>5}{'accept':>8}{'refus':>7}"
          f"{'p50 s':>7}{'p95 s':>7}{'in tok':>8}{'out tok':>8}{'$/1k':>8}")
    for r in sorted(reports, key=lambda x: x.mean_rank or 99):
        print(f"{r.arm:14s}{r.mean_rank:7.2f}{r.ranked_n:5d}"
              f"{r.accept_rate:8.2f}{f'{r.refusal_correct}/{r.refusal_total}':>7}"
              f"{r.p50_generate_ms / 1000:7.1f}{r.p95_generate_ms / 1000:7.1f}"
              f"{r.mean_input_tokens:8,.0f}{r.mean_output_tokens:8,.0f}"
              f"{r.cost_per_1k:8.2f}")

    print("\nmean rank by stratum")
    strata = sorted({s for r in reports for s in r.mean_rank_by_stratum})
    print(f"{'arm':14s}" + "".join(f"{s:>14s}" for s in strata))
    for r in sorted(reports, key=lambda x: x.mean_rank or 99):
        cells = []
        for s in strata:
            mean_n = r.mean_rank_by_stratum.get(s)
            cells.append(f"{mean_n[0]:.2f} (n={mean_n[1]})" if mean_n else "-")
        print(f"{r.arm:14s}" + "".join(f"{c:>14s}" for c in cells))

    print("\ndiagnostics")
    print(f"  order flips           {diag['order_flips']}/{diag['order_pairs']} "
          f"({diag['order_flip_rate']:.0%}) -- reversing arm order changed the ranking")
    print(f"  judge errors          {diag['judge_errors']}")
    print(f"  divergent chunk sets  {len(diag['cases_with_divergent_chunks'])} "
          "(arms must see identical chunks for the comparison to be paired)")
    print(f"  truncated answers     "
          f"{ {r.arm: r.truncated for r in reports if r.truncated} or 'none'}")
    print(f"  safety refusals       "
          f"{ {r.arm: r.refused_by_safety for r in reports if r.refused_by_safety} or 'none'}")
    print(f"  generation spend      ${diag['total_cost_usd']:.2f}")

    print(f"\nDECISION: {why}")
    if winner:
        print(f"  -> {winner.arm} ({winner.model}, effort={winner.effort}) "
              f"at ${winner.cost_per_1k:.2f}/1k queries")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"arms": [r.__dict__ for r in reports], "diagnostics": diag,
             "winner": winner.arm if winner else None, "why": why}, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(prog="evaluation.selectllm")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dry-run", help="projected cost, no generation calls")
    d.add_argument("--repeats", type=int, default=2)
    d.add_argument("--probe", type=int, default=6)
    d.set_defaults(func=cmd_dry_run)

    g = sub.add_parser("generate", help="run every arm over every case")
    g.add_argument("--repeats", type=int, default=2)
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("rank", help="blind counterbalanced ranking")
    r.add_argument("--cap", type=float, default=40.0)
    r.add_argument("--only", default="", help="comma-separated case ids")
    r.set_defaults(func=cmd_rank)

    p = sub.add_parser("report", help="apply the pre-registered decision rule")
    p.add_argument("--json", default="")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

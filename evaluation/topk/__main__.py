"""Find the best top_k for ONE namespace, by measurement.

    python -m evaluation.topk --namespace people --dry-run    # retrieve only, cost estimate
    python -m evaluation.topk --namespace people              # the real run
    python -m evaluation.topk --namespace courses
    python -m evaluation.topk --namespace people --from-dump  # re-score, no API calls

Every question is scored against exactly ONE namespace — the corpus that owns
it — because the knob being tuned is per-namespace. Retrieval happens once at
``--k-max``; each smaller k is a prefix of that result, so the sweep costs one
retrieval and one judging pass.

Required env: ``PINECONE_API_KEY`` and the embedding provider's key for
retrieval, ``ANTHROPIC_API_KEY`` for the judge (not needed with ``--from-dump``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.harness import load_cases

from .experiment import (
    INPUT_USD_PER_MTOK,
    OUTPUT_USD_PER_MTOK,
    STRATA,
    judge_cases,
    metrics_at,
    read_cases,
    retrieve_cases,
    stratum_of,
    write_cases,
)
from .judge import DEFAULT_EFFORT, DEFAULT_JUDGE_MODEL
from .report import choose_k, format_choice, format_table

HERE = Path(__file__).parent
DEFAULT_QUESTIONS = HERE / "questions.jsonl"
DEFAULT_RESULTS = Path("evaluation/results/2026-09-21-topk")
DEFAULT_CACHE = Path(".topk_cache")

#: The sweep. Odd values two apart, so "marginal gain per +2 k" is one step.
DEFAULT_KS = (3, 5, 7, 9, 11)


def _parse_ks(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(v) for v in raw.split(",") if v.strip()}))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {raw!r}") from None
    if not values:
        raise argparse.ArgumentTypeError("--ks needs at least one value")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--namespace", required=True, help="Score only the questions owned by this namespace"
    )
    parser.add_argument("--ks", type=_parse_ks, default=DEFAULT_KS, help="Sweep, comma-separated")
    parser.add_argument(
        "--k-max", type=int, default=None, help="Retrieval depth (default: the largest --ks)"
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"]
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="First N questions only")
    parser.add_argument(
        "--stratum",
        default=None,
        choices=list(STRATA),
        help="Score only one stratum. Exists because narrow questions flatten "
        "immediately while broad ones may not have flattened by --k-max: "
        "extending the sweep is then only worth paying for on the broad half",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Retrieve and estimate cost, but call no judge",
    )
    parser.add_argument(
        "--from-dump",
        action="store_true",
        help="Re-score a recorded run; no Pinecone, no judge, no cost",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    cases = [c for c in load_cases(args.questions) if c.namespace == args.namespace]
    if not cases:
        sys.exit(f"no questions for namespace {args.namespace!r} in {args.questions}")
    if args.stratum:
        cases = [c for c in cases if stratum_of(c) == args.stratum]
        if not cases:
            sys.exit(f"no {args.stratum!r} questions for namespace {args.namespace!r}")
    if args.limit:
        cases = cases[: args.limit]

    k_max = args.k_max or max(args.ks)
    suffix = f"-{args.stratum}" if args.stratum else ""
    dump = args.results_dir / f"{args.namespace}{suffix}-k{k_max}-judged.jsonl"

    if args.from_dump:
        if not dump.exists():
            sys.exit(f"no recorded run at {dump}")
        judged = read_cases(dump, {c.id: c for c in cases})
        print(f"Replaying {len(judged)} case(s) from {dump}")
    else:
        from evaluation.live import connect
        from shared.settings import MissingSettingError

        try:
            live = connect()
        except MissingSettingError as e:
            sys.exit(f"error: {e}")

        print(
            f"Retrieving {len(cases)} question(s) from namespace {args.namespace!r} "
            f"at k_max={k_max} ({live.settings.index_name})"
        )
        judged = retrieve_cases(live, cases, k_max)

        pairs = sum(len(c.chunks) for c in judged)
        if args.dry_run:
            print(_estimate(pairs, judged))
            return

        from .judge import RelevanceJudge

        judge = RelevanceJudge(
            model=args.judge_model,
            effort=args.effort,
            cache_dir=None if args.no_cache else args.cache_dir,
        )
        print(f"\nJudging {pairs} pair(s) with {args.judge_model} at effort={args.effort}")
        stats = judge_cases(judge, judged, concurrency=args.concurrency)
        print(f"\n  {stats.format()}")

        write_cases(dump, judged)
        print(f"  recorded to {dump}")

    points = [metrics_at(judged, k) for k in args.ks]
    print(format_table(args.namespace, points))
    print(format_choice(choose_k(points)))

    # Broad and narrow questions answer different questions about k, so they are
    # also reported apart: a narrow question pins at one relevant chunk and is
    # flat at every k by construction, which drags the pooled marginal gain
    # toward zero and would pick a k that starves the broad questions.
    for stratum in ("broad", "narrow"):
        subset = [c for c in judged if stratum_of(c.case) == stratum]
        if not subset:
            continue
        subset_points = [metrics_at(subset, k) for k in args.ks]
        print(format_table(f"{args.namespace} / {stratum} only", subset_points))
        print(format_choice(choose_k(subset_points)))


def _estimate(pairs: int, judged) -> str:
    """A cost estimate from the real chunk sizes, for --dry-run.

    Uses 4 characters per token, which is close enough for a go/no-go and
    honest about being an estimate; the real run reports measured tokens.
    """
    characters = sum(len(c.text) for case in judged for c in case.chunks)
    input_tokens = characters / 4 + pairs * 350  # + rubric and question per call
    output_tokens = pairs * 250  # thinking is always on and bills as output
    usd = input_tokens / 1e6 * INPUT_USD_PER_MTOK + output_tokens / 1e6 * OUTPUT_USD_PER_MTOK
    return (
        f"\nDRY RUN — no judge called.\n"
        f"  {pairs} pair(s) to judge\n"
        f"  ~{input_tokens:,.0f} input + ~{output_tokens:,.0f} output tokens\n"
        f"  ~${usd:.2f} estimated (output is a guess; thinking bills as output)\n"
    )


if __name__ == "__main__":
    main()

"""Measure what reranking buys, and where the cutoff should go.

    python -m evaluation.rerank --dry-run     # retrieve + rerank, no judge
    python -m evaluation.rerank               # the real run
    python -m evaluation.rerank --corpus courses
    python -m evaluation.rerank --from-dump   # re-score, no API calls, free

Both arms hold the SAME chunks — reranking reorders a set, it does not change
it — so each chunk is judged once and scores both. `evaluation.topk` already
bought 900 of those verdicts and its cache key does not include the ordering,
so most of the judging here is free. The 10 blended questions are new and are
the ones that cost.

Env: PINECONE_API_KEY (retrieval AND reranking — the same key does both),
MISTRAL_API_KEY for the query embedding, ANTHROPIC_API_KEY for the judge.

Budget note: one rerank request per question, against a free tier of 500 per
month. A full 60-question run spends 60 of them. `--from-dump` spends none.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.harness import load_cases
from evaluation.topk.experiment import judge_cases
from evaluation.topk.judge import DEFAULT_EFFORT, DEFAULT_JUDGE_MODEL, RelevanceJudge

from .experiment import (
    STRATA,
    corpus_of,
    read_cases,
    rerank_cases,
    retrieve_cases,
    stratum_of,
    write_cases,
)
from .report import (
    choose_cutoff,
    format_arms,
    format_by_stratum,
    format_choice,
    format_cutoffs,
    format_distribution,
    format_raw_scores,
    sweep_cutoffs,
)

HERE = Path(__file__).parent
DEFAULT_QUESTIONS = HERE / "questions.jsonl"
DEFAULT_RESULTS = Path("evaluation/results/2026-09-22-rerank")

#: Shared with evaluation.topk on purpose. The verdicts already in it are the
#: whole reason this experiment is cheap; a separate cache would re-buy them.
DEFAULT_CACHE = Path(".topk_cache")

#: Retrieval depth PER NAMESPACE, matching DEFAULT_TOP_K so the baseline arm is
#: what production actually serves today.
DEFAULT_K = 11

#: Prefixes scored after reranking — "keep the best N of what we over-fetched".
DEFAULT_KS = (3, 5, 8, 11)


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
        "--corpus",
        default=None,
        choices=["people", "courses", "blended"],
        help="Score only questions against this corpus (default: all 60)",
    )
    parser.add_argument(
        "--stratum", default=None, choices=list(STRATA), help="Score only one stratum"
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K, help="Retrieval depth PER NAMESPACE"
    )
    parser.add_argument("--ks", type=_parse_ks, default=DEFAULT_KS, help="Prefixes to score")
    parser.add_argument("--rerank-model", default=None, help="Default: the provider's own")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--effort", default=DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"]
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="First N questions only")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Retrieve and rerank, print the score distribution, call no judge. "
        "This is the cheapest way to see whether the score separates anything "
        "at all — but it still spends one rerank request per question",
    )
    parser.add_argument(
        "--from-dump",
        action="store_true",
        help="Re-score a recorded run; no Pinecone, no reranker, no judge, no cost",
    )
    return parser


def _is_cached(args, question: str, chunk) -> bool:
    """Whether this pair is already in the shared cache — the cost estimate.

    Asks the judge's own key function rather than reimplementing the hash, so
    the estimate cannot drift from what the run will actually do.
    """
    if args.no_cache:
        return False
    probe = RelevanceJudge(
        model=args.judge_model, effort=args.effort, cache_dir=args.cache_dir, client=object()
    )
    path = probe._path(question, chunk.document_id, chunk.text)
    return bool(path and path.exists())


def _select(cases, args):
    if args.corpus:
        cases = [c for c in cases if corpus_of(c) == args.corpus]
    if args.stratum:
        cases = [c for c in cases if stratum_of(c) == args.stratum]
    return cases[: args.limit] if args.limit else cases


def main() -> None:
    args = _build_parser().parse_args()

    cases = _select(load_cases(args.questions), args)
    if not cases:
        sys.exit(f"no questions matched in {args.questions}")

    suffix = f"-{args.corpus}" if args.corpus else ""
    suffix += f"-{args.stratum}" if args.stratum else ""
    dump = args.results_dir / f"rerank{suffix}-k{args.k}.jsonl"

    # Imported here, not at module scope, so --help works without credentials.
    if args.from_dump:
        if not dump.exists():
            sys.exit(f"no recorded run at {dump}")
        print(f"Replaying {dump} — no API calls")
        reranked, run = read_cases(dump, {c.id: c for c in cases})
        print(f"  {len(reranked)} case(s) from a run at k={run.get('k', '?')}")
    else:
        from core.rerank import FailOpenReranker, build_reranker
        from evaluation.live import connect

        live = connect()
        reranker = FailOpenReranker(
            build_reranker("pinecone", model=args.rerank_model),
            # No re-arming: an experiment that silently half-degrades produces
            # a table nobody can interpret. If the quota dies mid-run, every
            # later case should stay degraded and be visibly excluded.
            retry_after_seconds=0,
        )

        print(
            f"Retrieving {len(cases)} question(s) at k={args.k} per namespace "
            f"from {live.settings.index_name}"
        )
        reranked = retrieve_cases(live, cases, args.k)

        print(f"\nReranking with {reranker.model} — {len(reranked)} request(s)")
        degraded = rerank_cases(reranker, reranked)
        if degraded:
            print(
                f"\n  WARNING: {degraded}/{len(reranked)} case(s) degraded and are "
                f"EXCLUDED from the cutoff analysis. Out of quota?"
            )

    if args.dry_run:
        print(format_raw_scores(reranked))
        pairs = sum(len(c.chunks) for c in reranked)
        cached = sum(
            1
            for case in reranked
            for chunk in case.chunks
            if _is_cached(args, case.case.question, chunk)
        )
        print(
            f"\n  dry run: {pairs} pair(s) to judge, {cached} already cached, "
            f"{pairs - cached} would be bought. Nothing was judged."
        )
        return

    if not args.from_dump:
        judge = RelevanceJudge(
            model=args.judge_model,
            effort=args.effort,
            cache_dir=None if args.no_cache else args.cache_dir,
        )
        print(f"\nJudging with {args.judge_model} (effort={args.effort})")
        stats = judge_cases(judge, reranked, concurrency=args.concurrency)
        print(f"  {stats.format()}")

        write_cases(
            dump,
            reranked,
            {"k": args.k, "rerank_model": reranker.model, "judge_model": args.judge_model,
             "effort": args.effort, "index": live.settings.index_name},
        )
        print(f"  recorded to {dump}")

    print(format_arms(reranked, list(args.ks)))
    print(format_by_stratum(reranked, args.k))
    print(format_distribution(reranked))
    points = sweep_cutoffs(reranked)
    print(format_cutoffs(points))
    print(format_choice(choose_cutoff(points)))


if __name__ == "__main__":
    main()

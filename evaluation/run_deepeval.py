"""Judge the RAG system stage by stage with DeepEval, against the golden set.

    python -m evaluation.run_deepeval --stage retrieval        # cheapest: no generation
    python -m evaluation.run_deepeval                          # all three stages
    python -m evaluation.run_deepeval --dump runs/today.jsonl  # record the run
    python -m evaluation.run_deepeval --from-dump runs/today.jsonl   # re-judge, no index
    python -m evaluation.run_deepeval --min faithfulness=0.8   # exit 1 below the bar

Which stage to run, and why:

    retrieval   embed + search only. Diagnoses the index, the embedding model,
                top_k, and the score floor. No generator, so it is cheap and
                answers "is the right context even reaching the model?".
    generation  runs the real pipeline and judges the answer against the context
                it was given: hallucination, evasion, ignored context.
    end_to_end  compares the answer to the golden reference. Only as good as the
                references — a case without one is skipped, not failed.

Costs, in order of what to reach for first. A judged case is dozens of judge
calls, so: ``--stage retrieval`` avoids the generator entirely, the disk cache
(on by default) makes an unchanged re-run nearly free, ``--from-dump`` re-judges
a recorded run without touching Pinecone or the chat model, and ``--limit``
bounds the whole thing. Re-joining a dump to the golden file also means you can
write reference answers *after* a run and score end_to_end on it for the price
of the judge alone.

Required env: ``ANTHROPIC_API_KEY`` (judge), plus ``PINECONE_API_KEY`` and the
embedding provider's key for anything that is not ``--from-dump``. Install with
``pip install -e ".[eval]"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from evaluation.deepeval_eval.metrics import SPECS, STAGE_ORDER, specs_for
from evaluation.deepeval_eval.samples import (
    JudgedSample,
    SampleSet,
    cases_by_id,
    read_samples,
    sample_from_result,
    sample_from_retrieval,
    write_samples,
)
from evaluation.harness import CaseOutcome, EvalReport, load_cases, sample_cases
from shared.config import DEFAULT_TOP_K, MIN_RETRIEVAL_SCORE

DEFAULT_GOLDEN = Path(__file__).parent / "golden.jsonl"

#: Judge responses are cached here, keyed on the prompt. Gitignored: it is a
#: cost optimisation, not a result.
DEFAULT_CACHE_DIR = ".deepeval_cache"


def _parse_minimum(raw: str) -> tuple[str, float]:
    """Parse a ``metric=value`` gate, rejecting unknown metric names early."""
    metric, _, value = raw.partition("=")
    metric = metric.strip()
    if metric not in SPECS:
        raise argparse.ArgumentTypeError(
            f"unknown metric {metric!r}; available: {', '.join(sorted(SPECS))}"
        )
    try:
        return metric, float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected {metric}=<number>, got {raw!r}") from None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="JSONL golden set")
    parser.add_argument(
        "--stage", action="append", choices=[*STAGE_ORDER, "all"],
        help="Pipeline stage to judge; repeatable (default: all)",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Chunks to retrieve")
    parser.add_argument(
        "--min-score", type=float, default=MIN_RETRIEVAL_SCORE,
        help="Relevance floor, as serving applies it; 0 judges raw retrieval",
    )
    parser.add_argument("--limit", type=int, default=None, help="Judge only the first N cases")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Judge N cases drawn at random (reproducible; see --seed). Prefer this to "
             "--limit, whose prefix only covers the first block of the golden file",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for --sample, so a run can be repeated"
    )
    parser.add_argument(
        "--chat-provider", default=None,
        help="Override the answer provider for this run (model-selection experiments)",
    )
    parser.add_argument(
        "--chat-model", default=None,
        help="Override the answer model for this run; everything else is held fixed",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help="Judge model (default: a cheap one; see evaluation/deepeval_eval/judge.py)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Judge calls in flight at once"
    )
    parser.add_argument(
        "--cache-dir", default=DEFAULT_CACHE_DIR, help="Where to cache judge responses"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Judge every call afresh (costs quota)"
    )
    parser.add_argument(
        "--dump", type=Path, default=None,
        help="Write the retrieved contexts and answers to JSONL for re-judging",
    )
    parser.add_argument(
        "--from-dump", type=Path, default=None,
        help="Judge a recorded run instead of calling the index and the model",
    )
    parser.add_argument(
        "--min", dest="minimums", action="append", type=_parse_minimum, default=None,
        metavar="METRIC=VALUE", help="Exit non-zero if a metric misses this (repeatable)",
    )
    parser.add_argument(
        "--report-json", type=Path, default=None,
        help="Write the full run (metadata, per-metric summaries, per-case scores) as JSON",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print every per-case score and every error"
    )
    return parser


def _median(values: list[float]) -> float | None:
    """Median, or None when nothing was timed (a --from-dump run times nothing)."""
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _resolve_stages(requested: list[str] | None) -> tuple[str, ...]:
    if not requested or "all" in requested:
        return STAGE_ORDER
    return tuple(stage for stage in STAGE_ORDER if stage in requested)


def _collect_live(cases, stages, args) -> SampleSet:
    """Run the real system over the cases and record what it produced."""
    from evaluation.live import connect
    from shared.settings import MissingSettingError

    try:
        live = connect()
    except MissingSettingError as e:
        sys.exit(f"error: {e}")

    needs_answer = any(spec.needs_response for spec in specs_for(stages))
    print(
        f"Running {len(cases)} case(s) against {live.settings.index_name} "
        f"(top_k={args.top_k}, min_score={args.min_score}, "
        f"generate={needs_answer})"
    )

    pipeline = (
        live.pipeline(args.top_k, args.min_score, args.chat_provider, args.chat_model)
        if needs_answer
        else None
    )
    sample_set = SampleSet(
        index_name=live.settings.index_name,
        top_k=args.top_k,
        min_score=args.min_score,
        chat_model=pipeline.generator.model if pipeline else "",
    )
    failures: list[str] = []
    for case in cases:
        try:
            if pipeline is not None:
                sample = sample_from_result(case, pipeline.answer(case.question))
            else:
                sample = sample_from_retrieval(
                    case, live.retrieve(case.question, args.top_k, args.min_score)
                )
        except Exception as e:  # noqa: BLE001 - one bad case must not lose the others
            # A run is minutes of quota; a blip on case 5 of 30 should not discard
            # cases 1-4 or the dump. A broken key, though, fails every case, and
            # reporting that as "nothing retrieved" would be a lie — so if
            # nothing at all succeeded, the error is re-raised below.
            failures.append(f"[{case.id}] {type(e).__name__}: {' '.join(str(e).split())[:160]}")
            print(f"  [{case.id}] FAILED — {failures[-1]}")
            sample_set.samples.append(
                JudgedSample(
                    case_id=case.id, input=case.question, expected_output=case.reference
                )
            )
            continue
        sample_set.samples.append(sample)
        state = "no answer" if sample.no_answer else f"{len(sample.retrieval_context)} chunks"
        print(f"  [{case.id}] {state}")

    if failures and len(failures) == len(cases):
        sys.exit(
            "error: every case failed against the live system, so there is nothing "
            f"to judge. First failure:\n  {failures[0]}"
        )
    if failures:
        print(f"  note: {len(failures)}/{len(cases)} case(s) failed and will only produce skips")
    return sample_set


def _rejoin_references(sample_set: SampleSet, cases) -> SampleSet:
    """Refresh each sample's reference from the golden file.

    So that writing reference answers after a run does not mean re-running it.
    """
    known = cases_by_id(cases)
    rejoined = []
    unknown = []
    for sample in sample_set.samples:
        case = known.get(sample.case_id)
        if case is None:
            unknown.append(sample.case_id)
            rejoined.append(sample)
        else:
            rejoined.append(replace(sample, expected_output=case.reference))
    if unknown:
        print(f"  note: {len(unknown)} dumped case(s) are not in the golden file: "
              f"{', '.join(unknown[:5])}")
    sample_set.samples = rejoined
    return sample_set


def _baseline(cases, sample_set: SampleSet, top_k: int, min_score: float) -> str:
    """The non-LLM retrieval scores for the same run, free and unambiguous."""
    known = cases_by_id(cases)
    outcomes = [
        CaseOutcome(
            case=known[sample.case_id],
            retrieved_ids=list(sample.retrieved_ids),
            answer=sample.actual_output or None,
        )
        for sample in sample_set.samples
        if sample.case_id in known
    ]
    if not outcomes:
        return ""
    report = EvalReport(outcomes=outcomes, top_k=top_k, min_score=min_score)
    lines = [
        "",
        "NON-LLM BASELINE (evaluation/harness.py, same run)",
        f"  recall@{top_k}: {report.recall_at_k:.1%}   MRR: {report.mrr:.3f}",
    ]
    if report.citation_precision is not None:
        lines[-1] += f"   citation precision: {report.citation_precision:.1%}"
    return "\n".join(lines)


def main() -> None:
    args = _build_parser().parse_args()
    stages = _resolve_stages(args.stage)
    specs = specs_for(stages)

    cases = load_cases(args.golden)
    if not cases:
        sys.exit(f"no cases in {args.golden}")
    if args.sample and args.limit:
        sys.exit("error: use --sample (random) or --limit (prefix), not both")
    if args.sample:
        cases = sample_cases(cases, args.sample, args.seed)
    elif args.limit:
        cases = cases[: args.limit]

    if args.from_dump:
        sample_set = _rejoin_references(read_samples(args.from_dump), cases)
        # A dump may hold more cases than this run selected; keep the selection.
        selected = {case.id for case in cases}
        sample_set.samples = [s for s in sample_set.samples if s.case_id in selected]
        if not sample_set.samples:
            sys.exit(
                f"error: none of the {len(selected)} selected case(s) are in {args.from_dump}"
            )
        print(f"Judging a recorded run: {sample_set.describe()}")
        top_k = sample_set.top_k or args.top_k
    else:
        sample_set = _collect_live(cases, stages, args)
        top_k = args.top_k
        if args.dump:
            write_samples(args.dump, sample_set)
            print(f"  wrote {len(sample_set)} sample(s) to {args.dump}")

    missing_reference = sum(1 for s in sample_set.samples if not s.expected_output)
    if missing_reference and any(spec.needs_reference for spec in specs):
        print(
            f"  note: {missing_reference}/{len(sample_set)} case(s) have no reference; "
            "reference-based metrics will skip them"
        )

    # Imported here so --help and a dry read of this file need no SDK or key.
    from evaluation.deepeval_eval.judge import (
        DEFAULT_JUDGE_MODEL,
        CachingJudge,
        build_judge,
        build_judge_embedder,
    )
    from evaluation.deepeval_eval.metrics import build_metric
    from evaluation.deepeval_eval.runner import score
    from shared.settings import MissingSettingError

    cache_dir = None if args.no_cache else args.cache_dir
    try:
        judge = (
            build_judge(args.judge_model, cache_dir=cache_dir)
            if any(spec.needs_llm for spec in specs)
            else None
        )
        embedder = (
            build_judge_embedder() if any(spec.needs_embeddings for spec in specs) else None
        )
    except MissingSettingError as e:
        sys.exit(f"error: {e}")

    print(
        f"Judging {len(sample_set)} sample(s) x {len(specs)} metric(s) "
        f"[{', '.join(stages)}] with {args.judge_model or DEFAULT_JUDGE_MODEL}"
        + (f", cache {cache_dir}" if cache_dir else ", cache off")
    )

    # A factory per metric, not a metric: DeepEval metrics hold their result on
    # the instance, so concurrent samples need one each. See runner.py.
    metrics = [
        (spec, (lambda s=spec: build_metric(s, llm=judge, embedder=embedder)))
        for spec in specs
    ]
    done = {"n": 0}
    total = len(sample_set) * len(metrics)

    def progress(case_id: str, metric: str, status: str) -> None:
        done["n"] += 1
        marker = {"ok": "", "skipped": " SKIP", "error": " ERROR"}[status]
        print(f"  {done['n']:3}/{total} [{case_id}] {metric}{marker}")

    report = score(sample_set.samples, metrics, args.concurrency, progress)

    if isinstance(judge, CachingJudge):
        print(f"  judge cache: {judge.describe_cache()}")

    baseline = _baseline(cases, sample_set, top_k, sample_set.min_score or args.min_score)
    if baseline:
        print(baseline)
    print(report.format(verbose=args.verbose))

    if args.report_json:
        meta = {
            "golden": str(args.golden),
            "stages": list(stages),
            "cases_selected": [case.id for case in cases],
            "sample": args.sample,
            "seed": args.seed if args.sample else None,
            "top_k": top_k,
            "min_score": sample_set.min_score or args.min_score,
            "index_name": sample_set.index_name,
            "chat_provider": args.chat_provider,
            "chat_model": sample_set.chat_model,
            "generate_ms_median": _median(
                [s.timings_ms.get("generate", 0.0) for s in sample_set.samples
                 if s.timings_ms.get("generate")]
            ),
            "judge_model": args.judge_model or DEFAULT_JUDGE_MODEL,
            "judge_framework": "deepeval",
            "concurrency": args.concurrency,
            "cache_dir": cache_dir,
            "from_dump": str(args.from_dump) if args.from_dump else None,
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report.to_dict(meta), indent=2, ensure_ascii=False) + "\n"
        )
        print(f"\nwrote the full run to {args.report_json}")

    problems = report.failing(dict(args.minimums or []))
    if problems:
        sys.exit("\nFAIL:\n" + "\n".join(f"  {p}" for p in problems))


if __name__ == "__main__":
    main()

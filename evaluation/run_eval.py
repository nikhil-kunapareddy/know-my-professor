"""Run the golden question set against the live index and report quality.

Hits the real embedding provider and Pinecone (and, with --generate, the chat
model), so it is a manual/CI-nightly tool rather than part of the offline suite.
The scoring itself is pure and unit-tested in tests/test_eval.py.

    python -m evaluation.run_eval                      # retrieval only
    python -m evaluation.run_eval --generate           # also score citations
    python -m evaluation.run_eval --top-k 12
    python -m evaluation.run_eval --min-recall 0.8     # exit 1 below the bar

Required env: PINECONE_API_KEY + the embedding provider's key (and the chat
provider's key with --generate).

Scores what a slug list can score: was the right professor retrieved, how high
up, and did the answer cite them. For the questions that need a judge — is the
answer supported by the context, does it address the question, is it correct —
see ``evaluation/run_deepeval.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.harness import CaseOutcome, EvalReport, load_cases
from shared.config import DEFAULT_TOP_K, MIN_RETRIEVAL_SCORE

DEFAULT_GOLDEN = Path(__file__).parent / "golden.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN, help="JSONL golden set")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Chunks to retrieve")
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help=f"Relevance floor; 0 measures raw retrieval (serving uses {MIN_RETRIEVAL_SCORE})",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Also call the chat model so citation precision can be scored (slower, costs quota)",
    )
    parser.add_argument(
        "--min-recall", type=float, default=None,
        help="Exit non-zero if recall@k falls below this (for a CI quality gate)",
    )
    args = parser.parse_args()

    cases = load_cases(args.golden)
    if not cases:
        sys.exit(f"no cases in {args.golden}")

    # Imported here so --help works without credentials or provider SDKs.
    from evaluation.live import connect
    from shared.settings import MissingSettingError

    try:
        live = connect()
    except MissingSettingError as e:
        sys.exit(f"error: {e}")

    pipeline = live.pipeline(args.top_k, args.min_score) if args.generate else None

    print(f"Evaluating {len(cases)} case(s) against {live.settings.index_name} "
          f"(top_k={args.top_k}, min_score={args.min_score}, generate={args.generate})")

    outcomes: list[CaseOutcome] = []
    for case in cases:
        if pipeline is not None:
            result = pipeline.answer(case.question)
            outcomes.append(CaseOutcome(
                case=case,
                retrieved_ids=[s.document_id for s in result.sources],
                answer=result.answer,
            ))
        else:
            kept = live.retrieve(case.question, args.top_k, args.min_score)
            outcomes.append(CaseOutcome(case=case, retrieved_ids=[r.document_id for r in kept]))
        print(f"  {'HIT ' if outcomes[-1].hit else 'MISS'} [{case.id}]")

    report = EvalReport(outcomes=outcomes, top_k=args.top_k, min_score=args.min_score)
    print(report.format())

    if args.min_recall is not None and report.recall_at_k < args.min_recall:
        sys.exit(
            f"\nFAIL: recall@{args.top_k} {report.recall_at_k:.1%} "
            f"is below the required {args.min_recall:.1%}"
        )


if __name__ == "__main__":
    main()

"""Scoring samples against metrics, concurrently and without giving up.

Takes already-built metrics and already-built samples, so it imports no Ragas
and no SDK: everything here is testable with a fake metric.

Three rules, each learned the tedious way:

- **Never raise.** A judged run is minutes long and costs real quota. One
  metric erroring on one case must not discard the other ninety-nine results, so
  every outcome — scored, skipped, failed — is recorded and reported.
- **Missing input is a skip, not a zero.** A case with no reference cannot be
  scored by a reference metric. Averaging in a zero there would read as a
  quality regression that no code change could fix.
- **NaN is a failure, not a score.** Ragas returns NaN when it cannot parse the
  judge's verdict. Left in, it silently poisons a mean (``nan`` propagates
  through arithmetic); counted as a failure, it shows up as what it is.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from typing import Any

from .metrics import MetricSpec, required_fields
from .report import MetricScore, RagasReport
from .samples import RagasSample

#: How many judge calls to have in flight. The judge is rate-limited and each
#: metric call is itself several requests, so this is deliberately modest.
DEFAULT_CONCURRENCY = 4

#: (case_id, metric_key, status) -> None. Used by the CLI to print progress.
ProgressHook = Callable[[str, str, str], None]


def _one_line(text: object, limit: int = 200) -> str:
    """Collapse a detail to one line so the report stays a table.

    Judge reasons and SDK errors both arrive multi-line (a 401 body, for
    instance, is JSON on its own line), which turns a five-row error list into a
    page.
    """
    return " ".join(str(text or "").split())[:limit]


def _skip_reason(sample: RagasSample, missing: frozenset[str]) -> str:
    """Say which input was absent, in the terms the user can act on."""
    if "reference" in missing:
        return "no reference in the golden case"
    if "response" in missing:
        return "pipeline declined to answer" if sample.no_answer else "no answer recorded"
    if "retrieved_contexts" in missing:
        return "nothing retrieved above the score floor"
    return f"missing {', '.join(sorted(missing))}"


async def _score_one(
    metric: Any,
    spec: MetricSpec,
    sample: RagasSample,
    semaphore: asyncio.Semaphore,
    progress: ProgressHook | None,
) -> MetricScore:
    """Score one (sample, metric) pair, turning any failure into a result."""
    inputs = sample.inputs()
    needed = required_fields(metric)
    missing = frozenset(needed) - sample.available()
    if missing:
        score = MetricScore(
            case_id=sample.case_id,
            metric=spec.key,
            status="skipped",
            detail=_skip_reason(sample, missing),
        )
    else:
        async with semaphore:
            try:
                result = await metric.ascore(**{name: inputs[name] for name in needed})
                value = float(result.value)
            except Exception as e:  # noqa: BLE001 - the whole point is not to abort the run
                score = MetricScore(
                    case_id=sample.case_id,
                    metric=spec.key,
                    status="error",
                    detail=_one_line(f"{type(e).__name__}: {e}"),
                )
            else:
                if math.isnan(value):
                    score = MetricScore(
                        case_id=sample.case_id,
                        metric=spec.key,
                        status="error",
                        detail="judge returned NaN (unparseable verdict)",
                    )
                else:
                    score = MetricScore(
                        case_id=sample.case_id,
                        metric=spec.key,
                        value=value,
                        status="ok",
                        detail=_one_line(getattr(result, "reason", "")),
                    )

    if progress is not None:
        progress(sample.case_id, spec.key, score.status)
    return score


async def score_async(
    samples: Sequence[RagasSample],
    metrics: Sequence[tuple[MetricSpec, Any]],
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: ProgressHook | None = None,
) -> RagasReport:
    """Score every sample against every metric, at most ``concurrency`` at once."""
    semaphore = asyncio.Semaphore(max(1, concurrency))
    scores = await asyncio.gather(
        *(
            _score_one(metric, spec, sample, semaphore, progress)
            for sample in samples
            for spec, metric in metrics
        )
    )
    return RagasReport(scores=list(scores), cases=len(samples))


def score(
    samples: Sequence[RagasSample],
    metrics: Sequence[tuple[MetricSpec, Any]],
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: ProgressHook | None = None,
) -> RagasReport:
    """Synchronous entry point for the CLI."""
    return asyncio.run(score_async(samples, metrics, concurrency, progress))

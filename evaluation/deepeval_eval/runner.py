"""Scoring samples against metrics, concurrently and without giving up.

Takes already-built samples and a *factory* per metric, so it imports no
DeepEval and no SDK: everything here is testable with a fake metric.

Why a factory and not a metric. DeepEval metrics keep their result on the
instance — ``metric.score``, ``metric.reason``, ``metric.error`` are all set by
``a_measure`` — so one instance shared across concurrently scored samples would
hand back another sample's reason, or another sample's score. Constructing a
metric costs nothing (no network, no key check), so each (sample, metric) pair
gets its own and the race cannot happen.

Three rules, each learned the tedious way:

- **Never raise.** A judged run is minutes long and costs real quota. One
  metric erroring on one case must not discard the other ninety-nine results, so
  every outcome — scored, skipped, failed — is recorded and reported.
- **Missing input is a skip, not a zero.** A case with no reference cannot be
  scored by a reference metric. Averaging in a zero there would read as a
  quality regression that no code change could fix.
- **NaN is a failure, not a score.** A judge whose verdict cannot be parsed
  yields NaN. Left in, it silently poisons a mean (``nan`` propagates through
  arithmetic); counted as a failure, it shows up as what it is.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from typing import Any

from .metrics import MetricSpec, required_fields
from .report import JudgeReport, MetricScore
from .samples import JudgedSample

#: How many judge calls to have in flight. The judge is rate-limited and each
#: metric call is itself several requests, so this is deliberately modest.
DEFAULT_CONCURRENCY = 4

#: (case_id, metric_key, status) -> None. Used by the CLI to print progress.
ProgressHook = Callable[[str, str, str], None]

#: Builds one metric instance. See the module docstring for why.
MetricFactory = Callable[[], Any]

#: Turns a sample into the object a metric scores. Injectable for one reason:
#: ``JudgedSample.to_test_case`` imports DeepEval, and the three rules this
#: module exists to enforce — never raise, skip rather than zero, NaN is a
#: failure — are exactly the ones that must hold in CI, where the ``eval`` extra
#: is not installed. A test passes the identity function and a fake metric.
TestCaseBuilder = Callable[[JudgedSample], Any]


def _default_test_case(sample: JudgedSample) -> Any:
    return sample.to_test_case()


def _one_line(text: object, limit: int = 200) -> str:
    """Collapse a detail to one line so the report stays a table.

    Judge reasons and SDK errors both arrive multi-line (a 401 body, for
    instance, is JSON on its own line), which turns a five-row error list into a
    page.
    """
    return " ".join(str(text or "").split())[:limit]


def _skip_reason(sample: JudgedSample, missing: frozenset[str]) -> str:
    """Say which input was absent, in the terms the user can act on."""
    if "expected_output" in missing:
        return "no reference in the golden case"
    if "actual_output" in missing:
        return "pipeline declined to answer" if sample.no_answer else "no answer recorded"
    if "retrieval_context" in missing:
        return "nothing retrieved above the score floor"
    return f"missing {', '.join(sorted(missing))}"


async def _score_one(
    factory: MetricFactory,
    spec: MetricSpec,
    sample: JudgedSample,
    semaphore: asyncio.Semaphore,
    progress: ProgressHook | None,
    build_test_case: TestCaseBuilder,
) -> MetricScore:
    """Score one (sample, metric) pair, turning any failure into a result."""
    try:
        metric = factory()
        missing = required_fields(metric) - sample.available()
    except Exception as e:  # noqa: BLE001 - a metric that cannot be built is a result
        score = MetricScore(
            case_id=sample.case_id,
            metric=spec.key,
            status="error",
            detail=_one_line(f"{type(e).__name__}: {e}"),
        )
        if progress is not None:
            progress(sample.case_id, spec.key, score.status)
        return score

    if missing:
        score = MetricScore(
            case_id=sample.case_id,
            metric=spec.key,
            status="skipped",
            detail=_skip_reason(sample, missing),
        )
    else:
        async with semaphore:
            score = await _measure(metric, spec, sample, build_test_case)

    if progress is not None:
        progress(sample.case_id, spec.key, score.status)
    return score


async def _measure(
    metric: Any, spec: MetricSpec, sample: JudgedSample, build_test_case: TestCaseBuilder
) -> MetricScore:
    """Run one metric and turn whatever comes back into a ``MetricScore``.

    DeepEval reports trouble two ways: by raising, and by returning normally
    with ``metric.error`` set (its own ``ignore_errors`` path). Both are checked
    — reading only the return value would record a confident 0.0 for a metric
    that in fact never ran.
    """
    def fail(detail: str) -> MetricScore:
        return MetricScore(
            case_id=sample.case_id, metric=spec.key, status="error", detail=_one_line(detail)
        )

    try:
        await metric.a_measure(build_test_case(sample), _show_indicator=False)
    except Exception as e:  # noqa: BLE001 - the whole point is not to abort the run
        return fail(f"{type(e).__name__}: {e}")

    if getattr(metric, "error", None):
        return fail(str(metric.error))
    if getattr(metric, "skipped", False):
        return MetricScore(
            case_id=sample.case_id,
            metric=spec.key,
            status="skipped",
            detail="the metric skipped this case",
        )

    value = getattr(metric, "score", None)
    if value is None:
        return fail("the metric returned no score")
    value = float(value)
    if math.isnan(value):
        return fail("judge returned NaN (unparseable verdict)")

    return MetricScore(
        case_id=sample.case_id,
        metric=spec.key,
        value=value,
        status="ok",
        detail=_one_line(getattr(metric, "reason", "")),
    )


async def score_async(
    samples: Sequence[JudgedSample],
    metrics: Sequence[tuple[MetricSpec, MetricFactory]],
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: ProgressHook | None = None,
    build_test_case: TestCaseBuilder = _default_test_case,
) -> JudgeReport:
    """Score every sample against every metric, at most ``concurrency`` at once."""
    semaphore = asyncio.Semaphore(max(1, concurrency))
    scores = await asyncio.gather(
        *(
            _score_one(factory, spec, sample, semaphore, progress, build_test_case)
            for sample in samples
            for spec, factory in metrics
        )
    )
    return JudgeReport(scores=list(scores), cases=len(samples))


def score(
    samples: Sequence[JudgedSample],
    metrics: Sequence[tuple[MetricSpec, MetricFactory]],
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: ProgressHook | None = None,
    build_test_case: TestCaseBuilder = _default_test_case,
) -> JudgeReport:
    """Synchronous entry point for the CLI."""
    return asyncio.run(score_async(samples, metrics, concurrency, progress, build_test_case))

"""Aggregating judged scores into something a decision can be made from.

Pure: no Ragas, no network. Given a list of per-case, per-metric scores it
produces the printed report and the pass/fail gate.

The formatting choices are the substance of this module:

- **A mean is printed with the number of cases behind it.** "faithfulness 0.62"
  over three scored cases out of eight is not a measurement, and a report that
  hides the denominator invites acting on one anyway.
- **Skips and errors are printed, not swallowed.** They are the most common
  reason a number looks surprising (usually: the golden case has no reference).
- **noise_sensitivity is flagged as inverted.** It is the one metric where high
  is bad; printing it in the same column as the rest without saying so is how
  someone "improves" it in the wrong direction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .metrics import LOWER_IS_BETTER, SPECS, STAGE_ORDER


@dataclass(frozen=True)
class MetricScore:
    """One metric's outcome on one case."""

    case_id: str
    metric: str
    value: float | None = None
    #: "ok" (scored), "skipped" (an input was missing), "error" (judge failed).
    status: str = "ok"
    detail: str = ""

    @property
    def scored(self) -> bool:
        return self.status == "ok" and self.value is not None


@dataclass(frozen=True)
class MetricSummary:
    """A metric's aggregate across every case it was run on."""

    metric: str
    mean: float | None
    scored: int
    skipped: int
    errors: int
    worst: tuple[str, float] | None = None

    @property
    def stage(self) -> str:
        spec = SPECS.get(self.metric)
        return spec.stage if spec else ""

    @property
    def inverted(self) -> bool:
        return self.metric in LOWER_IS_BETTER


@dataclass
class RagasReport:
    """Every score from one run, plus the aggregates over them."""

    scores: list[MetricScore] = field(default_factory=list)
    cases: int = 0

    def for_metric(self, metric: str) -> list[MetricScore]:
        return [s for s in self.scores if s.metric == metric]

    @property
    def metrics(self) -> list[str]:
        """Metric keys in stage order, unknown keys last (stable, not sorted)."""
        seen = list(dict.fromkeys(s.metric for s in self.scores))
        order = {key: (STAGE_ORDER.index(spec.stage), i) for i, (key, spec) in enumerate(SPECS.items())}
        return sorted(seen, key=lambda key: order.get(key, (len(STAGE_ORDER), 0)))

    def summary(self, metric: str) -> MetricSummary:
        scores = self.for_metric(metric)
        scored = [s for s in scores if s.scored]
        values = [s.value for s in scored if s.value is not None]
        worst: tuple[str, float] | None = None
        if scored:
            pick = max if metric in LOWER_IS_BETTER else min
            chosen = pick(scored, key=lambda s: s.value or 0.0)
            worst = (chosen.case_id, float(chosen.value or 0.0))
        return MetricSummary(
            metric=metric,
            mean=sum(values) / len(values) if values else None,
            scored=len(values),
            skipped=sum(1 for s in scores if s.status == "skipped"),
            errors=sum(1 for s in scores if s.status == "error"),
            worst=worst,
        )

    def summaries(self) -> list[MetricSummary]:
        return [self.summary(metric) for metric in self.metrics]

    def failing(self, minimums: Mapping[str, float]) -> list[str]:
        """Which gates a run misses, as printable lines.

        An unmeasured metric fails its gate. A gate that passes because nothing
        could be scored is worse than a red build: it is a green one.
        """
        problems: list[str] = []
        for metric, floor in minimums.items():
            summary = self.summary(metric)
            if summary.mean is None:
                problems.append(f"{metric}: not measured (no case could be scored)")
            elif metric in LOWER_IS_BETTER:
                if summary.mean > floor:
                    problems.append(
                        f"{metric}: {summary.mean:.3f} is above the allowed {floor:.3f}"
                    )
            elif summary.mean < floor:
                problems.append(f"{metric}: {summary.mean:.3f} is below the required {floor:.3f}")
        return problems

    def to_dict(self, meta: Mapping[str, object] | None = None) -> dict:
        """The whole run as JSON-able data, for storing next to the prose report.

        A printed table answers "how did it go"; this answers "how did case X
        score on metric Y, and what was the judge's reason" months later, when
        the console output is gone and the question is whether a change helped.
        """
        return {
            "meta": dict(meta or {}),
            "cases": self.cases,
            "metrics": {
                summary.metric: {
                    "stage": summary.stage,
                    "mean": summary.mean,
                    "scored": summary.scored,
                    "skipped": summary.skipped,
                    "errors": summary.errors,
                    "lower_is_better": summary.inverted,
                    "worst_case": summary.worst[0] if summary.worst else None,
                    "worst_value": summary.worst[1] if summary.worst else None,
                }
                for summary in self.summaries()
            },
            "scores": [
                {
                    "case_id": score.case_id,
                    "metric": score.metric,
                    "value": score.value,
                    "status": score.status,
                    "detail": score.detail,
                }
                for score in self.scores
            ],
        }

    def format(self, verbose: bool = False) -> str:
        """The printed report: a table per stage, then skips, errors, and a legend."""
        lines: list[str] = []
        summaries = self.summaries()

        for stage in STAGE_ORDER:
            in_stage = [s for s in summaries if s.stage == stage]
            if not in_stage:
                continue
            lines.append("")
            lines.append(f"{stage.replace('_', ' ').upper()}")
            lines.append(f"  {'metric':24} {'mean':>7}  {'n':>5}  {'skip':>5}  {'err':>4}  worst case")
            for summary in in_stage:
                mean = "     --" if summary.mean is None else f"{summary.mean:7.3f}"
                worst = "—"
                if summary.worst is not None:
                    worst = f"{summary.worst[0]} ({summary.worst[1]:.2f})"
                flag = "  (lower is better)" if summary.inverted else ""
                lines.append(
                    f"  {summary.metric:24} {mean}  {summary.scored:5}  "
                    f"{summary.skipped:5}  {summary.errors:4}  {worst}{flag}"
                )

        skipped = [s for s in self.scores if s.status == "skipped"]
        if skipped:
            reasons: dict[str, int] = {}
            for score in skipped:
                reasons[score.detail] = reasons.get(score.detail, 0) + 1
            lines.append("")
            lines.append(f"Skipped {len(skipped)} metric-case pair(s):")
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {count:4}  {reason}")

        errors = [s for s in self.scores if s.status == "error"]
        if errors:
            lines.append("")
            lines.append(f"Errors ({len(errors)}):")
            for score in errors[: len(errors) if verbose else 5]:
                lines.append(f"  [{score.case_id}] {score.metric}: {score.detail}")
            if not verbose and len(errors) > 5:
                lines.append(f"  ... and {len(errors) - 5} more (--verbose to list)")

        weak = [s for s in summaries if s.mean is not None and self._is_weak(s)]
        if weak:
            lines.append("")
            lines.append("What the weak numbers mean:")
            for summary in weak:
                spec = SPECS.get(summary.metric)
                if spec and spec.means:
                    lines.append(f"  {summary.metric}: {spec.means}")

        if verbose:
            lines.append("")
            lines.append("Per case:")
            for score in self.scores:
                value = "   --" if score.value is None else f"{score.value:5.2f}"
                lines.append(
                    f"  [{score.case_id:22}] {score.metric:24} {value}  {score.status}"
                    + (f" — {score.detail}" if score.detail else "")
                )

        return "\n".join(lines)

    @staticmethod
    def _is_weak(summary: MetricSummary) -> bool:
        """Weak enough to be worth explaining. A threshold for prose, not a gate."""
        if summary.mean is None:
            return False
        return summary.mean > 0.3 if summary.inverted else summary.mean < 0.8

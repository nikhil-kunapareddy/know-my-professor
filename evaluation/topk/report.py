"""The table, and the rule that reads it.

The rule is defined here — in code, before any data exists — rather than
applied by eye afterwards. An experiment whose stopping condition is chosen
after seeing the curve does not have a stopping condition; every k can be
argued for once you know its number.
"""

from __future__ import annotations

from dataclasses import dataclass

from .experiment import KMetrics

#: Stop where another +2 of k buys less than half a usable chunk per question.
#: Half a chunk is the point where the added context stops paying for itself:
#: below it, most questions gain nothing and every question pays the tokens.
MARGINAL_GAIN_THRESHOLD = 0.5

#: ...but never at a k that gives up recall. One point of recall@k is a single
#: question in a 50-case set, so this tolerates rounding and nothing more.
RECALL_TOLERANCE = 0.01


@dataclass(frozen=True)
class Choice:
    """The k the rule picked, and the reason — printed with the table."""

    k: int
    reason: str


def choose_k(points: list[KMetrics]) -> Choice:
    """Apply the pre-registered rule to an ascending list of k measurements."""
    if not points:
        raise ValueError("no measurements to choose from")
    ordered = sorted(points, key=lambda p: p.k)
    best_recall = max(p.recall_at_k for p in ordered)

    # strict=False is the point: the lists differ in length by one, which is
    # what pairs each k with the step that follows it.
    for current, following in zip(ordered, ordered[1:], strict=False):
        gain = following.mean_relevant - current.mean_relevant
        if gain >= MARGINAL_GAIN_THRESHOLD:
            continue
        if current.recall_at_k < best_recall - RECALL_TOLERANCE:
            # The curve has flattened, but this k is still dropping answers the
            # bigger ones find. Keep walking: flat-and-wrong is not the knee.
            continue
        return Choice(
            k=current.k,
            reason=(
                f"k={following.k} adds only {gain:+.2f} relevant chunks over k={current.k} "
                f"(threshold {MARGINAL_GAIN_THRESHOLD:+.2f}), and recall@{current.k} "
                f"{current.recall_at_k:.1%} is within {RECALL_TOLERANCE:.0%} of the best "
                f"observed {best_recall:.1%}"
            ),
        )

    largest = ordered[-1]
    return Choice(
        k=largest.k,
        reason=(
            f"every step still bought >= {MARGINAL_GAIN_THRESHOLD:+.2f} relevant chunks, or "
            f"gave up recall; the curve has not flattened by k={largest.k} — extend the sweep"
        ),
    )


def format_table(namespace: str, points: list[KMetrics]) -> str:
    """The measurement, one row per k."""
    ordered = sorted(points, key=lambda p: p.k)
    lines = [
        "",
        f"namespace: {namespace}   ({ordered[0].cases} question(s))",
        "",
        f"{'k':>3}  {'retrieved':>9}  {'relevant':>8}  {'marginal':>8}  "
        f"{'prec':>6}  {'recall':>7}  {'MRR':>6}  {'clean':>6}",
        f"{'':->3}  {'':->9}  {'':->8}  {'':->8}  {'':->6}  {'':->7}  {'':->6}  {'':->6}",
    ]
    previous: float | None = None
    for point in ordered:
        marginal = "     —" if previous is None else f"{point.mean_relevant - previous:+6.2f}"
        clean = "     —" if point.no_answer_clean is None else f"{point.no_answer_clean:6.1%}"
        lines.append(
            f"{point.k:>3}  {point.mean_retrieved:>9.1f}  {point.mean_relevant:>8.2f}  "
            f"{marginal:>8}  {point.precision:>6.1%}  {point.recall_at_k:>7.1%}  "
            f"{point.mrr:>6.3f}  {clean:>6}"
        )
        previous = point.mean_relevant

    unjudged = max(p.unjudged for p in ordered)
    if unjudged:
        lines += [
            "",
            f"  note: {unjudged} pair(s) the judge failed on are excluded from "
            "'relevant' and 'prec'.",
        ]

    lines += [
        "",
        "  relevant  mean chunks per question a correct answer would actually use",
        "  marginal  what this k added over the previous one — the rule reads THIS column",
        "  prec      relevant / retrieved",
        "  recall    slug-based, no judge involved — the independent check",
        "  clean     share of no-answer questions where the judge found nothing relevant",
    ]
    return "\n".join(lines)


def format_choice(choice: Choice) -> str:
    return f"\n  CHOSEN: k = {choice.k}\n  because {choice.reason}\n"

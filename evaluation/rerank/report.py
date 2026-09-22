"""The tables, and the rule that picks a cutoff.

Both rules are written here, before any data exists. A threshold chosen after
seeing the curve is not a threshold, it is a description of the curve — which
is precisely how MIN_RETRIEVAL_SCORE came to be inert.
"""

from __future__ import annotations

from dataclasses import dataclass

from evaluation.topk.experiment import KMetrics, metrics_at

from .experiment import RerankedCase, corpus_of, score_pairs, stratum_of

#: A cutoff must not cost recall to buy precision. One relevant chunk lost per
#: twenty is the most this experiment will trade, and the rule prefers the
#: highest cutoff that stays inside it — precision is the thing being bought.
MAX_RELEVANT_LOSS = 0.05

#: Candidate cutoffs swept. Deliberately includes 0.0, so "no cutoff is worth
#: it" remains an available answer rather than being excluded by construction.
CUTOFFS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


@dataclass(frozen=True)
class CutoffPoint:
    """What one candidate cutoff would keep and throw away."""

    cutoff: float
    kept_relevant: int
    kept_irrelevant: int
    dropped_relevant: int
    dropped_irrelevant: int

    @property
    def precision(self) -> float:
        kept = self.kept_relevant + self.kept_irrelevant
        return self.kept_relevant / kept if kept else 0.0

    @property
    def relevant_loss(self) -> float:
        total = self.kept_relevant + self.dropped_relevant
        return self.dropped_relevant / total if total else 0.0


@dataclass(frozen=True)
class Choice:
    cutoff: float | None
    reason: str


def sweep_cutoffs(cases: list[RerankedCase]) -> list[CutoffPoint]:
    """Score every candidate cutoff against the judged verdicts."""
    pairs = score_pairs(cases)
    points = []
    for cutoff in CUTOFFS:
        kept = [(s, rel) for s, rel in pairs if s >= cutoff]
        dropped = [(s, rel) for s, rel in pairs if s < cutoff]
        points.append(
            CutoffPoint(
                cutoff=cutoff,
                kept_relevant=sum(1 for _, rel in kept if rel),
                kept_irrelevant=sum(1 for _, rel in kept if not rel),
                dropped_relevant=sum(1 for _, rel in dropped if rel),
                dropped_irrelevant=sum(1 for _, rel in dropped if not rel),
            )
        )
    return points


def choose_cutoff(points: list[CutoffPoint]) -> Choice:
    """The highest cutoff that stays within MAX_RELEVANT_LOSS.

    Returns ``None`` when even the smallest nonzero cutoff costs too much,
    which is a real answer: it would mean the cross-encoder does not separate
    relevant from irrelevant on this corpus either, and the honest conclusion
    is to ship no cutoff rather than a decorative one.
    """
    if not points:
        raise ValueError("no measurements to choose from")
    affordable = [p for p in points if p.cutoff > 0 and p.relevant_loss <= MAX_RELEVANT_LOSS]
    if not affordable:
        return Choice(
            None,
            f"every cutoff above 0 discards more than {MAX_RELEVANT_LOSS:.0%} of the "
            f"relevant chunks; the score does not separate relevant from irrelevant "
            f"here, so ship no cutoff",
        )
    best = max(affordable, key=lambda p: p.cutoff)
    baseline = next(p for p in points if p.cutoff == 0.0)
    return Choice(
        best.cutoff,
        f"it drops {best.dropped_irrelevant} irrelevant chunk(s) while losing "
        f"{best.relevant_loss:.1%} of the relevant ones (budget {MAX_RELEVANT_LOSS:.0%}), "
        f"lifting precision {baseline.precision:.1%} -> {best.precision:.1%}",
    )


def format_arms(cases: list[RerankedCase], ks: list[int]) -> str:
    """Before/after at each k — the headline comparison."""
    lines = [
        "",
        f"  cosine order vs reranked order   ({len(cases)} question(s))",
        "",
        f"  {'k':>3}  {'arm':>8}  {'relevant':>8}  {'prec':>6}  {'recall':>7}  "
        f"{'MRR':>6}  {'clean':>6}",
        f"  {'':->3}  {'':->8}  {'':->8}  {'':->6}  {'':->7}  {'':->6}  {'':->6}",
    ]
    for k in sorted(ks):
        for label, arm in (("before", [c.before() for c in cases]),
                           ("after", [c.after() for c in cases])):
            point: KMetrics = metrics_at(arm, k)
            clean = "     —" if point.no_answer_clean is None else f"{point.no_answer_clean:6.1%}"
            lines.append(
                f"  {k:>3}  {label:>8}  {point.mean_relevant:>8.2f}  {point.precision:>6.1%}  "
                f"{point.recall_at_k:>7.1%}  {point.mrr:>6.3f}  {clean:>6}"
            )
        lines.append("")

    lines.append(
        f"  at k={max(ks)} (the full retrieved set) recall MUST be identical in both "
        f"arms —"
    )
    lines.append(
        "  reordering cannot add a chunk retrieval never returned. A difference "
        "there is a bug."
    )
    lines.append(
        "  At smaller k it can move BOTH ways, because the prefix is a different "
        "set of chunks:"
    )
    lines.append(
        "  that is exactly what reranking is for, and also exactly how a cutoff "
        "loses recall."
    )
    return "\n".join(lines)


def format_by_stratum(cases: list[RerankedCase], k: int) -> str:
    """Where the gain lands. Pooled numbers hide this completely.

    exp-topk measured course-narrow precision at 15.2% against course-broad at
    91.9%, so a pooled mean is an average of two unrelated regimes.
    """
    lines = [
        "",
        f"  by stratum at k={k}",
        "",
        f"  {'corpus':>8}  {'stratum':>8}  {'n':>3}  {'prec before':>11}  "
        f"{'prec after':>10}  {'MRR before':>10}  {'MRR after':>9}",
        f"  {'':->8}  {'':->8}  {'':->3}  {'':->11}  {'':->10}  {'':->10}  {'':->9}",
    ]
    groups: dict[tuple[str, str], list[RerankedCase]] = {}
    for case in cases:
        groups.setdefault((corpus_of(case.case), stratum_of(case.case)), []).append(case)

    for (corpus, stratum), group in sorted(groups.items()):
        before = metrics_at([c.before() for c in group], k)
        after = metrics_at([c.after() for c in group], k)
        lines.append(
            f"  {corpus:>8}  {stratum:>8}  {len(group):>3}  {before.precision:>11.1%}  "
            f"{after.precision:>10.1%}  {before.mrr:>10.3f}  {after.mrr:>9.3f}"
        )

    lines.append("")
    lines.append(
        "  blended is a CONTROL and should be flat — the person chunk is never "
        "retrieved,"
    )
    lines.append(
        "  so no reordering can surface it (see evaluation/rerank/README.md)."
    )
    return "\n".join(lines)


def format_cutoffs(points: list[CutoffPoint]) -> str:
    """The distribution question: is there a score that separates the two?"""
    lines = [
        "",
        "  cutoff sweep — what each candidate keeps and discards",
        "",
        f"  {'cutoff':>6}  {'keep rel':>8}  {'keep irr':>8}  {'drop rel':>8}  "
        f"{'drop irr':>8}  {'prec':>6}  {'rel lost':>8}",
        f"  {'':->6}  {'':->8}  {'':->8}  {'':->8}  {'':->8}  {'':->6}  {'':->8}",
    ]
    for point in points:
        lines.append(
            f"  {point.cutoff:>6.2f}  {point.kept_relevant:>8}  {point.kept_irrelevant:>8}  "
            f"{point.dropped_relevant:>8}  {point.dropped_irrelevant:>8}  "
            f"{point.precision:>6.1%}  {point.relevant_loss:>8.1%}"
        )
    return "\n".join(lines)


def format_distribution(cases: list[RerankedCase]) -> str:
    """Percentiles for relevant vs irrelevant chunks, side by side.

    If these two distributions overlap the way the COSINE ones did
    (answerable 0.706-0.866 vs out-of-scope 0.726-0.780, per CLAUDE.md), then
    no cutoff works on this signal either and the answer is to ship none.
    """
    pairs = score_pairs(cases)
    if not pairs:
        return "\n  no scored+judged pairs — was the run degraded, or unjudged?"

    relevant = sorted(s for s, rel in pairs if rel)
    irrelevant = sorted(s for s, rel in pairs if not rel)

    def at(values: list[float], pct: float) -> str:
        if not values:
            return "    —"
        return f"{values[min(int(pct * len(values)), len(values) - 1)]:.3f}"

    lines = [
        "",
        f"  rerank score distribution   ({len(pairs)} judged pair(s))",
        "",
        f"  {'':>12}  {'n':>4}  {'min':>6}  {'p10':>6}  {'p25':>6}  {'p50':>6}  "
        f"{'p75':>6}  {'p90':>6}  {'max':>6}",
        f"  {'':->12}  {'':->4}  {'':->6}  {'':->6}  {'':->6}  {'':->6}  "
        f"{'':->6}  {'':->6}  {'':->6}",
    ]
    for label, values in (("relevant", relevant), ("irrelevant", irrelevant)):
        lines.append(
            f"  {label:>12}  {len(values):>4}  {at(values, 0.0):>6}  {at(values, 0.10):>6}  "
            f"{at(values, 0.25):>6}  {at(values, 0.50):>6}  {at(values, 0.75):>6}  "
            f"{at(values, 0.90):>6}  {at(values, 1.0):>6}"
        )

    if relevant and irrelevant:
        overlap = min(relevant) < max(irrelevant)
        lines.append("")
        lines.append(
            f"  distributions {'OVERLAP' if overlap else 'SEPARATE'}"
            f" (lowest relevant {min(relevant):.3f} vs highest irrelevant {max(irrelevant):.3f})"
        )
    return "\n".join(lines)


def format_choice(choice: Choice) -> str:
    if choice.cutoff is None:
        return f"\n  CHOSEN: no cutoff\n  because {choice.reason}\n"
    return f"\n  CHOSEN: RERANK_MIN_SCORE = {choice.cutoff}\n  because {choice.reason}\n"


def format_raw_scores(cases: list[RerankedCase]) -> str:
    """Score spread with no relevance split — what `--dry-run` can know.

    This is the probe on its own: before spending anything on a judge, it says
    whether the cross-encoder even uses its range, or returns everything in a
    narrow band the way cosine does on this corpus (nothing below ~0.70, per
    CLAUDE.md, which is why the cosine floor never fires).
    """
    scores = sorted(s for case in cases if not case.degraded for s in case.rerank_scores.values())
    if not scores:
        return "\n  no rerank scores — every case degraded, or none was reranked."

    def at(pct: float) -> float:
        return scores[min(int(pct * len(scores)), len(scores) - 1)]

    return "\n".join(
        [
            "",
            f"  rerank score spread   ({len(scores)} scored chunk(s), "
            f"{sum(1 for c in cases if c.degraded)} case(s) degraded)",
            "",
            f"  {'min':>6}  {'p10':>6}  {'p25':>6}  {'p50':>6}  {'p75':>6}  "
            f"{'p90':>6}  {'max':>6}",
            f"  {'':->6}  {'':->6}  {'':->6}  {'':->6}  {'':->6}  {'':->6}  {'':->6}",
            f"  {scores[0]:>6.3f}  {at(0.10):>6.3f}  {at(0.25):>6.3f}  {at(0.50):>6.3f}  "
            f"{at(0.75):>6.3f}  {at(0.90):>6.3f}  {scores[-1]:>6.3f}",
            "",
            f"  range used: {scores[-1] - scores[0]:.3f}. A narrow band means the "
            f"score cannot separate",
            "  anything and no cutoff will work — the same finding as "
            "MIN_RETRIEVAL_SCORE.",
        ]
    )

"""Checking the judge against an independent ranker.

Labelled honestly: this measures **inter-judge agreement**, not agreement with a
human. The independent ranker here is Claude Opus 5, which is also two of the
four arms — so this cannot detect a bias the two models share, and it is not the
human validation the plan reserves.

What it does detect is the practical risk: a rubric bug. A judge ranking by
length, ignoring the ground truth, favouring the first candidate, or
misreading a premise-rejection case as a non-answer all show up as low
agreement. Those are the failures that would make the whole grid meaningless,
and they are worth catching before trusting 180 rankings.

The independent ranking is produced **blind**: the ranker sees the question, the
ground truth and answers labelled A-D, and never sees which arm produced which
answer or what the judge decided.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .cases import Case
from .generate import Sample
from .rank import LABELS, RankResult


@dataclass(frozen=True)
class Blind:
    """One case presented for independent ranking, with the mapping withheld."""

    case_id: str
    repeat: int
    ordering: str
    question: str
    ground_truth: str
    #: label -> answer text. The arm behind each label is NOT included.
    answers: dict[str, str]


def sample_for_validation(
    cases: list[Case], ranks: list[RankResult], samples: list[Sample], n: int = 15
) -> list[Blind]:
    """Pick ``n`` judged cases spread across strata, as blind presentations."""
    by_id = {c.id: c for c in cases}
    lookup = {(s.case_id, s.arm, s.repeat): s for s in samples}

    # One presentation per case (repeat 1, forward), stratified round-robin so
    # the sample is not 15 lookups.
    per_stratum: dict[str, list[RankResult]] = {}
    for r in sorted(ranks, key=lambda r: (r.case_id, r.repeat)):
        if r.repeat != 1 or r.ordering != "forward" or not r.mapping:
            continue
        case = by_id.get(r.case_id)
        if case is None:
            continue
        per_stratum.setdefault(case.stratum, []).append(r)

    picked: list[RankResult] = []
    while len(picked) < n and any(per_stratum.values()):
        for stratum in list(per_stratum):
            if len(picked) >= n:
                break
            if per_stratum[stratum]:
                picked.append(per_stratum[stratum].pop(0))

    out = []
    for r in picked:
        case = by_id[r.case_id]
        answers = {}
        for label, arm in r.mapping.items():
            s = lookup.get((r.case_id, arm, r.repeat))
            answers[label] = (s.answer if s else "") or "(the system returned no answer)"
        out.append(Blind(
            case_id=r.case_id, repeat=r.repeat, ordering=r.ordering,
            question=case.question, ground_truth=case.ground_truth, answers=answers,
        ))
    return out


def _pairs(ranks: dict[str, int], labels: list[str]) -> dict[tuple[str, str], str]:
    """Every pairwise preference implied by a ranking.

    Pairwise rather than raw rank numbers: two rankings can disagree on absolute
    positions while agreeing on every comparison, and it is the comparisons that
    decide an arm.
    """
    out = {}
    for a, b in itertools.combinations(sorted(labels), 2):
        ra, rb = ranks.get(a), ranks.get(b)
        if ra is None or rb is None:
            continue
        out[(a, b)] = "tie" if ra == rb else ("a" if ra < rb else "b")
    return out


def _kappa(mine: list[str], theirs: list[str]) -> float:
    """Cohen's kappa over {a, b, tie} preferences."""
    if not mine:
        return 0.0
    n = len(mine)
    observed = sum(1 for x, y in zip(mine, theirs, strict=True) if x == y) / n
    cm, ct = Counter(mine), Counter(theirs)
    expected = sum((cm[k] / n) * (ct[k] / n) for k in set(cm) | set(ct))
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def score(
    blind: list[Blind], independent: dict[str, dict[str, int]], ranks: list[RankResult]
) -> dict:
    """Compare the independent blind ranking to the judge's, pairwise.

    ``independent`` maps case_id -> {label: rank}.
    """
    judged = {(r.case_id, r.repeat, r.ordering): r for r in ranks}
    mine: list[str] = []
    theirs: list[str] = []
    per_case = []

    for item in blind:
        ours = independent.get(item.case_id)
        judge = judged.get((item.case_id, item.repeat, item.ordering))
        if not ours or judge is None or not judge.ranks:
            continue
        labels = sorted(item.answers)
        # The judge stores ranks by arm; map back through the withheld mapping.
        arm_to_label = {arm: label for label, arm in judge.mapping.items()}
        judge_by_label = {
            arm_to_label[arm]: rank
            for arm, rank in judge.ranks.items()
            if arm in arm_to_label
        }
        a = _pairs(ours, labels)
        b = _pairs(judge_by_label, labels)
        shared = sorted(set(a) & set(b))
        agree = sum(1 for k in shared if a[k] == b[k])
        mine.extend(a[k] for k in shared)
        theirs.extend(b[k] for k in shared)
        per_case.append({
            "case_id": item.case_id,
            "pairs": len(shared),
            "agree": agree,
            "rate": agree / len(shared) if shared else 0.0,
        })

    total = len(mine)
    return {
        "cases": len(per_case),
        "pairs": total,
        "agreement": (sum(1 for x, y in zip(mine, theirs, strict=True) if x == y) / total)
        if total
        else 0.0,
        "cohens_kappa": _kappa(mine, theirs),
        "per_case": per_case,
        "note": (
            "INTER-JUDGE agreement (Claude Opus 5 vs Claude Fable 5.1), not human "
            "validation. Cannot detect a bias the two models share."
        ),
    }


def write_blind(blind: list[Blind], path: Path) -> None:
    """Dump the blind presentations for an independent ranker to read."""
    path.write_text("\n".join(json.dumps(b.__dict__) for b in blind) + "\n")


def render(blind: list[Blind]) -> str:
    """Human-readable form of the blind presentations."""
    out = []
    for i, b in enumerate(blind, start=1):
        out.append(f"{'=' * 78}\nCASE {i}: {b.case_id}\nQUESTION: {b.question}\n")
        out.append(f"GROUND TRUTH:\n{b.ground_truth}\n")
        for label in sorted(b.answers):
            out.append(f"--- CANDIDATE {label} ---\n{b.answers[label].strip()}\n")
    return "\n".join(out)


__all__ = ["Blind", "render", "sample_for_validation", "score", "write_blind", "LABELS"]

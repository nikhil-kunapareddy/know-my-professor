"""Turning the dump and the rankings into the numbers that decide the arm.

Every mean is printed with the count behind it. A mean over 3 cases and a mean
over 45 are different claims, and a table that hides which is which invites the
reader to treat them alike.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from .arms import ARMS
from .cases import Case
from .generate import Sample
from .rank import RankResult

#: The pre-registered rule, fixed before any data existed. Editing these after
#: seeing results turns the experiment into a rationalisation of a preference.
RANK_TOLERANCE = 0.5
MIN_ACCEPT_RATE = 0.95
MAX_P95_LATENCY_MS = 20_000.0


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


@dataclass
class ArmReport:
    arm: str
    model: str
    effort: str | None
    mean_rank: float = 0.0
    ranked_n: int = 0
    accept_rate: float = 0.0
    accept_n: int = 0
    refusal_correct: int = 0
    refusal_total: int = 0
    p50_generate_ms: float = 0.0
    p95_generate_ms: float = 0.0
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    mean_answer_chars: float = 0.0
    cost_per_query: float = 0.0
    cost_per_1k: float = 0.0
    truncated: int = 0
    refused_by_safety: int = 0
    errors: int = 0
    mean_rank_by_stratum: dict[str, tuple[float, int]] = field(default_factory=dict)

    @property
    def passes_refusal(self) -> bool:
        return self.refusal_total > 0 and self.refusal_correct == self.refusal_total

    @property
    def passes_accept(self) -> bool:
        return self.accept_rate >= MIN_ACCEPT_RATE

    @property
    def passes_latency(self) -> bool:
        return self.p95_generate_ms < MAX_P95_LATENCY_MS


def build(
    cases: list[Case], samples: list[Sample], ranks: list[RankResult]
) -> tuple[list[ArmReport], dict]:
    """Per-arm numbers, plus the run-level diagnostics."""
    stratum_of = {c.id: c.stratum for c in cases}
    by_arm: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_arm[s.arm].append(s)

    arm_ranks: dict[str, list[int]] = defaultdict(list)
    arm_accept: dict[str, list[bool]] = defaultdict(list)
    arm_stratum_ranks: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in ranks:
        for arm, rank in r.ranks.items():
            arm_ranks[arm].append(rank)
            arm_stratum_ranks[(arm, stratum_of.get(r.case_id, "?"))].append(rank)
        for arm, ok in r.acceptable.items():
            arm_accept[arm].append(ok)

    reports = []
    for arm_name, arm in ARMS.items():
        rows = by_arm.get(arm_name, [])
        ok_rows = [s for s in rows if s.error is None]
        noans = [s for s in ok_rows if stratum_of.get(s.case_id) == "noans"]
        gen_ms = [s.latency_ms.get("generate", 0.0) for s in ok_rows]
        ranks_for_arm = arm_ranks.get(arm_name, [])
        accepts = arm_accept.get(arm_name, [])
        per_query = statistics.mean([s.cost_usd for s in ok_rows]) if ok_rows else 0.0

        reports.append(
            ArmReport(
                arm=arm_name,
                model=arm.model,
                effort=arm.effort,
                mean_rank=statistics.mean(ranks_for_arm) if ranks_for_arm else 0.0,
                ranked_n=len(ranks_for_arm),
                accept_rate=(sum(accepts) / len(accepts)) if accepts else 0.0,
                accept_n=len(accepts),
                refusal_correct=sum(1 for s in noans if s.declined),
                refusal_total=len(noans),
                p50_generate_ms=_pct(gen_ms, 0.50),
                p95_generate_ms=_pct(gen_ms, 0.95),
                mean_input_tokens=statistics.mean([s.input_tokens for s in ok_rows]) if ok_rows else 0,
                mean_output_tokens=statistics.mean([s.output_tokens for s in ok_rows]) if ok_rows else 0,
                mean_answer_chars=statistics.mean([s.answer_chars for s in ok_rows]) if ok_rows else 0,
                cost_per_query=per_query,
                cost_per_1k=per_query * 1000,
                truncated=sum(1 for s in ok_rows if s.truncated),
                refused_by_safety=sum(1 for s in ok_rows if s.refused),
                errors=sum(1 for s in rows if s.error is not None),
                mean_rank_by_stratum={
                    stratum: (statistics.mean(v), len(v))
                    for (a, stratum), v in arm_stratum_ranks.items()
                    if a == arm_name and v
                },
            )
        )

    # Counterbalance check: did reversing the order change the verdict?
    paired: dict[tuple[str, int], dict[str, dict[str, int]]] = defaultdict(dict)
    for r in ranks:
        if r.ranks:
            paired[(r.case_id, r.repeat)][r.ordering] = r.ranks
    flips = sum(
        1
        for v in paired.values()
        if len(v) == 2 and v.get("forward") != v.get("reversed")
    )
    complete = sum(1 for v in paired.values() if len(v) == 2)

    # Did every arm see the same chunks? Pairing depends on it.
    chunks_by_case: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for s in samples:
        if s.error is None and s.chunks:
            chunks_by_case[s.case_id].add(tuple(c["document_id"] for c in s.chunks))

    diagnostics = {
        "order_flips": flips,
        "order_pairs": complete,
        "order_flip_rate": (flips / complete) if complete else 0.0,
        "judge_errors": sum(1 for r in ranks if r.error),
        "cases_with_divergent_chunks": sorted(
            cid for cid, sets in chunks_by_case.items() if len(sets) > 1
        ),
        "total_cost_usd": sum(s.cost_usd for s in samples if s.error is None),
    }
    return reports, diagnostics


def decide(reports: list[ArmReport]) -> tuple[ArmReport | None, str]:
    """Apply the pre-registered rule and say why it landed where it did."""
    ranked = [r for r in reports if r.ranked_n]
    if not ranked:
        return None, "no arm was ranked; nothing to decide"

    best = min(r.mean_rank for r in ranked)
    eligible = [
        r
        for r in ranked
        if r.mean_rank <= best + RANK_TOLERANCE
        and r.passes_accept
        and r.passes_refusal
        and r.passes_latency
    ]
    if not eligible:
        failures = []
        for r in ranked:
            why = [
                name
                for name, ok in (
                    (f"rank>{best + RANK_TOLERANCE:.2f}",
                     r.mean_rank <= best + RANK_TOLERANCE),
                    (f"accept<{MIN_ACCEPT_RATE:.2f}", r.passes_accept),
                    ("refusals", r.passes_refusal),
                    ("p95", r.passes_latency),
                )
                if not ok
            ]
            failures.append(f"{r.arm} fails {'+'.join(why)}")
        return None, (
            "the rule declined to fire -- no arm cleared every gate. "
            + "; ".join(failures)
            + ". The incumbent stays; that is a result, not a failure."
        )

    winner = min(eligible, key=lambda r: (r.cost_per_query, r.mean_rank))
    return winner, (
        f"{winner.arm} is the cheapest arm within {RANK_TOLERANCE} mean rank of the "
        f"best ({best:.2f}), with accept-rate {winner.accept_rate:.2f}, "
        f"refusals {winner.refusal_correct}/{winner.refusal_total}, "
        f"p95 {winner.p95_generate_ms / 1000:.1f}s"
    )

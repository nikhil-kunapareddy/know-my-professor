"""Deleting the vectors ingest no longer produces -- and refusing when that looks wrong.

Ingest only ever upserted, so a vector outlived whatever produced it: a
professor who left, a course the catalog dropped, a profile that thinned below
the substantive filter. Structure-aware chunking makes this unavoidable rather
than occasional: a section that shrinks from three parts to two leaves
``#section@3`` behind, still answering queries with text that no longer exists.

Prune is the set difference ``ids in the namespace - ids this run produced``,
which is only safe when "this run produced" is the whole truth. So it is
opt-in (``--prune``), impossible with ``--limit``, and refused per namespace
when the deletion looks like a failed read instead of real turnover:

- the run produced nothing for the namespace at all, or
- any one section would lose more than ``max_fraction`` of its vectors (beyond
  a small allowance, so a section of twelve can lose three).

A source whose GCS read quietly came back empty trips the second rule on its own
sections even when the namespace as a whole barely moves -- which is why the
guard is per section and not per namespace. Deleted vectors are not
recoverable, only re-embeddable: the next ingest recreates anything prune got
wrong, at the cost of its embeddings.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..sources.base import PART_SEPARATOR

#: Largest fraction of any one section's vectors a prune may delete unasked.
PRUNE_MAX_FRACTION = 0.10
#: A section may always lose this many, whatever the fraction: small sections
#: churn by whole percentages every month.
PRUNE_FREE_COUNT = 25


def section_of(vector_id: str) -> str:
    """``khoury-x#biography@2`` -> ``biography``."""
    if "#" not in vector_id:
        return "(no section)"
    return vector_id.split("#", 1)[1].split(PART_SEPARATOR, 1)[0]


@dataclass(frozen=True)
class PrunePlan:
    namespace: str
    stale: tuple[str, ...]
    #: section -> (stale, existing)
    by_section: dict[str, tuple[int, int]]
    refused: str | None = None


def plan_prune(
    namespace: str,
    existing: set[str],
    produced: set[str],
    max_fraction: float = PRUNE_MAX_FRACTION,
    free_count: int = PRUNE_FREE_COUNT,
) -> PrunePlan:
    """What a prune of ``namespace`` would delete, and whether it is allowed to."""
    stale = tuple(sorted(existing - produced))
    totals = Counter(section_of(v) for v in existing)
    losses = Counter(section_of(v) for v in stale)
    by_section = {s: (losses[s], totals[s]) for s in sorted(losses)}

    refused = None
    if not produced:
        refused = "this run produced no chunks for the namespace"
    else:
        over = [
            f"{s} {lost}/{total}"
            for s, (lost, total) in by_section.items()
            if lost > max(free_count, max_fraction * total)
        ]
        if over:
            refused = (
                f"would delete more than {max_fraction:.0%} of section(s) {', '.join(over)}; "
                f"if that is intended, re-run with a higher --prune-max-fraction"
            )
    return PrunePlan(namespace, stale, by_section, refused)

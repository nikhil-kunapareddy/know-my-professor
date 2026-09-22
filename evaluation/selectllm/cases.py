"""The question set, and what counts as ground truth for each kind of question.

One ``Case`` is one question plus the answer it should have produced. The
awkward part — and the reason ``GroundTruthKind`` exists — is that "the answer
it should have produced" is not one kind of thing across this corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

CASES_PATH = Path(__file__).with_name("cases.jsonl")


class GroundTruthKind(StrEnum):
    """How a case's ground truth should be read.

    ``PROSE`` — one verifiable answer, written from corpus facts. The oracle is
    complete, so the judge needs nothing but the question, the truth and the
    answers.

    ``EXHAUSTIVE`` — a broad question where many entities qualify, so the truth
    is *every* qualifying entity, harvested offline. This exists because a
    partial reference is actively wrong here, not merely incomplete: CLAUDE.md
    records that a reference naming 3 of 30 valid people scores the other 27 as
    errors, and the topk experiment measured 46 valid people for one
    neuroscience question and 137 valid courses for one ML question. An answer
    naming 8 of 46 must score as good coverage, not as 38 misses.

    ``DECLINE`` — the corpus cannot answer it, so the only correct answer is a
    refusal. Scored deterministically rather than judged.
    """

    PROSE = "prose"
    EXHAUSTIVE = "exhaustive"
    DECLINE = "decline"


#: Stratum -> what it isolates. Topic spread is applied *within* strata: varying
#: subject matter varies content but not difficulty, and a set of 50 topically
#: diverse one-entity lookups would separate no two arms.
STRATA: dict[str, str] = {
    "lookup": "one verifiable entity; ordinary correctness",
    "broad": "many valid answers; coverage without fabrication",
    "blended": "needs both corpora in one answer",
    "trap": "false premise; fabrication resistance",
    "noans": "out of scope; refusal discipline",
}


@dataclass(frozen=True)
class Case:
    """One question and its ground truth.

    ``source_facts`` is the corpus text the ground truth was written from, kept
    so every ground truth is auditable back to the corpus rather than taken on
    trust. It is never shown to a candidate arm.
    """

    id: str
    question: str
    stratum: str
    kind: GroundTruthKind
    ground_truth: str
    expected_slugs: list[str] = field(default_factory=list)
    source_facts: str = ""
    notes: str = ""
    #: False while the set is being built, before Fable has authored the ground
    #: truth. ``load()`` defaults to requiring it, so the grid can never be run
    #: against a half-built question set — the failure it prevents is spending
    #: real money ranking answers against an empty oracle.
    allow_unwritten: bool = False

    @property
    def judged(self) -> bool:
        """Whether this case is ranked by the judge at all.

        ``noans`` cases are scored by whether the arm declined, which is
        deterministic and free — sending them to the judge would pay for an
        answer already known.
        """
        return self.kind is not GroundTruthKind.DECLINE

    def __post_init__(self) -> None:
        if self.stratum not in STRATA:
            raise ValueError(f"{self.id}: unknown stratum {self.stratum!r}")
        if not self.id.startswith(f"{self.stratum}-"):
            raise ValueError(f"{self.id}: id must start with its stratum")
        if self.kind is GroundTruthKind.EXHAUSTIVE and not self.expected_slugs:
            raise ValueError(f"{self.id}: exhaustive ground truth needs expected_slugs")
        if (
            not self.allow_unwritten
            and self.kind is not GroundTruthKind.DECLINE
            and not self.ground_truth.strip()
        ):
            raise ValueError(
                f"{self.id}: no ground truth — run `python -m evaluation.selectllm "
                "author-truth` before the grid"
            )


def load(path: Path | None = None, *, allow_unwritten: bool = False) -> list[Case]:
    """Read ``cases.jsonl``; blank lines and #-comments are ignored.

    ``allow_unwritten`` is for the two steps that build the set. Everything that
    spends money leaves it False.
    """
    lines = (path or CASES_PATH).read_text().splitlines()
    cases = [
        _case(json.loads(ln), allow_unwritten)
        for ln in lines
        if ln.strip() and not ln.startswith("#")
    ]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"duplicate case id {c.id!r}")
        seen.add(c.id)
    return cases


def _case(raw: dict, allow_unwritten: bool = False) -> Case:
    return Case(
        allow_unwritten=allow_unwritten,
        id=raw["id"],
        question=raw["question"],
        stratum=raw["stratum"],
        kind=GroundTruthKind(raw["kind"]),
        ground_truth=raw.get("ground_truth", ""),
        expected_slugs=list(raw.get("expected_slugs", [])),
        source_facts=raw.get("source_facts", ""),
        notes=raw.get("notes", ""),
    )


def by_stratum(cases: list[Case]) -> Iterator[tuple[str, list[Case]]]:
    """Cases grouped in the declared stratum order, so reports are stable."""
    for stratum in STRATA:
        group = [c for c in cases if c.stratum == stratum]
        if group:
            yield stratum, group

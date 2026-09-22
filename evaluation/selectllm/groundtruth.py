"""Writing the ground truth, with Claude Fable 5.1.

Not written by the author of this code. That author is Claude Opus 5, and two
of the four arms under test are Claude Opus 5 — ground truth phrased the way an
arm phrases things hands that arm a stylistic edge a ranking reads as quality.
Fable is not an arm, so it cannot advantage itself here.

The tradeoff this accepts knowingly: Fable later ranks answers against text it
wrote. That is weaker than a human oracle and is recorded as a limit. It is
still the better of the two available biases, because the alternative favours a
candidate.

Every ground truth is written from ``source_facts`` harvested out of the raw
corpus, never from the model's own knowledge of Northeastern.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .arms import JUDGE_EFFORT, JUDGE_MODEL
from .cases import CASES_PATH, Case, GroundTruthKind, load
from .client import Spend, ask

#: Bump on any prompt change below: it is part of the cache key.
VERSION = "gt-v1"

_SHARED = """You are writing the ground-truth answer for a retrieval-QA test set \
about Northeastern University faculty and courses.

You will be given a QUESTION and SOURCE FACTS extracted from the underlying \
corpus. Write the answer that a perfect system would give.

Absolute rules:
- Use ONLY the SOURCE FACTS. Never add anything from your own knowledge of \
Northeastern, however confident you are. If the facts do not support a detail, \
leave it out.
- Be specific: name people, titles and research areas exactly as the facts give \
them.
- Write plain prose with no citation markers, no markdown headings and no \
preamble. Do not address the reader.
- Do not hedge. This is the reference answer, not an attempt at one."""

_PROSE = _SHARED + """
- Two to five sentences."""

_EXHAUSTIVE = _SHARED + """

This is a BROAD question: many people legitimately qualify, and the facts list \
every candidate the corpus turned up by keyword search. That search is \
imperfect, so your second job is to filter it.

- Drop any candidate whose snippet does not genuinely support the question. A \
snippet matching only incidentally -- "research bridges the disciplines" for a \
question about bridge engineering, or a keyword appearing in an unrelated \
narrative -- does NOT qualify.
- Keep everyone who does genuinely qualify, even if that is thirty people.
- Write the answer as prose naming every person you kept, each with a short \
phrase on what they do. Group them if that reads better.
- Return the slugs you kept in `qualifying_slugs`, exactly as given in the facts."""

_TRAP = _SHARED + """

This question contains a FALSE PREMISE. The corpus contains no such person, \
department, lab or award.

- Write the answer as a clear rejection of the premise: state plainly that the \
corpus has no such thing.
- Do NOT name a nearest-match person as though they satisfied the question. You \
may note that no one matches, but naming a substitute is the exact failure this \
case is designed to catch.
- Two to three sentences."""


class _Prose(BaseModel):
    ground_truth: str = Field(description="The reference answer.")


class _Exhaustive(BaseModel):
    ground_truth: str = Field(description="Reference answer naming everyone kept.")
    qualifying_slugs: list[str] = Field(description="Slugs kept after filtering.")
    dropped_slugs: list[str] = Field(default_factory=list,
                                     description="Slugs rejected as false matches.")


def _system_for(case: Case) -> str:
    if case.kind is GroundTruthKind.EXHAUSTIVE:
        return _EXHAUSTIVE
    return _TRAP if case.stratum == "trap" else _PROSE


def author(spend: Spend, path: Path | None = None) -> list[dict]:
    """Fill in every empty ground truth, writing cases.jsonl back in place.

    ``noans`` cases are skipped: their correct answer is a refusal, which is
    scored deterministically rather than compared against text.
    """
    target = path or CASES_PATH
    cases = load(target, allow_unwritten=True)
    raw = {
        json.loads(ln)["id"]: json.loads(ln)
        for ln in target.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    }

    for case in cases:
        record = raw[case.id]
        if case.kind is GroundTruthKind.DECLINE:
            record["ground_truth"] = (
                "The corpus cannot answer this question. The only correct "
                "response declines and says the information is not available."
            )
            continue
        if record.get("ground_truth", "").strip():
            continue

        user = f"QUESTION\n{case.question}\n\nSOURCE FACTS\n{case.source_facts}"
        exhaustive = case.kind is GroundTruthKind.EXHAUSTIVE
        result = ask(
            _Exhaustive if exhaustive else _Prose,
            model=JUDGE_MODEL,
            system=_system_for(case),
            user=user,
            version=VERSION,
            spend=spend,
            effort=JUDGE_EFFORT,
        )
        record["ground_truth"] = result.ground_truth.strip()
        if exhaustive:
            # The filtered set replaces the raw harvest: these are the people
            # the judge will hold an answer against, so it must be the set that
            # survived inspection rather than the keyword match.
            record["expected_slugs"] = result.qualifying_slugs
            record["notes"] += (
                f"; filtered {len(result.dropped_slugs)} false matches "
                f"-> {len(result.qualifying_slugs)} qualify"
            )

    with target.open("w") as sink:
        sink.write("# Generated by build_cases.py from the raw GCS corpus. "
                   "ground_truth authored by Claude Fable 5.1 (groundtruth.py).\n")
        for case in cases:
            sink.write(json.dumps(raw[case.id]) + "\n")
    return [raw[c.id] for c in cases]

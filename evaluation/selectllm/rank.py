"""Blind, counterbalanced ranking of the arms' answers against ground truth.

One call per (case, repeat, ordering). Not per-claim faithfulness: ranking
measures "which answer is better" directly, where claim checking was a proxy
for it, and forced choice discriminates where absolute rubric scores compress
-- CLAUDE.md records ``context_utilization`` reading 0.00 on every list-style
answer, this product's main question shape.

Three guardrails, each closing a way a naive ranking measures the wrong thing:

1. **Anonymised.** Arms are relabelled A-D, so the judge cannot prefer a model
   by name.
2. **Counterbalanced.** Every case is ranked twice with the arm order reversed.
   LLM judges favour earlier options; without this, part of what is measured is
   position. Disagreement between the two orderings is reported, not averaged
   away.
3. **Absolute gate alongside the rank.** ``acceptable`` costs nothing extra and
   answers the question a ranking cannot: whether the winner is any good. A pure
   ranking names the least bad of four bad answers and the decision rule fires
   regardless.

Ties are allowed. Forcing a strict 1-2-3-4 when two answers are equivalent
manufactures a difference that is then read as signal.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from .arms import JUDGE_EFFORT, JUDGE_MODEL
from .cases import Case, GroundTruthKind
from .client import JudgeRefusal, Spend, ask
from .generate import Sample

#: Bump on any rubric change: it is part of the cache key, so an edited rubric
#: re-judges instead of returning verdicts formed under the old one.
VERSION = "rank-v1"

LABELS = "ABCDEFGH"

RUBRIC = """You are grading answers from a retrieval-augmented question \
answering system about Northeastern University faculty and courses.

You will be given a QUESTION, the GROUND TRUTH answer, and several candidate \
answers labelled by letter. Rank the candidates by how well each one matches \
the ground truth.

How to rank:
- Rank 1 is best. Ties are allowed and expected -- give two candidates the same \
rank when they are genuinely equivalent. Do not invent a difference.
- Judge against the GROUND TRUTH, not against your own knowledge of \
Northeastern. If the ground truth does not contain something, a candidate \
asserting it is wrong, not well-informed.
- A claim the ground truth does not support is a serious fault. Fabricating a \
person, title, lab or award outranks every stylistic consideration: an answer \
that invents a plausible person is worse than one that admits it cannot answer.
- Where the ground truth rejects the question's premise, or says the corpus \
cannot answer, the best candidate is the one that also declines. A candidate \
that names someone as though the premise held is the worst outcome, even if the \
person named is real.
- Where the ground truth names many qualifying people, reward coverage: more \
correct people named is better. Do not penalise a candidate for omitting some \
-- no answer can name them all. Do penalise naming someone the ground truth \
does not include.
- Length is not quality. Do not reward a longer answer for being longer, and do \
not reward confident phrasing.

Also mark each candidate `acceptable`: true if a user asking this question \
would be well served by it, false if it is wrong, fabricated, empty, or \
misses the point. This is an absolute judgement, independent of the ranking -- \
all candidates may be unacceptable, or all acceptable.

Give a one-sentence reason per candidate, naming the specific thing that \
decided it."""


class _Verdict(BaseModel):
    label: str = Field(description="The candidate's letter.")
    rank: int = Field(description="1 is best. Ties allowed.")
    acceptable: bool = Field(description="Would a user be well served by this?")
    reason: str = Field(description="One sentence naming what decided it.")


class _Ranking(BaseModel):
    verdicts: list[_Verdict] = Field(description="One entry per candidate shown.")


@dataclass
class RankResult:
    """One judged (case, repeat, ordering)."""

    case_id: str
    repeat: int
    ordering: str
    #: label -> arm, recorded so the anonymisation is auditable after the fact.
    mapping: dict[str, str] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    acceptable: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def _prompt(case: Case, answers: list[tuple[str, str]]) -> str:
    blocks = [
        f"CANDIDATE {label}\n{text.strip() or '(the system returned no answer)'}"
        for label, text in answers
    ]
    return (
        f"QUESTION\n{case.question}\n\n"
        f"GROUND TRUTH\n{case.ground_truth}\n\n" + "\n\n".join(blocks)
    )


def rank_one(
    case: Case,
    samples: dict[str, Sample],
    repeat: int,
    ordering: str,
    spend: Spend,
    seed: int = 0,
) -> RankResult:
    """Rank the arms' answers for one case, under one ordering."""
    arms = sorted(samples)
    # A per-case shuffle, then reversed for the counterbalance. Seeded so a
    # re-run reproduces the same anonymisation and hits the cache.
    rng = random.Random(f"{case.id}|{repeat}|{seed}")
    rng.shuffle(arms)
    if ordering == "reversed":
        arms = list(reversed(arms))

    mapping = {LABELS[i]: arm for i, arm in enumerate(arms)}
    answers = [(label, samples[arm].answer) for label, arm in mapping.items()]

    result = RankResult(
        case_id=case.id, repeat=repeat, ordering=ordering, mapping=mapping
    )
    try:
        parsed = ask(
            _Ranking,
            model=JUDGE_MODEL,
            system=RUBRIC,
            user=_prompt(case, answers),
            version=VERSION,
            spend=spend,
            effort=JUDGE_EFFORT,
        )
    except JudgeRefusal as exc:
        result.error = f"JudgeRefusal: {exc}"
        return result

    for verdict in parsed.verdicts:
        arm = mapping.get(verdict.label.strip().upper()[:1])
        if arm is None:
            continue
        result.ranks[arm] = verdict.rank
        result.acceptable[arm] = verdict.acceptable
        result.reasons[arm] = verdict.reason
    if set(result.ranks) != set(mapping.values()):
        missing = sorted(set(mapping.values()) - set(result.ranks))
        result.error = f"judge omitted {missing}"
    return result


def run(
    cases: list[Case],
    samples: list[Sample],
    out_path: Path,
    spend: Spend,
    orderings: tuple[str, ...] = ("forward", "reversed"),
    only: set[str] | None = None,
) -> list[RankResult]:
    """Rank every judged case, appending each verdict to ``out_path``.

    ``noans`` cases are skipped: a refusal is scored deterministically, and
    sending them here would pay for an answer already known.
    """
    grouped: dict[tuple[str, int], dict[str, Sample]] = defaultdict(dict)
    for s in samples:
        if s.error is None:
            grouped[(s.case_id, s.repeat)][s.arm] = s

    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["case_id"], r["repeat"], r["ordering"]))

    results: list[RankResult] = []
    by_id = {c.id: c for c in cases}
    with out_path.open("a") as sink:
        for (case_id, repeat), arm_samples in sorted(grouped.items()):
            case = by_id.get(case_id)
            if case is None or case.kind is GroundTruthKind.DECLINE:
                continue
            if only is not None and case_id not in only:
                continue
            if len(arm_samples) < 2:
                continue  # nothing to rank against
            for ordering in orderings:
                if (case_id, repeat, ordering) in done:
                    continue
                res = rank_one(case, arm_samples, repeat, ordering, spend)
                sink.write(json.dumps(asdict(res)) + "\n")
                sink.flush()
                results.append(res)
    return results


def load_ranks(path: Path) -> list[RankResult]:
    if not path.exists():
        return []
    return [
        RankResult(**json.loads(ln)) for ln in path.read_text().splitlines() if ln.strip()
    ]

"""Guards for the model-selection experiment.

Two of these pin a copy against its original. The copies exist because ruff's
layering rule keeps ``evaluation`` off ``preprocessing``; the tests exist because
a copy that drifts silently would put people in the ground truth who have no
vectors, and score every arm as wrong for a reason unrelated to the model.
"""

from __future__ import annotations

import pytest

from evaluation.selectllm import harvest
from evaluation.selectllm.arms import ARMS, JUDGE_MODEL, PRICING, Arm, _validate
from evaluation.selectllm.cases import GroundTruthKind, load
from preprocessing.sources.entities import college_of as real_college_of
from preprocessing.sources.profiles.source import ProfileSource


@pytest.mark.parametrize(
    "record",
    [
        {"slug": "a"},
        {"slug": "a", "college": "cos"},
        {"slug": "a", "college": "COS"},
        {"slug": "a", "college": "khoury"},
        {"slug": "a", "college": None},
        {"slug": "a", "college": "camd"},
    ],
)
def test_harvest_college_matches_the_one_id_scheme(record):
    """The local copy must agree with entities.college_of on every shape."""
    assert harvest.college_of(record) == real_college_of(record)


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"biography": "x"},
        {"research_interests": ["x"]},
        {"areas_of_interest": ["x"]},
        {"education": ["PhD"]},
        {"biography": "", "research_interests": []},
        {"name": "n", "title": "t"},
    ],
)
def test_harvest_ingestable_filter_matches_the_live_source(record):
    """Harvesting a person the pipeline never ingests makes a case unanswerable."""
    assert harvest._ingestable(record) is ProfileSource().is_ingestable(record)


def test_no_arm_is_the_judge():
    """A model cannot rank its own answers."""
    assert JUDGE_MODEL not in {arm.model for arm in ARMS.values()}


def test_every_arm_is_priced():
    assert all(arm.model in PRICING for arm in ARMS.values())


def test_an_arm_cannot_claim_an_effort_its_model_rejects(monkeypatch):
    """Claude Haiku 4.5 drops effort silently, so a mislabelled arm is the risk.

    A dropped parameter does not raise: the arm would run with no effort while
    every table in the report claimed otherwise, which is worse than a crash
    because the number looks real.
    """
    monkeypatch.setattr(
        "evaluation.selectllm.arms.ARMS",
        {"haiku45-high": Arm("claude-haiku-4-5", "high")},
    )
    with pytest.raises(ValueError, match="mislabelled"):
        _validate()


def test_cost_uses_billed_output_tokens():
    """Opus 5 at $5/$25 per MTok, on the measured 5,272-token request."""
    assert ARMS["opus5-low"].cost(5272, 800) == pytest.approx(0.04636)


def test_cases_file_is_well_formed():
    """Strata, ids and ground-truth kinds, as committed."""
    cases = load(allow_unwritten=True)
    assert len(cases) == 50

    counts: dict[str, int] = {}
    for case in cases:
        counts[case.stratum] = counts.get(case.stratum, 0) + 1
    assert counts == {"lookup": 18, "broad": 12, "blended": 10, "trap": 5, "noans": 5}

    for case in cases:
        assert case.id.startswith(f"{case.stratum}-")
        assert case.question.strip()
        # Every ground truth must be traceable to corpus text.
        assert case.source_facts.strip()

    # Broad cases carry the exhaustive harvested set, not a partial reference.
    for case in cases:
        if case.kind is GroundTruthKind.EXHAUSTIVE:
            assert len(case.expected_slugs) >= 5


def test_loading_for_a_run_refuses_unwritten_ground_truth():
    """The grid must never be ranked against an empty oracle."""
    with pytest.raises(ValueError, match="no ground truth"):
        load()

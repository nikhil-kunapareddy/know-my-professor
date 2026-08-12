"""Unit tests for the evaluation scoring (offline; no index, no models)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.harness import (
    CaseOutcome,
    EvalCase,
    EvalReport,
    load_cases,
    section_of,
    slug_of,
)

GOLDEN = Path(__file__).parent.parent / "evaluation" / "golden.jsonl"


def _case(expected=("ann",), sections=()) -> EvalCase:
    return EvalCase(id="c1", question="q?", expected_slugs=expected, expected_sections=sections)


# --- id parsing ------------------------------------------------------------


def test_slug_and_section_split():
    assert slug_of("jane-doe#biography") == "jane-doe"
    assert section_of("jane-doe#biography") == "biography"
    assert slug_of("nohash") == "nohash"
    assert section_of("nohash") == ""


# --- hit / rank ------------------------------------------------------------


def test_hit_and_rank_use_first_matching_slug():
    outcome = CaseOutcome(
        case=_case(expected=("ann",)),
        retrieved_ids=["bob#biography", "cy#biography", "ann#biography"],
    )
    assert outcome.hit
    assert outcome.rank == 3
    assert outcome.reciprocal_rank == pytest.approx(1 / 3)


def test_rank_counts_professors_not_chunks():
    """Three chunks from one professor are one position, not three."""
    outcome = CaseOutcome(
        case=_case(expected=("ann",)),
        retrieved_ids=["bob#biography", "bob#projects", "bob#education", "ann#biography"],
    )
    assert outcome.retrieved_slugs == ["bob", "ann"]
    assert outcome.rank == 2


def test_miss_scores_zero():
    outcome = CaseOutcome(case=_case(expected=("ann",)), retrieved_ids=["bob#biography"])
    assert not outcome.hit
    assert outcome.rank is None
    assert outcome.reciprocal_rank == 0.0


def test_any_expected_slug_counts_as_a_hit():
    outcome = CaseOutcome(
        case=_case(expected=("ann", "cy")),
        retrieved_ids=["cy#biography"],
    )
    assert outcome.hit


def test_empty_retrieval_is_a_miss():
    assert not CaseOutcome(case=_case(), retrieved_ids=[]).hit


# --- sections --------------------------------------------------------------


def test_section_hit_is_none_when_the_case_names_no_sections():
    assert CaseOutcome(case=_case(), retrieved_ids=["ann#biography"]).section_hit is None


def test_section_hit_checks_the_expected_section_was_retrieved():
    case = _case(sections=("areas_of_interest",))
    assert CaseOutcome(case=case, retrieved_ids=["ann#areas_of_interest"]).section_hit
    assert not CaseOutcome(case=case, retrieved_ids=["ann#biography"]).section_hit


# --- citations -------------------------------------------------------------


def test_cited_slugs_resolve_markers_against_the_chunk_list():
    outcome = CaseOutcome(
        case=_case(expected=("ann",)),
        retrieved_ids=["ann#biography", "bob#biography", "cy#biography"],
        answer="Ann [1] and Cy [3] both do this.",
    )
    assert outcome.cited_slugs == ["ann", "cy"]
    assert outcome.citation_precision == pytest.approx(0.5)


def test_citation_precision_is_one_when_every_citation_is_expected():
    outcome = CaseOutcome(
        case=_case(expected=("ann", "bob")),
        retrieved_ids=["ann#biography", "bob#biography"],
        answer="Ann [1] and Bob [2].",
    )
    assert outcome.citation_precision == 1.0


def test_citation_precision_is_none_without_an_answer():
    assert CaseOutcome(case=_case(), retrieved_ids=["ann#biography"]).citation_precision is None


def test_out_of_range_markers_are_ignored():
    """A hallucinated [9] against 2 sources must not crash or count."""
    outcome = CaseOutcome(
        case=_case(expected=("ann",)),
        retrieved_ids=["ann#biography", "bob#biography"],
        answer="See [1] and also [9].",
    )
    assert outcome.cited_slugs == ["ann"]


def test_duplicate_citations_of_one_professor_count_once():
    outcome = CaseOutcome(
        case=_case(expected=("ann",)),
        retrieved_ids=["ann#biography", "ann#projects"],
        answer="Ann [1] and again Ann [2].",
    )
    assert outcome.cited_slugs == ["ann"]
    assert outcome.citation_precision == 1.0


# --- aggregate report ------------------------------------------------------


def test_report_aggregates_recall_and_mrr():
    outcomes = [
        CaseOutcome(case=_case(expected=("ann",)), retrieved_ids=["ann#biography"]),      # rank 1
        CaseOutcome(case=_case(expected=("bob",)), retrieved_ids=["ann#b", "bob#b"]),     # rank 2
        CaseOutcome(case=_case(expected=("cy",)), retrieved_ids=["ann#b"]),               # miss
    ]
    report = EvalReport(outcomes=outcomes, top_k=8)

    assert report.recall_at_k == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert len(report.misses) == 1


def test_empty_report_does_not_divide_by_zero():
    report = EvalReport(outcomes=[], top_k=8)
    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0


def test_report_lists_misses_with_what_was_expected():
    outcomes = [CaseOutcome(case=_case(expected=("cy",)), retrieved_ids=["ann#biography"])]
    text = EvalReport(outcomes=outcomes, top_k=8).format()
    assert "Misses (1)" in text
    assert "expected: cy" in text
    assert "got:      ann" in text


# --- golden file -----------------------------------------------------------


def test_the_shipped_golden_file_parses():
    cases = load_cases(GOLDEN)
    assert cases
    assert all(c.expected_slugs for c in cases)


def test_golden_ids_are_unique(tmp_path):
    path = tmp_path / "dup.jsonl"
    path.write_text(
        '{"id": "a", "question": "q", "expected_slugs": ["x"]}\n'
        '{"id": "a", "question": "q2", "expected_slugs": ["y"]}\n'
    )
    with pytest.raises(ValueError, match="duplicate case ids"):
        load_cases(path)


def test_golden_rejects_a_case_missing_required_fields(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a", "question": "q"}\n')
    with pytest.raises(ValueError, match="expected_slugs"):
        load_cases(path)


def test_golden_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "ok.jsonl"
    path.write_text(
        "# a comment\n"
        "\n"
        '{"id": "a", "question": "q", "expected_slugs": ["x"]}\n'
    )
    assert len(load_cases(path)) == 1

"""Unit tests for the evaluation scoring (offline; no index, no models)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.harness import (
    CaseOutcome,
    EvalCase,
    EvalReport,
    load_cases,
    sample_cases,
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
    # Every case says what "right" looks like: named slugs, or an expected refusal.
    assert all(c.expected_slugs or c.expect_no_answer for c in cases)
    assert not any(c.expected_slugs and c.expect_no_answer for c in cases)


def test_the_shipped_golden_file_covers_every_section_type():
    """A section with no case is a section whose regressions are invisible."""
    from preprocessing.sources.registry import SOURCES

    registered = {spec.key for source in SOURCES for spec in source.sections}
    named = {s for c in load_cases(GOLDEN) for s in c.expected_sections}
    assert not registered - named, f"section types with no golden case: {sorted(registered - named)}"


def test_the_shipped_golden_file_has_no_answer_cases():
    """Without them, raising the relevance floor can only ever look free."""
    cases = load_cases(GOLDEN)
    negatives = [c for c in cases if c.expect_no_answer]
    assert len(negatives) >= 5, "too few no-answer cases to measure the score floor"
    assert all(not c.reference for c in negatives), "a refusal case needs no reference answer"


def test_a_no_answer_case_needs_no_slugs(tmp_path):
    path = tmp_path / "neg.jsonl"
    path.write_text('{"id": "n1", "question": "who studies alchemy?", "expect_no_answer": true}\n')
    case = load_cases(path)[0]
    assert case.expect_no_answer
    assert case.expected_slugs == ()


def test_a_case_that_says_nothing_about_correctness_is_rejected(tmp_path):
    path = tmp_path / "vague.jsonl"
    path.write_text('{"id": "v1", "question": "q?"}\n')
    with pytest.raises(ValueError, match="expected_slugs or set expect_no_answer"):
        load_cases(path)


def test_a_no_answer_case_cannot_also_expect_slugs(tmp_path):
    path = tmp_path / "both.jsonl"
    path.write_text(
        '{"id": "b1", "question": "q?", "expected_slugs": ["x"], "expect_no_answer": true}\n'
    )
    with pytest.raises(ValueError, match="cannot also expect slugs"):
        load_cases(path)


# --- scoring a no-answer case ---------------------------------------------


def _negative_case() -> EvalCase:
    return EvalCase(id="n1", question="who studies alchemy?", expect_no_answer=True)


def test_declining_a_no_answer_case_is_a_hit():
    assert CaseOutcome(case=_negative_case(), retrieved_ids=[]).hit


def test_answering_a_no_answer_case_is_a_miss():
    outcome = CaseOutcome(case=_negative_case(), retrieved_ids=["ann#biography"], answer="Ann! [1]")
    assert not outcome.hit


def test_the_no_answer_string_counts_as_declining():
    from core.pipeline import NO_ANSWER

    outcome = CaseOutcome(
        case=_negative_case(), retrieved_ids=["ann#biography"], answer=NO_ANSWER
    )
    assert outcome.hit


def test_recall_and_mrr_ignore_no_answer_cases():
    """Mixing them in would let a system that retrieves nothing score well."""
    outcomes = [
        CaseOutcome(case=_case(expected=("ann",)), retrieved_ids=["ann#biography"]),
        CaseOutcome(case=_negative_case(), retrieved_ids=[]),
    ]
    report = EvalReport(outcomes=outcomes, top_k=8, min_score=0.35)

    assert report.recall_at_k == 1.0          # one positive case, retrieved
    assert report.mrr == pytest.approx(1.0)   # not halved by the refusal
    assert report.no_answer_accuracy == 1.0
    assert report.total == 2


def test_no_answer_accuracy_is_none_without_negative_cases():
    report = EvalReport(outcomes=[CaseOutcome(case=_case())], top_k=8)
    assert report.no_answer_accuracy is None


def test_the_report_lists_a_case_that_should_have_declined():
    outcomes = [CaseOutcome(case=_negative_case(), retrieved_ids=["ann#biography"])]
    text = EvalReport(outcomes=outcomes, top_k=8, min_score=0.0).format()
    assert "Should have declined (1)" in text
    assert "retrieved: ann" in text


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


def test_a_refusal_with_a_caveat_still_counts_as_declining():
    """The model opens with the refusal, then names the nearest material it saw."""
    outcome = CaseOutcome(
        case=_negative_case(),
        retrieved_ids=["gyori#biography"],
        answer=(
            "I don't have that information in my data. None of the listed faculty work on "
            "organic chemistry; the closest is Benjamin Gyori, who works on computational "
            "systems biology [1]."
        ),
    )
    assert outcome.hit


def test_a_reworded_refusal_counts_too():
    outcome = CaseOutcome(
        case=_negative_case(),
        retrieved_ids=["ann#biography"],
        answer="I don't have information about a veterinary medicine researcher in my data.",
    )
    assert outcome.declined


def test_a_confident_wrong_answer_does_not_count_as_declining():
    outcome = CaseOutcome(
        case=_negative_case(),
        retrieved_ids=["ann#biography"],
        answer="Ann Example works on organic chemistry synthesis [1].",
    )
    assert not outcome.declined
    assert not outcome.hit


def test_the_report_says_when_refusals_were_judged_on_retrieval_alone():
    outcomes = [CaseOutcome(case=_negative_case(), retrieved_ids=["ann#biography"])]
    text = EvalReport(outcomes=outcomes, top_k=8, min_score=0.35).format()
    assert "decided by the generator" in text

    answered = [
        CaseOutcome(case=_negative_case(), retrieved_ids=["ann#b"], answer="Ann does [1].")
    ]
    assert "decided by the generator" not in EvalReport(
        outcomes=answered, top_k=8, min_score=0.35
    ).format()


def test_recall_is_not_printed_for_a_run_of_only_refusals():
    """An empty average is not a 0% score, and must not read like one."""
    outcomes = [CaseOutcome(case=_negative_case(), retrieved_ids=[])]
    text = EvalReport(outcomes=outcomes, top_k=8, min_score=0.35).format()
    assert "Recall@" not in text
    assert "MRR" not in text
    assert "Declined right:   100.0%" in text


# --- reproducible sampling -------------------------------------------------


def _cases(n: int) -> list[EvalCase]:
    return [EvalCase(id=f"c{i}", question="q?", expected_slugs=("ann",)) for i in range(n)]


def test_the_same_seed_picks_the_same_cases():
    """An experiment whose sample cannot be reproduced cannot be compared."""
    first = sample_cases(_cases(60), 10, seed=7)
    assert [c.id for c in first] == [c.id for c in sample_cases(_cases(60), 10, seed=7)]
    assert len(first) == 10


def test_a_different_seed_picks_a_different_sample():
    assert [c.id for c in sample_cases(_cases(60), 10, seed=1)] != [
        c.id for c in sample_cases(_cases(60), 10, seed=2)
    ]


def test_sampling_preserves_golden_file_order():
    picked = sample_cases(_cases(60), 10, seed=3)
    indices = [int(c.id[1:]) for c in picked]
    assert indices == sorted(indices)


def test_sampling_more_than_exists_returns_everything():
    cases = _cases(5)
    assert sample_cases(cases, 10, seed=0) == cases


def test_a_prefix_would_have_missed_the_refusals():
    """Why sampling is random: the golden file groups its cases by kind."""
    cases = load_cases(GOLDEN)
    assert not any(c.expect_no_answer for c in cases[:10]), (
        "the first ten cases contain no refusal, so --limit 10 cannot measure one"
    )
    assert any(c.expect_no_answer for c in sample_cases(cases, 20, seed=0))

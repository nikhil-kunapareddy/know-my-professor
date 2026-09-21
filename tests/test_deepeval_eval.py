"""Tests for the DeepEval evaluation framework (offline; no judge, no index).

Split the way the package is: the parts that decide *what* gets scored are pure
and always tested; the parts that talk to DeepEval are tested only when the
`eval` extra is installed (``pytest.importorskip``), so CI stays green without
it.

The DeepEval-dependent tests are deliberately narrow. They do not check that
faithfulness returns a sensible number — that is DeepEval's job. They check the
places this repo could silently break: that every metric still takes the inputs
the table says it takes, that the judge is still built the way the Anthropic API
requires, and that a cache hit is really a hit.
"""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from core.pipeline import NO_ANSWER, RAGResult
from core.retrieval.base import RetrievalResult
from evaluation.deepeval_eval import metrics as metrics_module
from evaluation.deepeval_eval import report as report_module
from evaluation.deepeval_eval.metrics import (
    END_TO_END,
    GENERATION,
    RETRIEVAL,
    SPECS,
    STAGE_ORDER,
    required_fields,
    specs_for,
)
from evaluation.deepeval_eval.report import JudgeReport, MetricScore
from evaluation.deepeval_eval.runner import score
from evaluation.deepeval_eval.samples import (
    JUDGED_FIELDS,
    JudgedSample,
    SampleSet,
    context_texts,
    read_samples,
    sample_from_result,
    sample_from_retrieval,
    write_samples,
)
from evaluation.harness import EvalCase

#: The runner is handed the sample itself instead of an LLMTestCase, so these
#: tests exercise it without the SDK. See runner.TestCaseBuilder.
AS_IS = lambda sample: sample  # noqa: E731


def _case(reference: str = "Ann works on cryptography.") -> EvalCase:
    return EvalCase(id="c1", question="who works on crypto?", expected_slugs=("ann",),
                    reference=reference)


def _result(document_id: str = "ann#biography", score_value: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        document_id=document_id,
        score=score_value,
        metadata={
            "professor_name": "Ann Example",
            "professor_title": "Professor",
            "section_type": "biography",
            "text": "Ann works on cryptography.",
        },
    )


# --- samples ---------------------------------------------------------------


def test_contexts_are_rendered_exactly_as_the_generator_sees_them():
    """Judging a different rendering than the prompt used measures nothing."""
    from core.llm.prompts import PromptBuilder

    results = [_result(), _result("bob#projects")]
    assert context_texts(results) == [
        PromptBuilder.context_block(1, results[0]),
        PromptBuilder.context_block(2, results[1]),
    ]


def test_retrieval_sample_carries_context_and_reference_but_no_answer():
    sample = sample_from_retrieval(_case(), [_result()])
    assert sample.retrieved_ids == ("ann#biography",)
    assert sample.expected_output
    assert sample.actual_output == ""
    assert sample.available() == {"input", "retrieval_context", "expected_output"}


def test_pipeline_sample_records_the_answer_and_the_sources_it_used():
    result = RAGResult(answer="Ann does [1].", sources=[_result()], retrieved=1)
    sample = sample_from_result(_case(), result)
    assert sample.actual_output == "Ann does [1]."
    assert not sample.no_answer
    assert "actual_output" in sample.available()


def test_a_declined_answer_is_flagged_not_judged():
    """The no-answer string is correct behaviour, not a response to score."""
    sample = sample_from_result(_case(), RAGResult(answer=NO_ANSWER, retrieved=3))
    assert sample.no_answer
    assert sample.actual_output == ""
    assert "actual_output" not in sample.available()


def test_available_treats_empty_as_absent():
    sample = JudgedSample(case_id="c1", input="q?", retrieval_context=(), expected_output="")
    assert sample.available() == {"input"}


def test_the_sample_fields_are_the_fields_metrics_ask_for():
    """JUDGED_FIELDS is what available() can return; a drift here breaks skips."""
    sample = JudgedSample(
        case_id="c1", input="q", retrieval_context=("c",), actual_output="a", expected_output="r"
    )
    assert set(JUDGED_FIELDS) == set(sample.inputs())
    assert sample.available() == set(JUDGED_FIELDS)


def test_dump_roundtrips_samples_and_the_run_settings(tmp_path):
    original = SampleSet(
        samples=[sample_from_result(_case(), RAGResult(answer="a [1].", sources=[_result()]))],
        index_name="know-my-professor-m1024",
        top_k=8,
        min_score=0.35,
        chat_model="claude-opus-5",
    )
    path = tmp_path / "runs" / "run.jsonl"
    write_samples(path, original)

    loaded = read_samples(path)
    assert loaded.index_name == "know-my-professor-m1024"
    assert (loaded.top_k, loaded.min_score) == (8, 0.35)
    assert loaded.chat_model == "claude-opus-5"
    assert [s.to_dict() for s in loaded.samples] == [s.to_dict() for s in original.samples]


def test_a_dump_written_before_the_deepeval_port_still_reads(tmp_path):
    """--from-dump exists to re-judge recorded runs; the stored ones predate this."""
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps({"_run": {"index_name": "idx", "top_k": 8, "min_score": 0.35}}) + "\n"
        + json.dumps({
            "case_id": "c1",
            "user_input": "who works on crypto?",
            "retrieved_contexts": ["[1] ctx"],
            "response": "Ann does [1].",
            "reference": "Ann works on cryptography.",
            "retrieved_ids": ["ann#biography"],
        }) + "\n"
    )
    sample = read_samples(path).samples[0]
    assert sample.input == "who works on crypto?"
    assert sample.retrieval_context == ("[1] ctx",)
    assert sample.actual_output == "Ann does [1]."
    assert sample.expected_output == "Ann works on cryptography."


def test_the_on_disk_format_is_unchanged_by_the_port():
    """Writing new names would strand every dump stored under evaluation/results."""
    sample = JudgedSample(
        case_id="c1", input="q", retrieval_context=("c",), actual_output="a", expected_output="r"
    )
    written = sample.to_dict()
    assert {"user_input", "retrieved_contexts", "response", "reference"} <= set(written)
    assert JudgedSample.from_dict(written) == sample


def test_reading_a_dump_with_no_samples_fails(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text('{"_run": {"top_k": 8}}\n')
    with pytest.raises(ValueError, match="no samples"):
        read_samples(path)


def test_reading_a_malformed_dump_names_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"case_id": "c1", "user_input": "q"}\n{"user_input": "no id"}\n')
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        read_samples(path)


# --- the metric table ------------------------------------------------------


def test_every_stage_has_metrics_and_every_metric_has_a_known_stage():
    for stage in STAGE_ORDER:
        assert specs_for((stage,)), f"{stage} has no metrics"
    assert {spec.stage for spec in SPECS.values()} <= set(STAGE_ORDER)


def test_specs_come_back_in_stage_order():
    stages = [spec.stage for spec in specs_for(STAGE_ORDER)]
    assert stages == sorted(stages, key=STAGE_ORDER.index)


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="unknown stage"):
        specs_for(("retreival",))


def test_retrieval_stage_never_needs_an_answer():
    """Otherwise --stage retrieval would quietly start paying for generation."""
    assert not any(spec.needs_response for spec in specs_for((RETRIEVAL,)))


def test_generation_stage_always_needs_an_answer():
    assert all(spec.needs_response for spec in specs_for((GENERATION,)))


def test_end_to_end_stage_always_needs_a_reference():
    assert all(spec.needs_reference for spec in specs_for((END_TO_END,)))


def test_duplicate_metric_keys_are_rejected():
    from dataclasses import replace

    spec = SPECS["faithfulness"]
    original = metrics_module._SPECS
    try:
        metrics_module._SPECS = (*original, replace(spec))
        with pytest.raises(ValueError, match="duplicate metric keys"):
            metrics_module._validate()
    finally:
        metrics_module._SPECS = original


def test_required_fields_reads_the_declared_parameters():
    class Fake:
        _required_params = [
            type("P", (), {"value": "input"})(),
            type("P", (), {"value": "actual_output"})(),
        ]

    assert required_fields(Fake()) == {"input", "actual_output"}


def test_required_fields_prefers_evaluation_params_over_an_annotation():
    """GEval declares _required_params as a bare annotation and never assigns it.

    Reading that annotation yields a typing object, which used to blow up the
    skip logic with 'expected str instance, _UnpackGenericAlias found'. Its real
    inputs are the evaluation_params it was constructed with.
    """
    class FakeGEval:
        # An annotation with no assignment, exactly as DeepEval declares it.
        _required_params: list
        evaluation_params = [type("P", (), {"value": "input"})()]

    assert required_fields(FakeGEval()) == {"input"}


def test_a_metric_that_declares_nothing_readable_fails_loudly():
    with pytest.raises(TypeError, match="no readable required parameters"):
        required_fields(object())


# --- runner ----------------------------------------------------------------


class _FakeMetric:
    """Stands in for a DeepEval metric: same surface, no judge.

    DeepEval metrics report their result on the instance rather than through the
    return value, so the fake does too — that is the shape the runner has to
    cope with, and the reason it builds one metric per pair.
    """

    def __init__(self, value=1.0, raises: Exception | None = None, error: str | None = None,
                 skipped: bool = False, needs=("input", "actual_output", "retrieval_context")):
        self.value = value
        self.raises = raises
        self.error = error
        self.skipped = skipped
        self.calls: list = []
        self._required_params = [type("P", (), {"value": name})() for name in needs]
        self.score = None
        self.reason = None

    async def a_measure(self, test_case, *_, **__):
        self.calls.append(test_case)
        if self.raises:
            raise self.raises
        self.score = self.value
        self.reason = "because"
        return self.value


def _factories(*metrics):
    """Pair fake metrics with specs, newest-style factories over fixed instances."""
    return [(spec, (lambda m=metric: m)) for spec, metric in metrics]


def _full_sample(case_id: str = "c1") -> JudgedSample:
    return JudgedSample(
        case_id=case_id,
        input="q?",
        retrieval_context=("[1] ctx",),
        actual_output="an answer [1]",
        expected_output="the truth",
    )


def test_a_scored_case_records_the_value_and_the_judges_reason():
    metric = _FakeMetric(value=0.75)
    report = score([_full_sample()], _factories((SPECS["faithfulness"], metric)),
                   build_test_case=AS_IS)

    assert [(s.status, s.value) for s in report.scores] == [("ok", 0.75)]
    assert report.scores[0].detail == "because"
    assert metric.calls[0].retrieval_context == ("[1] ctx",)


def test_each_pair_gets_its_own_metric_instance():
    """A shared instance would hand back another sample's score and reason."""
    built: list[_FakeMetric] = []

    def factory():
        built.append(_FakeMetric())
        return built[-1]

    score([_full_sample("c1"), _full_sample("c2")],
          [(SPECS["faithfulness"], factory)], build_test_case=AS_IS)
    assert len(built) == 2
    assert built[0] is not built[1]


def test_a_metric_that_cannot_be_built_is_recorded_not_raised():
    def factory():
        raise RuntimeError("no judge configured")

    report = score([_full_sample()], [(SPECS["faithfulness"], factory)], build_test_case=AS_IS)
    assert report.scores[0].status == "error"
    assert "no judge configured" in report.scores[0].detail


def test_a_missing_input_skips_rather_than_scoring_zero():
    """A zero here would read as a regression no code change could fix."""
    sample = JudgedSample(case_id="c1", input="q?", retrieval_context=("[1] ctx",))
    metric = _FakeMetric()
    report = score([sample], _factories((SPECS["faithfulness"], metric)), build_test_case=AS_IS)

    assert report.scores[0].status == "skipped"
    assert report.scores[0].value is None
    assert not metric.calls, "a metric must not be called without its inputs"


def test_skip_reasons_name_the_input_that_was_missing():
    no_reference = JudgedSample(
        case_id="c1", input="q?", retrieval_context=("[1] ctx",), actual_output="a"
    )
    declined = JudgedSample(
        case_id="c2", input="q?", retrieval_context=("[1] ctx",),
        expected_output="t", no_answer=True,
    )
    nothing_retrieved = JudgedSample(
        case_id="c3", input="q?", actual_output="a", expected_output="t"
    )

    needs_everything = _FakeMetric(
        needs=("input", "actual_output", "retrieval_context", "expected_output")
    )
    report = score(
        [no_reference, declined, nothing_retrieved],
        _factories((SPECS["answer_correctness"], needs_everything)),
        build_test_case=AS_IS,
    )
    details = {s.case_id: s.detail for s in report.scores}
    assert details["c1"] == "no reference in the golden case"
    assert details["c2"] == "pipeline declined to answer"
    assert details["c3"] == "nothing retrieved above the score floor"


def test_a_judge_failure_is_recorded_not_raised():
    """One bad call must not discard a run that cost minutes and quota."""
    report = score(
        [_full_sample()],
        _factories(
            (SPECS["faithfulness"], _FakeMetric(raises=RuntimeError("upstream 529"))),
            (SPECS["answer_relevancy"], _FakeMetric(value=1.0)),
        ),
        build_test_case=AS_IS,
    )

    by_metric = {s.metric: s for s in report.scores}
    assert by_metric["faithfulness"].status == "error"
    assert "upstream 529" in by_metric["faithfulness"].detail
    assert by_metric["answer_relevancy"].status == "ok"


def test_an_error_reported_without_raising_is_still_an_error():
    """DeepEval's own ignore_errors path returns normally with .error set."""
    metric = _FakeMetric(error="judge refused")
    report = score([_full_sample()], _factories((SPECS["faithfulness"], metric)),
                   build_test_case=AS_IS)
    assert report.scores[0].status == "error"
    assert "judge refused" in report.scores[0].detail


def test_a_metric_that_skips_itself_is_recorded_as_a_skip():
    metric = _FakeMetric(skipped=True)
    report = score([_full_sample()], _factories((SPECS["faithfulness"], metric)),
                   build_test_case=AS_IS)
    assert report.scores[0].status == "skipped"


def test_a_metric_that_returns_no_score_is_an_error():
    metric = _FakeMetric(value=None)
    report = score([_full_sample()], _factories((SPECS["faithfulness"], metric)),
                   build_test_case=AS_IS)
    assert report.scores[0].status == "error"
    assert "no score" in report.scores[0].detail


def test_nan_counts_as_a_failure_not_a_score():
    report = score([_full_sample()], _factories((SPECS["faithfulness"], _FakeMetric(math.nan))),
                   build_test_case=AS_IS)
    assert report.scores[0].status == "error"
    assert "NaN" in report.scores[0].detail
    assert report.summary("faithfulness").mean is None


def test_every_sample_is_scored_by_every_metric_and_progress_is_reported():
    seen: list[tuple[str, str, str]] = []
    samples = [_full_sample("c1"), _full_sample("c2")]
    metrics = _factories(
        (SPECS["faithfulness"], _FakeMetric()), (SPECS["answer_relevancy"], _FakeMetric())
    )

    report = score(samples, metrics, concurrency=2, progress=lambda *a: seen.append(a),
                   build_test_case=AS_IS)

    assert len(report.scores) == 4
    assert len(seen) == 4
    assert report.cases == 2


def test_scoring_no_samples_is_not_an_error():
    assert score([], _factories((SPECS["faithfulness"], _FakeMetric())),
                 build_test_case=AS_IS).scores == []


# --- report ----------------------------------------------------------------


def test_the_mean_is_over_scored_cases_only():
    report = JudgeReport(
        scores=[
            MetricScore("c1", "faithfulness", 1.0),
            MetricScore("c2", "faithfulness", 0.5),
            MetricScore("c3", "faithfulness", status="skipped", detail="no reference"),
            MetricScore("c4", "faithfulness", status="error", detail="boom"),
        ],
        cases=4,
    )
    summary = report.summary("faithfulness")
    assert summary.mean == pytest.approx(0.75)
    assert (summary.scored, summary.skipped, summary.errors) == (2, 1, 1)
    assert summary.worst == ("c2", 0.5)


def test_an_unmeasured_metric_has_no_mean():
    report = JudgeReport(scores=[MetricScore("c1", "context_recall", status="skipped")], cases=1)
    assert report.summary("context_recall").mean is None


def test_the_worst_case_of_an_inverted_metric_is_the_highest_one(monkeypatch):
    """LOWER_IS_BETTER is empty today; the handling still has to be right."""
    monkeypatch.setattr(report_module, "LOWER_IS_BETTER", frozenset({"faithfulness"}))
    report = JudgeReport(
        scores=[MetricScore("c1", "faithfulness", 0.1), MetricScore("c2", "faithfulness", 0.9)],
        cases=2,
    )
    assert report.summary("faithfulness").worst == ("c2", 0.9)


def test_gates_compare_in_the_right_direction(monkeypatch):
    monkeypatch.setattr(report_module, "LOWER_IS_BETTER", frozenset({"context_relevance"}))
    report = JudgeReport(
        scores=[
            MetricScore("c1", "faithfulness", 0.9),
            MetricScore("c1", "context_relevance", 0.4),
        ],
        cases=1,
    )
    assert report.failing({"faithfulness": 0.8, "context_relevance": 0.5}) == []
    problems = report.failing({"faithfulness": 0.95, "context_relevance": 0.2})
    assert len(problems) == 2
    assert "below the required" in problems[0]
    assert "above the allowed" in problems[1]


def test_a_gate_on_an_unmeasured_metric_fails():
    """A gate that passes because nothing was measured is worse than a red build."""
    report = JudgeReport(scores=[MetricScore("c1", "context_recall", status="skipped")], cases=1)
    assert report.failing({"context_recall": 0.5}) == [
        "context_recall: not measured (no case could be scored)"
    ]


def test_metrics_are_listed_in_stage_order():
    report = JudgeReport(
        scores=[
            MetricScore("c1", "answer_correctness", 1.0),
            MetricScore("c1", "faithfulness", 1.0),
            MetricScore("c1", "context_recall", 1.0),
        ],
        cases=1,
    )
    assert report.metrics == ["context_recall", "faithfulness", "answer_correctness"]


def test_the_report_shows_denominators_skips_and_errors():
    report = JudgeReport(
        scores=[
            MetricScore("c1", "faithfulness", 0.5),
            MetricScore("c2", "faithfulness", status="skipped", detail="no reference"),
            MetricScore("c1", "answer_correctness", 0.6),
            MetricScore("c1", "context_recall", status="error", detail="SDKError: 401"),
        ],
        cases=2,
    )
    text = report.format()
    assert "RETRIEVAL" in text and "GENERATION" in text and "END TO END" in text
    assert "no reference" in text
    assert "SDKError: 401" in text
    # A weak number is explained, so the reader knows what to do about it.
    assert "hallucination" in text


def test_an_inverted_metric_is_labelled_in_the_table(monkeypatch):
    monkeypatch.setattr(report_module, "LOWER_IS_BETTER", frozenset({"faithfulness"}))
    report = JudgeReport(scores=[MetricScore("c1", "faithfulness", 0.2)], cases=1)
    assert "lower is better" in report.format()


def test_verbose_lists_every_case():
    report = JudgeReport(scores=[MetricScore("c1", "faithfulness", 1.0)], cases=1)
    assert "Per case:" in report.format(verbose=True)
    assert "Per case:" not in report.format()


def test_the_json_report_carries_metadata_summaries_and_every_score():
    """The console table is gone in a month; this is what a comparison reads."""
    report = JudgeReport(
        scores=[
            MetricScore("c1", "faithfulness", 1.0, detail="all statements supported"),
            MetricScore("c2", "faithfulness", status="skipped", detail="no reference"),
            MetricScore("c1", "answer_correctness", 0.25),
        ],
        cases=2,
    )
    data = report.to_dict({"seed": 7, "stages": ["generation"]})

    assert data["meta"] == {"seed": 7, "stages": ["generation"]}
    assert data["cases"] == 2
    assert data["metrics"]["faithfulness"]["mean"] == 1.0
    assert data["metrics"]["faithfulness"]["skipped"] == 1
    assert data["metrics"]["faithfulness"]["stage"] == "generation"
    assert data["metrics"]["answer_correctness"]["stage"] == "end_to_end"
    assert len(data["scores"]) == 3
    assert data["scores"][0]["detail"] == "all statements supported"


def test_the_json_report_survives_a_roundtrip():
    report = JudgeReport(scores=[MetricScore("c1", "faithfulness", 0.5)], cases=1)
    assert json.loads(json.dumps(report.to_dict()))["metrics"]["faithfulness"]["mean"] == 0.5


# --- DeepEval-facing: skipped unless the `eval` extra is installed ---------


def _stub_judge():
    """A judge that fails if anything actually calls it."""
    from deepeval.models.base_model import DeepEvalBaseLLM

    class _Judge(DeepEvalBaseLLM):
        def load_model(self):
            return None

        def generate(self, prompt, schema=None):  # pragma: no cover - never called
            raise AssertionError("the table check must not call the judge")

        async def a_generate(self, prompt, schema=None):  # pragma: no cover
            raise AssertionError("the table check must not call the judge")

        def get_model_name(self):
            return "stub"

    return _Judge()


class _RecordingEmbedder:
    model = "mistral-embed-2312"
    dim = 4

    def __init__(self):
        self.batches: list[list[str]] = []

    def embed_texts(self, texts):
        self.batches.append(list(texts))
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return self.embed_texts([text])[0]


def test_the_metric_table_matches_the_inputs_deepeval_actually_asks_for():
    """The drift guard: an upgrade that changes a metric's inputs fails here."""
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.metrics import build_metric

    for spec in SPECS.values():
        metric = build_metric(spec, llm=_stub_judge(), embedder=_RecordingEmbedder())
        fields = required_fields(metric)
        assert fields <= set(JUDGED_FIELDS), (
            f"{spec.key} asks for an input samples do not carry: {fields}"
        )
        assert spec.needs_response == ("actual_output" in fields), spec.key
        assert spec.needs_reference == ("expected_output" in fields), spec.key


def test_every_metric_is_built_with_its_own_concurrency_off():
    """The runner's semaphore is the rate limit; internal fan-out would evade it."""
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.metrics import build_metric

    for spec in SPECS.values():
        metric = build_metric(spec, llm=_stub_judge(), embedder=_RecordingEmbedder())
        assert metric.async_mode is False, spec.key


def test_a_metric_is_refused_the_components_it_needs_are_missing():
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.metrics import build_metric

    with pytest.raises(ValueError, match="needs an embedder"):
        build_metric(SPECS["semantic_similarity"], llm=_stub_judge())


def test_a_judged_metric_without_a_judge_is_refused_not_defaulted_to_openai():
    """DeepEval silently resolves a missing model to OpenAI, which has no key here."""
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.metrics import build_metric

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_metric(SPECS["faithfulness"], llm=None)


def test_answer_correctness_ships_its_own_evaluation_steps():
    """G-Eval otherwise invents them per run, which is not a fixed yardstick."""
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.metrics import build_metric

    metric = build_metric(SPECS["answer_correctness"], llm=_stub_judge())
    assert metric.evaluation_steps, "G-Eval would generate its own steps"
    assert any("not contradicted" in step for step in metric.evaluation_steps), (
        "the step that stops broad questions scoring their extra correct people as errors"
    )


def test_the_judge_never_sends_sampling_parameters():
    """Anthropic's Messages.create() rejects temperature outright."""
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.judge import assert_no_temperature

    class _Judge:
        temperature = None

    assert assert_no_temperature(_Judge()) is not None

    class _Configured:
        temperature = 0.0

    with pytest.raises(ValueError, match="temperature"):
        assert_no_temperature(_Configured())


def test_an_unsupported_judge_provider_is_refused():
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.judge import build_judge

    with pytest.raises(ValueError, match="unknown judge provider"):
        build_judge(provider="llama")


def test_the_judge_embedder_is_paced_for_the_free_tier(monkeypatch):
    """Mistral allows 60 requests/minute, and every case wants one."""
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval import judge as judge_module
    from shared.config import EMBED_RATE_LIMIT_SLEEP_SECONDS

    seen: dict = {}

    def fake_build_embedder(*args, **kwargs):
        seen.update(kwargs)
        return _RecordingEmbedder()

    monkeypatch.setattr(judge_module, "build_embedder", fake_build_embedder)
    judge_module.build_judge_embedder()
    assert seen["pace_seconds"] == EMBED_RATE_LIMIT_SLEEP_SECONDS
    assert EMBED_RATE_LIMIT_SLEEP_SECONDS >= 1.0


def test_semantic_similarity_measures_cosine_in_the_corpus_space():
    pytest.importorskip("deepeval")
    from deepeval.test_case import LLMTestCase

    from evaluation.deepeval_eval.similarity import SemanticSimilarityMetric, cosine

    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # A negative cosine is clamped: every other column in the report is [0, 1].
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="widths differ"):
        cosine([1.0], [1.0, 0.0])

    embedder = _RecordingEmbedder()
    metric = SemanticSimilarityMetric(embedder)
    metric.measure(LLMTestCase(input="q", actual_output="a", expected_output="b"))
    assert metric.score == pytest.approx(1.0)
    # One batched request for the pair, not one request each.
    assert embedder.batches == [["a", "b"]]


def test_concurrent_embedding_is_serialized_across_metric_instances():
    """One Mistral client driven from several threads at once is flaky.

    The runner builds a metric per pair, so a per-instance lock would serialise
    nothing; this pins that the lock is shared.
    """
    pytest.importorskip("deepeval")
    import time

    from deepeval.test_case import LLMTestCase

    from evaluation.deepeval_eval.similarity import SemanticSimilarityMetric

    class _Overlapping:
        model = "mistral-embed-2312"
        dim = 2

        def __init__(self):
            self.in_flight = 0
            self.max_in_flight = 0

        def embed_texts(self, texts):
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            time.sleep(0.01)  # long enough for a second caller to overlap
            self.in_flight -= 1
            return [[1.0, 0.0] for _ in texts]

    embedder = _Overlapping()
    case = LLMTestCase(input="q", actual_output="a", expected_output="b")

    async def hammer():
        await asyncio.gather(
            *(SemanticSimilarityMetric(embedder).a_measure(case) for _ in range(5))
        )

    asyncio.run(hammer())
    assert embedder.max_in_flight == 1


# --- the judge cache -------------------------------------------------------


class _CountingJudge:
    """Counts calls so a cache hit is distinguishable from a cheap miss."""

    def __init__(self):
        self.calls = 0

    def get_model_name(self):
        return "counting"

    def generate(self, prompt, schema=None, *args, **kwargs):
        self.calls += 1
        return (schema(verdict="yes") if schema else "plain"), 0.5

    async def a_generate(self, prompt, schema=None, *args, **kwargs):
        return self.generate(prompt, schema, *args, **kwargs)


def _schema():
    from pydantic import BaseModel

    class Verdict(BaseModel):
        verdict: str

    return Verdict


def test_a_repeated_judge_call_is_served_from_disk(tmp_path):
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.judge import CachingJudge

    inner = _CountingJudge()
    judge = CachingJudge(inner, tmp_path / "cache")

    first, first_cost = judge.generate("prompt", _schema())
    second, second_cost = judge.generate("prompt", _schema())

    assert inner.calls == 1, "the second call should not have reached the judge"
    assert first.verdict == second.verdict == "yes"
    assert (first_cost, second_cost) == (0.5, 0.0), "a cache hit costs nothing"
    assert (judge.hits, judge.misses) == (1, 1)


def test_a_different_prompt_or_model_is_a_different_entry(tmp_path):
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.judge import CachingJudge

    inner = _CountingJudge()
    judge = CachingJudge(inner, tmp_path / "cache")
    judge.generate("prompt a", _schema())
    judge.generate("prompt b", _schema())
    assert inner.calls == 2, "a changed prompt must re-judge"


def test_plain_text_verdicts_cache_too(tmp_path):
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.judge import CachingJudge

    inner = _CountingJudge()
    judge = CachingJudge(inner, tmp_path / "cache")
    assert judge.generate("p")[0] == "plain"
    assert judge.generate("p")[0] == "plain"
    assert inner.calls == 1


def test_the_async_path_shares_the_same_cache(tmp_path):
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.judge import CachingJudge

    inner = _CountingJudge()
    judge = CachingJudge(inner, tmp_path / "cache")
    judge.generate("p", _schema())
    result, cost = asyncio.run(judge.a_generate("p", _schema()))
    assert inner.calls == 1
    assert (result.verdict, cost) == ("yes", 0.0)


def test_an_unreadable_cache_entry_is_a_miss_not_a_crash(tmp_path):
    """An interrupted run must not be able to fail the next one."""
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.judge import CachingJudge

    inner = _CountingJudge()
    judge = CachingJudge(inner, tmp_path / "cache")
    judge.generate("p", _schema())
    for entry in (tmp_path / "cache").glob("*.json"):
        entry.write_text("{ truncated")

    result, _ = judge.generate("p", _schema())
    assert result.verdict == "yes"
    assert inner.calls == 2


def test_the_cache_sees_through_to_the_wrapped_judges_temperature(tmp_path):
    """assert_no_temperature must not be fooled by the wrapper."""
    pytest.importorskip("deepeval")
    from evaluation.deepeval_eval.judge import CachingJudge, assert_no_temperature

    inner = _CountingJudge()
    inner.temperature = 0.0
    with pytest.raises(ValueError, match="temperature"):
        assert_no_temperature(CachingJudge(inner, tmp_path / "cache"))


# --- plots and the CLI -----------------------------------------------------


def test_the_plots_render_from_a_stored_report(tmp_path):
    """Smoke test: the figures must survive a report with skips and a missing metric."""
    pytest.importorskip("matplotlib")
    from evaluation import plots

    report = JudgeReport(
        scores=[
            MetricScore("c1", "context_relevance", 1.0),
            MetricScore("c1", "faithfulness", 0.5),
            MetricScore("c1", "answer_correctness", 0.0),
            MetricScore("c1", "semantic_similarity", 0.4),
            MetricScore("c2", "context_relevance", 0.9),
            MetricScore("c2", "faithfulness", 1.0),
            MetricScore("c2", "answer_correctness", 1.0),
            MetricScore("c2", "semantic_similarity", status="skipped", detail="no reference"),
        ],
        cases=2,
    ).to_dict({"seed": 1})

    assert plots.heatmap(report, tmp_path / "h.png").exists()
    assert plots.metric_means(report, tmp_path / "m.png").exists()
    assert plots.paired_metrics(
        report, "answer_correctness", "faithfulness", tmp_path / "p.png", "t", "s"
    ).exists()


def test_the_score_floor_plot_renders(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluation import plots

    data = {
        "answerable": [{"case_id": "c1", "top_score": 0.8, "first_correct_score": 0.8}],
        "out_of_scope": [{"case_id": "n1", "top_score": 0.75}],
        "meta": {"embed_model": "mistral-embed-2312", "index_name": "idx"},
    }
    assert plots.score_floor(data, tmp_path / "f.png").exists()


def test_a_sample_records_the_pipeline_stage_timings():
    """Model selection is never only a quality question."""
    result = RAGResult(
        answer="Ann does [1].",
        sources=[_result()],
        retrieved=1,
        timings_ms={"embed": 120.5, "retrieve": 40.0, "generate": 2300.0},
    )
    sample = sample_from_result(_case(), result)
    assert sample.timings_ms["generate"] == 2300.0


def test_timings_survive_a_dump_roundtrip(tmp_path):
    original = SampleSet(
        samples=[sample_from_result(
            _case(),
            RAGResult(answer="a [1].", sources=[_result()],
                      timings_ms={"embed": 1.0, "generate": 900.0}),
        )],
        chat_model="claude-sonnet-5",
    )
    path = tmp_path / "run.jsonl"
    write_samples(path, original)
    loaded = read_samples(path)
    assert loaded.samples[0].timings_ms == {"embed": 1.0, "generate": 900.0}
    assert loaded.chat_model == "claude-sonnet-5"


def _fake_run(tmp_path, name: str, faithfulness: float, generate_ms: float) -> None:
    """A minimal finished run on disk, as the comparison plots expect to find it."""
    directory = tmp_path / name
    directory.mkdir()
    report = JudgeReport(
        scores=[MetricScore("c1", "faithfulness", faithfulness),
                MetricScore("c1", "semantic_similarity", 0.2)],
        cases=1,
    ).to_dict({"chat_model": name})
    (directory / "report.json").write_text(json.dumps(report))
    (directory / "samples.jsonl").write_text(
        json.dumps({"_run": {"chat_model": name}}) + "\n"
        + json.dumps({"case_id": "c1", "user_input": "q?", "response": "a",
                      "timings_ms": {"embed": 100.0, "generate": generate_ms}}) + "\n"
    )


def test_the_comparison_plots_read_several_runs(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluation import plots

    _fake_run(tmp_path, "model-a", 0.9, 2500.0)
    _fake_run(tmp_path, "model-b", 0.8, 900.0)

    runs = plots.load_runs(tmp_path)
    assert [r["label"] for r in runs] == ["model-a", "model-b"]
    assert plots._generate_ms(runs[1]) == [900.0]

    assert plots.model_quality(runs, tmp_path / "q.png").exists()
    assert plots.model_latency(runs, tmp_path / "l.png").exists()
    assert plots.quality_vs_latency(runs, tmp_path / "s.png").exists()


def test_load_runs_ignores_a_directory_with_no_report(tmp_path):
    pytest.importorskip("matplotlib")
    from evaluation import plots

    _fake_run(tmp_path, "finished", 0.9, 1000.0)
    (tmp_path / "half-done").mkdir()
    (tmp_path / "plots").mkdir()
    assert [r["label"] for r in plots.load_runs(tmp_path)] == ["finished"]


def test_the_median_helper_handles_both_parities_and_emptiness():
    from evaluation.run_deepeval import _median

    assert _median([]) is None
    assert _median([5.0]) == 5.0
    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median([4.0, 1.0, 3.0, 2.0]) == 2.5


def test_unknown_metrics_are_rejected_by_the_gate_parser():
    from evaluation.run_deepeval import _parse_minimum

    assert _parse_minimum("faithfulness=0.8") == ("faithfulness", 0.8)
    with pytest.raises(Exception, match="unknown metric"):
        _parse_minimum("noise_sensitivity=0.3")

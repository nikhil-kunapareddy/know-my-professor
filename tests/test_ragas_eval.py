"""Tests for the Ragas evaluation framework (offline; no judge, no index).

Split the way the package is: the parts that decide *what* gets scored are pure
and always tested; the parts that talk to Ragas are tested only when the `eval`
extra is installed (``pytest.importorskip``), so CI stays green without it.

The Ragas-dependent tests are deliberately narrow. They do not check that
faithfulness returns a sensible number — that is Ragas' job. They check the two
places this repo could silently break: that every metric still takes the inputs
the table says it takes, and that the judge is still built the way the Anthropic
API requires.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from core.pipeline import NO_ANSWER, RAGResult
from core.retrieval.base import RetrievalResult
from evaluation.harness import EvalCase
from evaluation.ragas_eval import metrics as metrics_module
from evaluation.ragas_eval.metrics import (
    END_TO_END,
    GENERATION,
    RETRIEVAL,
    SPECS,
    STAGE_ORDER,
    required_fields,
    specs_for,
)
from evaluation.ragas_eval.report import MetricScore, RagasReport
from evaluation.ragas_eval.runner import score
from evaluation.ragas_eval.samples import (
    RagasSample,
    SampleSet,
    context_texts,
    read_samples,
    sample_from_result,
    sample_from_retrieval,
    write_samples,
)


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
    assert sample.reference
    assert sample.response == ""
    assert sample.available() == {"user_input", "retrieved_contexts", "reference"}


def test_pipeline_sample_records_the_answer_and_the_sources_it_used():
    result = RAGResult(answer="Ann does [1].", sources=[_result()], retrieved=1)
    sample = sample_from_result(_case(), result)
    assert sample.response == "Ann does [1]."
    assert not sample.no_answer
    assert "response" in sample.available()


def test_a_declined_answer_is_flagged_not_judged():
    """The no-answer string is correct behaviour, not a response to score."""
    sample = sample_from_result(_case(), RAGResult(answer=NO_ANSWER, retrieved=3))
    assert sample.no_answer
    assert sample.response == ""
    assert "response" not in sample.available()


def test_available_treats_empty_as_absent():
    sample = RagasSample(case_id="c1", user_input="q?", retrieved_contexts=(), reference="")
    assert sample.available() == {"user_input"}


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


def test_required_fields_reads_the_real_signature():
    class Fake:
        async def ascore(self, user_input: str, response: str) -> None: ...

    assert required_fields(Fake()) == {"user_input", "response"}


# --- runner ----------------------------------------------------------------


class _FakeMetric:
    """Stands in for a Ragas metric: same ``ascore`` shape, no judge."""

    def __init__(self, value=1.0, raises: Exception | None = None):
        self.value = value
        self.raises = raises
        self.calls: list[dict] = []

    async def ascore(self, user_input: str, response: str, retrieved_contexts: list[str]):
        self.calls.append(
            {"user_input": user_input, "response": response,
             "retrieved_contexts": retrieved_contexts}
        )
        if self.raises:
            raise self.raises
        return type("Result", (), {"value": self.value, "reason": "because"})()


def _full_sample(case_id: str = "c1") -> RagasSample:
    return RagasSample(
        case_id=case_id,
        user_input="q?",
        retrieved_contexts=("[1] ctx",),
        response="an answer [1]",
        reference="the truth",
    )


def test_a_scored_case_records_the_value_and_the_judges_reason():
    metric = _FakeMetric(value=0.75)
    report = score([_full_sample()], [(SPECS["faithfulness"], metric)])

    assert [(s.status, s.value) for s in report.scores] == [("ok", 0.75)]
    assert report.scores[0].detail == "because"
    assert metric.calls[0]["retrieved_contexts"] == ["[1] ctx"]


def test_a_missing_input_skips_rather_than_scoring_zero():
    """A zero here would read as a regression no code change could fix."""
    sample = RagasSample(case_id="c1", user_input="q?", retrieved_contexts=("[1] ctx",))
    metric = _FakeMetric()
    report = score([sample], [(SPECS["faithfulness"], metric)])

    assert report.scores[0].status == "skipped"
    assert report.scores[0].value is None
    assert not metric.calls, "a metric must not be called without its inputs"


def test_skip_reasons_name_the_input_that_was_missing():
    no_reference = RagasSample(
        case_id="c1", user_input="q?", retrieved_contexts=("[1] ctx",), response="a"
    )
    declined = RagasSample(
        case_id="c2", user_input="q?", retrieved_contexts=("[1] ctx",),
        reference="t", no_answer=True,
    )
    nothing_retrieved = RagasSample(case_id="c3", user_input="q?", response="a", reference="t")

    class _NeedsEverything:
        async def ascore(
            self, user_input: str, response: str, retrieved_contexts: list[str], reference: str
        ): ...

    report = score(
        [no_reference, declined, nothing_retrieved],
        [(SPECS["answer_correctness"], _NeedsEverything())],
    )
    details = {s.case_id: s.detail for s in report.scores}
    assert details["c1"] == "no reference in the golden case"
    assert details["c2"] == "pipeline declined to answer"
    assert details["c3"] == "nothing retrieved above the score floor"


def test_a_judge_failure_is_recorded_not_raised():
    """One bad call must not discard a run that cost minutes and quota."""
    metrics = [
        (SPECS["faithfulness"], _FakeMetric(raises=RuntimeError("upstream 529"))),
        (SPECS["context_utilization"], _FakeMetric(value=1.0)),
    ]
    report = score([_full_sample()], metrics)

    by_metric = {s.metric: s for s in report.scores}
    assert by_metric["faithfulness"].status == "error"
    assert "upstream 529" in by_metric["faithfulness"].detail
    assert by_metric["context_utilization"].status == "ok"


def test_nan_counts_as_a_failure_not_a_score():
    report = score([_full_sample()], [(SPECS["faithfulness"], _FakeMetric(value=math.nan))])
    assert report.scores[0].status == "error"
    assert "NaN" in report.scores[0].detail
    assert report.summary("faithfulness").mean is None


def test_every_sample_is_scored_by_every_metric_and_progress_is_reported():
    seen: list[tuple[str, str, str]] = []
    samples = [_full_sample("c1"), _full_sample("c2")]
    metrics = [(SPECS["faithfulness"], _FakeMetric()), (SPECS["context_utilization"], _FakeMetric())]

    report = score(samples, metrics, concurrency=2, progress=lambda *a: seen.append(a))

    assert len(report.scores) == 4
    assert len(seen) == 4
    assert report.cases == 2


def test_scoring_no_samples_is_not_an_error():
    assert score([], [(SPECS["faithfulness"], _FakeMetric())]).scores == []


# --- report ----------------------------------------------------------------


def test_the_mean_is_over_scored_cases_only():
    report = RagasReport(
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
    report = RagasReport(scores=[MetricScore("c1", "context_recall", status="skipped")], cases=1)
    assert report.summary("context_recall").mean is None


def test_the_worst_case_of_an_inverted_metric_is_the_highest_one():
    report = RagasReport(
        scores=[
            MetricScore("c1", "noise_sensitivity", 0.1),
            MetricScore("c2", "noise_sensitivity", 0.9),
        ],
        cases=2,
    )
    assert report.summary("noise_sensitivity").worst == ("c2", 0.9)


def test_gates_compare_in_the_right_direction():
    report = RagasReport(
        scores=[
            MetricScore("c1", "faithfulness", 0.9),
            MetricScore("c1", "noise_sensitivity", 0.4),
        ],
        cases=1,
    )
    assert report.failing({"faithfulness": 0.8, "noise_sensitivity": 0.5}) == []
    problems = report.failing({"faithfulness": 0.95, "noise_sensitivity": 0.2})
    assert len(problems) == 2
    assert "below the required" in problems[0]
    assert "above the allowed" in problems[1]


def test_a_gate_on_an_unmeasured_metric_fails():
    """A gate that passes because nothing was measured is worse than a red build."""
    report = RagasReport(scores=[MetricScore("c1", "context_recall", status="skipped")], cases=1)
    assert report.failing({"context_recall": 0.5}) == [
        "context_recall: not measured (no case could be scored)"
    ]


def test_metrics_are_listed_in_stage_order():
    report = RagasReport(
        scores=[
            MetricScore("c1", "answer_correctness", 1.0),
            MetricScore("c1", "faithfulness", 1.0),
            MetricScore("c1", "context_recall", 1.0),
        ],
        cases=1,
    )
    assert report.metrics == ["context_recall", "faithfulness", "answer_correctness"]


def test_the_report_shows_denominators_skips_and_the_inverted_metric():
    report = RagasReport(
        scores=[
            MetricScore("c1", "faithfulness", 0.5),
            MetricScore("c2", "faithfulness", status="skipped", detail="no reference"),
            MetricScore("c1", "noise_sensitivity", 0.6),
            MetricScore("c1", "context_recall", status="error", detail="SDKError: 401"),
        ],
        cases=2,
    )
    text = report.format()
    assert "RETRIEVAL" in text and "GENERATION" in text and "END TO END" in text
    assert "lower is better" in text
    assert "no reference" in text
    assert "SDKError: 401" in text
    # A weak number is explained, so the reader knows what to do about it.
    assert "hallucination" in text


def test_verbose_lists_every_case():
    report = RagasReport(scores=[MetricScore("c1", "faithfulness", 1.0)], cases=1)
    assert "Per case:" in report.format(verbose=True)
    assert "Per case:" not in report.format()


# --- Ragas-facing: skipped unless the `eval` extra is installed ------------


def test_the_metric_table_matches_the_inputs_ragas_actually_asks_for():
    """The drift guard: a Ragas upgrade that changes a metric's inputs fails here."""
    pytest.importorskip("ragas")
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms.base import InstructorBaseRagasLLM

    from evaluation.ragas_eval.metrics import build_metric

    class _Judge(InstructorBaseRagasLLM):
        def generate(self, prompt, response_model):  # pragma: no cover - never called
            raise AssertionError("the table check must not call the judge")

        async def agenerate(self, prompt, response_model):  # pragma: no cover
            raise AssertionError("the table check must not call the judge")

    class _Embeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs):  # pragma: no cover
            raise AssertionError("the table check must not embed")

        async def aembed_text(self, text: str, **kwargs):  # pragma: no cover
            raise AssertionError("the table check must not embed")

    for spec in SPECS.values():
        metric = build_metric(spec, llm=_Judge(), embeddings=_Embeddings())
        fields = required_fields(metric)
        assert fields <= {"user_input", "retrieved_contexts", "response", "reference"}, (
            f"{spec.key} asks for an input samples do not carry: {fields}"
        )
        assert spec.needs_response == ("response" in fields), spec.key
        assert spec.needs_reference == ("reference" in fields), spec.key


def test_a_metric_is_not_handed_components_it_does_not_take():
    pytest.importorskip("ragas")
    from evaluation.ragas_eval.metrics import build_metric

    # semantic_similarity takes embeddings only; passing a judge is a TypeError
    # in Ragas, so build_metric must send only what the spec declares -- and must
    # refuse outright when the component a metric does need is absent.
    with pytest.raises(ValueError, match="needs judge embeddings"):
        build_metric(SPECS["semantic_similarity"], llm=object())


def test_the_judge_never_sends_sampling_parameters():
    """Anthropic's Messages.create() rejects temperature/top_p outright."""
    pytest.importorskip("ragas")
    from evaluation.ragas_eval.judge import drop_unsupported_model_args

    class _Judge:
        model_args = {"temperature": 0.01, "top_p": 0.1, "max_tokens": 4096}

    judge = drop_unsupported_model_args(_Judge())
    assert judge.model_args == {"max_tokens": 4096}


def test_a_judge_without_model_args_fails_loudly():
    """If Ragas renames model_args, fail here rather than on every judged call."""
    pytest.importorskip("ragas")
    from evaluation.ragas_eval.judge import drop_unsupported_model_args

    with pytest.raises(TypeError, match="model_args"):
        drop_unsupported_model_args(object())


def test_an_unsupported_judge_provider_is_refused():
    pytest.importorskip("ragas")
    from evaluation.ragas_eval.judge import build_judge

    with pytest.raises(ValueError, match="unknown judge provider"):
        build_judge(provider="llama")


def test_the_judge_embedder_is_the_corpus_embedder_and_stays_batched():
    pytest.importorskip("ragas")
    from ragas.embeddings.base import BaseRagasEmbedding

    from evaluation.ragas_eval.judge import EmbedderAdapter
    from shared.embeddings.base import Embedder

    class _Recording(Embedder):
        model = "mistral-embed-2312"
        dim = 4

        def __init__(self):
            self.batches: list[list[str]] = []

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.batches.append(list(texts))
            return [[0.0] * self.dim for _ in texts]

    embedder = _Recording()
    adapter = EmbedderAdapter(embedder)

    assert isinstance(adapter, BaseRagasEmbedding)
    assert adapter.model == "mistral-embed-2312"
    assert len(adapter.embed_text("one")) == 4
    adapter.embed_texts(["a", "b", "c"])
    assert asyncio.run(adapter.aembed_text("async")) == [0.0] * 4
    assert asyncio.run(adapter.aembed_texts(["x", "y"])) == [[0.0] * 4] * 2
    # One request for the three-text batch, not three.
    assert embedder.batches == [["one"], ["a", "b", "c"], ["async"], ["x", "y"]]


def test_concurrent_embedding_is_serialized():
    """One Mistral client driven from several threads at once is flaky."""
    pytest.importorskip("ragas")
    import time

    from evaluation.ragas_eval.judge import EmbedderAdapter
    from shared.embeddings.base import Embedder

    class _Overlapping(Embedder):
        model = "mistral-embed-2312"
        dim = 2

        def __init__(self):
            self.in_flight = 0
            self.max_in_flight = 0

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            time.sleep(0.01)  # long enough for a second caller to overlap
            self.in_flight -= 1
            return [[0.0, 0.0] for _ in texts]

    embedder = _Overlapping()
    adapter = EmbedderAdapter(embedder)

    async def hammer():
        await asyncio.gather(*(adapter.aembed_text(f"t{i}") for i in range(5)))

    asyncio.run(hammer())
    assert embedder.max_in_flight == 1


def test_the_judge_embedder_is_paced_for_the_free_tier():
    """Mistral allows 60 requests/minute; a 60-case run wants ~420."""
    pytest.importorskip("ragas")
    from evaluation.ragas_eval.judge import build_judge_embeddings
    from shared.config import EMBED_RATE_LIMIT_SLEEP_SECONDS

    adapter = build_judge_embeddings()
    assert adapter.embedder.pace_seconds == EMBED_RATE_LIMIT_SLEEP_SECONDS
    assert EMBED_RATE_LIMIT_SLEEP_SECONDS >= 1.0


def test_the_json_report_carries_metadata_summaries_and_every_score():
    """The console table is gone in a month; this is what a comparison reads."""
    report = RagasReport(
        scores=[
            MetricScore("c1", "faithfulness", 1.0, detail="all statements supported"),
            MetricScore("c2", "faithfulness", status="skipped", detail="no reference"),
            MetricScore("c1", "noise_sensitivity", 0.25),
        ],
        cases=2,
    )
    data = report.to_dict({"seed": 7, "stages": ["generation"]})

    assert data["meta"] == {"seed": 7, "stages": ["generation"]}
    assert data["cases"] == 2
    assert data["metrics"]["faithfulness"]["mean"] == 1.0
    assert data["metrics"]["faithfulness"]["skipped"] == 1
    assert data["metrics"]["faithfulness"]["stage"] == "generation"
    assert data["metrics"]["noise_sensitivity"]["lower_is_better"] is True
    assert len(data["scores"]) == 3
    assert data["scores"][0]["detail"] == "all statements supported"


def test_the_json_report_survives_a_roundtrip():
    import json

    report = RagasReport(scores=[MetricScore("c1", "faithfulness", 0.5)], cases=1)
    assert json.loads(json.dumps(report.to_dict()))["metrics"]["faithfulness"]["mean"] == 0.5


def test_the_plots_render_from_a_stored_report(tmp_path):
    """Smoke test: the figures must survive a report with skips and a missing metric."""
    pytest.importorskip("matplotlib")
    from evaluation import plots

    report = RagasReport(
        scores=[
            MetricScore("c1", "context_relevance", 1.0),
            MetricScore("c1", "faithfulness", 0.5),
            MetricScore("c1", "context_utilization", 0.0),
            MetricScore("c1", "noise_sensitivity", 0.4),
            MetricScore("c2", "context_relevance", 0.9),
            MetricScore("c2", "faithfulness", 1.0),
            MetricScore("c2", "context_utilization", 1.0),
            MetricScore("c2", "noise_sensitivity", status="skipped", detail="no reference"),
        ],
        cases=2,
    ).to_dict({"seed": 1})

    assert plots.heatmap(report, tmp_path / "h.png").exists()
    assert plots.metric_means(report, tmp_path / "m.png").exists()
    assert plots.paired_metrics(
        report, "context_utilization", "faithfulness", tmp_path / "p.png", "t", "s"
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
    import json

    directory = tmp_path / name
    directory.mkdir()
    report = RagasReport(
        scores=[MetricScore("c1", "faithfulness", faithfulness),
                MetricScore("c1", "noise_sensitivity", 0.2)],
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
    from evaluation.run_ragas import _median

    assert _median([]) is None
    assert _median([5.0]) == 5.0
    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median([4.0, 1.0, 3.0, 2.0]) == 2.5

"""Which metric probes which point in the pipeline.

The table below is the whole framework's opinion: a RAG system has three places
it can fail, and each one has its own metrics.

    retrieval   the index and the embedding — was the right context returned?
    generation  the prompt and the model — was the returned context used, and
                only the returned context?
    end_to_end  everything together — was the answer right?

Reading them in that order localises a regression. A faithfulness drop with
context_recall flat is the generator's fault; both dropping together is the
retriever's, and fixing the prompt would be wasted work.

Metric classes are named as ``"module:ClassName"`` strings and imported on use,
the same pattern as ``_EMBEDDERS`` and ``_GENERATORS``: this module stays
importable — and testable — without DeepEval installed, and a stage that needs
no embeddings never builds an embedder.

``needs_response`` / ``needs_reference`` are declared here so a runner can tell,
before spending a cent, whether a stage requires calling the generator or a
golden reference. They are not trusted blindly: ``required_fields`` reads the
metric's real ``_required_params`` at score time, and a test asserts the two
agree, so a DeepEval upgrade that changes a metric's inputs fails loudly
instead of silently scoring the wrong thing.

Four metrics from the Ragas era have no DeepEval equivalent and were dropped
rather than reinvented: ``context_entity_recall``, ``response_groundedness``
(faithfulness already covers it), ``context_utilization`` (measured on
2026-09-17 as reading 0.00 on any list-style answer, which is this product's
main question shape), and ``noise_sensitivity``. Numbers here are therefore not
comparable to runs stored before the port — a different judge on a different
rubric never would have been anyway.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from typing import Any

RETRIEVAL = "retrieval"
GENERATION = "generation"
END_TO_END = "end_to_end"

#: Cheapest and most diagnostic first. Also the order results are printed in.
STAGE_ORDER = (RETRIEVAL, GENERATION, END_TO_END)

#: DeepEval runs a metric's own sub-calls concurrently when ``async_mode`` is
#: on. The runner already governs concurrency with a semaphore sized for the
#: judge's rate limit, so leaving it on would fan out underneath that budget and
#: make the limit meaningless. Every metric is built with it off.
_SHARED_OPTIONS: dict[str, Any] = {"async_mode": False}

#: What G-Eval is told to do for ``answer_correctness``. Spelled out as explicit
#: steps rather than left to ``criteria``, for two reasons: G-Eval otherwise
#: spends a judge call per run inventing them, and the steps it invents vary
#: between runs, which is not a property you want in the thing you are measuring
#: against. Step 4 is the important one here — measured 2026-09-17, broad
#: questions have far more valid people than any reference names, so counting
#: unnamed-but-correct people as errors would score the corpus, not the system.
_ANSWER_CORRECTNESS_STEPS = [
    "Read the expected output as ground truth about which people and facts are correct.",
    "Check each factual claim in the actual output — names, roles, research areas, "
    "affiliations — against the expected output.",
    "Penalise contradictions heavily. Penalise omitting a person or fact that the "
    "expected output names moderately.",
    "Do not penalise additional people or details that are absent from the expected "
    "output but not contradicted by it, and ignore citation markers such as [1].",
]


@dataclass(frozen=True)
class MetricSpec:
    """One metric: where it belongs, how to build it, and what it tells you."""

    key: str
    stage: str
    #: "module:ClassName", resolved by ``build_metric``.
    target: str
    needs_llm: bool = True
    needs_embeddings: bool = False
    needs_response: bool = False
    needs_reference: bool = False
    #: Constructor kwargs beyond the judge components.
    options: dict[str, Any] | None = None
    #: What a LOW score means — printed with the report, because a number
    #: nobody can interpret does not change any decision.
    means: str = ""


_SPECS: tuple[MetricSpec, ...] = (
    # --- retrieval: no generator involved, so this stage is cheap -----------
    MetricSpec(
        key="context_relevance",
        stage=RETRIEVAL,
        target="deepeval.metrics:ContextualRelevancyMetric",
        means="retrieved chunks are off-topic for the question (embedding or top_k problem)",
    ),
    MetricSpec(
        key="context_precision",
        stage=RETRIEVAL,
        target="deepeval.metrics:ContextualPrecisionMetric",
        needs_reference=True,
        means="the useful chunks are buried under irrelevant ones (ranking problem)",
    ),
    MetricSpec(
        key="context_recall",
        stage=RETRIEVAL,
        target="deepeval.metrics:ContextualRecallMetric",
        needs_reference=True,
        means="the context is missing claims the answer needs (chunking or corpus gap)",
    ),
    # --- generation: judged against the context the model was given --------
    MetricSpec(
        key="faithfulness",
        stage=GENERATION,
        target="deepeval.metrics:FaithfulnessMetric",
        needs_response=True,
        means="the answer asserts things the context does not support — hallucination",
    ),
    MetricSpec(
        key="answer_relevancy",
        stage=GENERATION,
        target="deepeval.metrics:AnswerRelevancyMetric",
        needs_response=True,
        means="the answer is padded or evasive rather than addressing the question asked",
    ),
    # --- end to end: needs a reference answer to compare against -----------
    MetricSpec(
        key="answer_correctness",
        stage=END_TO_END,
        target="deepeval.metrics:GEval",
        needs_response=True,
        needs_reference=True,
        options={
            "name": "answer_correctness",
            "evaluation_steps": _ANSWER_CORRECTNESS_STEPS,
            "evaluation_params": ["input", "actual_output", "expected_output"],
        },
        means="the answer disagrees with the reference on the facts",
    ),
    MetricSpec(
        key="semantic_similarity",
        stage=END_TO_END,
        target="evaluation.deepeval_eval.similarity:SemanticSimilarityMetric",
        needs_llm=False,
        needs_embeddings=True,
        needs_response=True,
        needs_reference=True,
        means="the answer is not saying the same thing as the reference (no judge involved)",
    ),
)

#: Metrics where a high score is bad. Empty since the port: Ragas'
#: ``noise_sensitivity`` was the only one and DeepEval has no equivalent. Kept
#: because the reporting path that reads it is written and tested, and a G-Eval
#: metric phrased as a rate (a hallucination rate, say) would belong here on the
#: day it is added.
LOWER_IS_BETTER: frozenset[str] = frozenset()

SPECS: dict[str, MetricSpec] = {spec.key: spec for spec in _SPECS}


def _validate() -> None:
    """Fail at import on a table that cannot be what the author meant."""
    keys = [spec.key for spec in _SPECS]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"duplicate metric keys: {duplicates}")

    unknown = {spec.stage for spec in _SPECS} - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"metrics declare unknown stage(s): {sorted(unknown)}")

    for spec in _SPECS:
        if not spec.needs_llm and not spec.needs_embeddings:
            raise ValueError(f"{spec.key} needs neither a judge nor embeddings")
        if ":" not in spec.target:
            raise ValueError(f"{spec.key} target must be 'module:ClassName', got {spec.target!r}")


_validate()


def specs_for(stages: tuple[str, ...]) -> tuple[MetricSpec, ...]:
    """The metrics belonging to ``stages``, in stage order."""
    unknown = set(stages) - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"unknown stage(s) {sorted(unknown)}; available: {STAGE_ORDER}")
    return tuple(
        spec for stage in STAGE_ORDER if stage in stages for spec in _SPECS if spec.stage == stage
    )


def build_metric(spec: MetricSpec, llm: Any = None, embedder: Any = None) -> Any:
    """Import and construct the DeepEval metric a spec names.

    Constructing a metric makes no network call, so this is cheap.

    The ``llm is None`` guard is not defensive noise. DeepEval resolves a
    missing — or merely string-valued — ``model`` to ``OpenAIModel``, which then
    fails on a missing ``OPENAI_API_KEY`` from somewhere deep inside the metric.
    There is no OpenAI key in this project by design, so the failure would be
    both confusing and late; refusing here names the real problem.
    """
    module_path, _, class_name = spec.target.partition(":")
    metric_class = getattr(importlib.import_module(module_path), class_name)

    kwargs: dict[str, Any] = dict(_SHARED_OPTIONS)
    kwargs.update(spec.options or {})
    if "evaluation_params" in kwargs:
        from deepeval.test_case import SingleTurnParams

        kwargs["evaluation_params"] = [
            SingleTurnParams(value) for value in kwargs["evaluation_params"]
        ]
    if spec.needs_llm:
        if llm is None:
            raise ValueError(
                f"metric {spec.key} needs a judge LLM; DeepEval would otherwise fall back "
                "to OpenAI and fail on a missing OPENAI_API_KEY"
            )
        kwargs["model"] = llm
    if spec.needs_embeddings:
        if embedder is None:
            raise ValueError(f"metric {spec.key} needs an embedder")
        kwargs["embedder"] = embedder
    return metric_class(**kwargs)


#: Where a metric states the test-case fields it reads, most specific first.
#: ``GEval`` is the reason there are two. It declares ``_required_params`` as a
#: bare *annotation* (``List[SingleTurnParams]``) and never assigns it, so the
#: attribute exists, passes ``hasattr``, and yields a ``typing`` object instead
#: of parameters; its real inputs are the ``evaluation_params`` it was built
#: with. Reading the wrong one skips every G-Eval case with an unreadable
#: message. Both are checked, and anything that is not a list of enum members
#: is rejected rather than coerced.
_PARAM_ATTRIBUTES = ("evaluation_params", "_required_params")


def required_fields(metric: Any) -> frozenset[str]:
    """The test-case fields a metric actually reads.

    Read from the live metric rather than hard-coded, so this framework follows
    a DeepEval upgrade instead of quietly scoring a test case that is missing
    something the metric needed.
    """
    for attribute in _PARAM_ATTRIBUTES:
        params = getattr(metric, attribute, None)
        if not isinstance(params, (list, tuple)) or not params:
            continue
        values = [getattr(p, "value", None) for p in params]
        if all(isinstance(value, str) for value in values):
            return frozenset(values)

    raise TypeError(
        f"{type(metric).__name__} states no readable required parameters "
        f"(checked {', '.join(_PARAM_ATTRIBUTES)}); DeepEval's metric contract "
        "changed and the skip logic can no longer tell what it needs."
    )


def supports_async(metric: Any) -> bool:
    """Whether a metric offers ``a_measure``. Every DeepEval metric does."""
    return callable(getattr(metric, "a_measure", None)) and inspect.iscoroutinefunction(
        metric.a_measure
    )

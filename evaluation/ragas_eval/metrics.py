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
importable — and testable — without Ragas installed, and a stage that needs no
embeddings never builds an embedder.

``needs_response`` / ``needs_reference`` are declared here so a runner can tell,
before spending a cent, whether a stage requires calling the generator or a
golden reference. They are not trusted blindly: ``required_fields`` reads the
real ``ascore`` signature at score time, and a test asserts the two agree, so a
Ragas upgrade that changes a metric's inputs fails loudly instead of silently
scoring the wrong thing.
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
        target="ragas.metrics.collections:ContextRelevance",
        means="retrieved chunks are off-topic for the question (embedding or top_k problem)",
    ),
    MetricSpec(
        key="context_precision",
        stage=RETRIEVAL,
        target="ragas.metrics.collections:ContextPrecisionWithReference",
        needs_reference=True,
        means="the useful chunks are buried under irrelevant ones (ranking problem)",
    ),
    MetricSpec(
        key="context_recall",
        stage=RETRIEVAL,
        target="ragas.metrics.collections:ContextRecall",
        needs_reference=True,
        means="the context is missing claims the answer needs (chunking or corpus gap)",
    ),
    MetricSpec(
        key="context_entity_recall",
        stage=RETRIEVAL,
        target="ragas.metrics.collections:ContextEntityRecall",
        needs_reference=True,
        means="the named entities the answer needs (people, labs) never got retrieved",
    ),
    # --- generation: judged against the context the model was given --------
    MetricSpec(
        key="faithfulness",
        stage=GENERATION,
        target="ragas.metrics.collections:Faithfulness",
        needs_response=True,
        means="the answer asserts things the context does not support — hallucination",
    ),
    MetricSpec(
        key="response_groundedness",
        stage=GENERATION,
        target="ragas.metrics.collections:ResponseGroundedness",
        needs_response=True,
        means="the answer is not traceable to the context (the citations are decoration)",
    ),
    MetricSpec(
        key="context_utilization",
        stage=GENERATION,
        target="ragas.metrics.collections:ContextPrecisionWithoutReference",
        needs_response=True,
        #: Measured 2026-09-17: this scores 0.00 on any answer that lists several
        #: people, even where faithfulness is 1.00 and context_precision is 1.00 on
        #: the same case — no single chunk "yields" a list answer, so the judge
        #: rejects every one. "Who works on X?" is this product's main question
        #: shape, so a low mean here is usually an artifact, not a diagnosis.
        means="the answer ignored the top-ranked chunks — but note this metric reads 0 "
              "on list-style answers regardless (see evaluation/results/2026-09-17-sample10)",
    ),
    MetricSpec(
        key="answer_relevancy",
        stage=GENERATION,
        target="ragas.metrics.collections:AnswerRelevancy",
        needs_embeddings=True,
        needs_response=True,
        means="the answer is padded or evasive rather than addressing the question asked",
    ),
    # --- end to end: needs a reference answer to compare against -----------
    MetricSpec(
        key="answer_correctness",
        stage=END_TO_END,
        target="ragas.metrics.collections:AnswerCorrectness",
        needs_embeddings=True,
        needs_response=True,
        needs_reference=True,
        means="the answer disagrees with the reference on the facts",
    ),
    MetricSpec(
        key="semantic_similarity",
        stage=END_TO_END,
        target="ragas.metrics.collections:SemanticSimilarity",
        needs_llm=False,
        needs_embeddings=True,
        needs_response=True,
        needs_reference=True,
        means="the answer is not saying the same thing as the reference (no judge involved)",
    ),
    MetricSpec(
        key="noise_sensitivity",
        stage=END_TO_END,
        target="ragas.metrics.collections:NoiseSensitivity",
        needs_response=True,
        needs_reference=True,
        #: The one metric here where LOWER is better, hence the inverted wording.
        means="HIGH means irrelevant retrieved chunks are corrupting correct answers",
    ),
)

#: Metrics where a high score is bad. Only noise_sensitivity, but a report that
#: prints every mean the same way would invite exactly the wrong conclusion.
LOWER_IS_BETTER = frozenset({"noise_sensitivity"})

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


def build_metric(spec: MetricSpec, llm: Any = None, embeddings: Any = None) -> Any:
    """Import and construct the Ragas metric a spec names.

    Constructing a metric makes no network call, so this is cheap; what it does
    need is the components the metric declares. Passing a judge to a metric that
    takes none is a ``TypeError`` in Ragas, so only the declared ones are sent.
    """
    module_path, _, class_name = spec.target.partition(":")
    metric_class = getattr(importlib.import_module(module_path), class_name)

    kwargs: dict[str, Any] = dict(spec.options or {})
    if spec.needs_llm:
        if llm is None:
            raise ValueError(f"metric {spec.key} needs a judge LLM")
        kwargs["llm"] = llm
    if spec.needs_embeddings:
        if embeddings is None:
            raise ValueError(f"metric {spec.key} needs judge embeddings")
        kwargs["embeddings"] = embeddings
    return metric_class(**kwargs)


def required_fields(metric: Any) -> frozenset[str]:
    """The sample fields a metric's ``ascore`` actually takes.

    Read from the live signature rather than hard-coded, so this framework
    follows a Ragas upgrade instead of quietly passing a stale set of kwargs.
    """
    parameters = inspect.signature(metric.ascore).parameters
    return frozenset(name for name in parameters if name != "self")

"""The record a judge scores: one question, its context, the answer, the truth.

Pure: no Ragas, no SDKs, no network. A sample is built either from a live run or
read back from a dump, and the metrics cannot tell the difference — which is the
point. Retrieval and generation are the expensive, rate-limited half of an
evaluation; judging is the half you iterate on. Recording samples once and
replaying them keeps those two costs separate.

Two details matter for the numbers to mean anything:

- **Contexts are rendered exactly as the generator saw them.** ``PromptBuilder``
  is what built the model's prompt, so it builds the judge's contexts too. Score
  the raw chunk text instead and faithfulness is measured against a prompt that
  was never sent.
- **Empty is missing.** A case with no reference, or a run with no retrieved
  context, does not get a zero: it gets skipped, and the skip is reported. A
  zero would silently drag a mean down and look like a regression.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from core.llm.prompts import PromptBuilder
from core.pipeline import NO_ANSWER, RAGResult
from core.retrieval.base import RetrievalResult
from evaluation.harness import EvalCase

#: Ragas' own field names, which are also this dataclass's field names. The
#: metrics declare what they need in these terms (see ``metrics.py``), so the
#: mapping from a sample to a metric call is a dict lookup, not a translation
#: layer that can drift.
JUDGED_FIELDS = ("user_input", "retrieved_contexts", "response", "reference")


def context_texts(results: Sequence[RetrievalResult]) -> list[str]:
    """Render retrieved chunks the way the generator's prompt renders them."""
    builder = PromptBuilder()
    return [builder.context_block(i, r) for i, r in enumerate(results, start=1)]


@dataclass(frozen=True)
class RagasSample:
    """One golden case joined to what the system did with it.

    ``retrieved_ids`` is carried alongside the context text so a single run can
    be scored both ways: by the judge, and by the slug-based metrics in
    ``evaluation.harness``. Two evaluations of the same run should never come
    from two different runs.
    """

    case_id: str
    user_input: str
    retrieved_contexts: tuple[str, ...] = ()
    response: str = ""
    reference: str = ""
    retrieved_ids: tuple[str, ...] = ()
    #: True when the pipeline declined to answer. Not a failure — the score
    #: floor firing on a bad match is correct behaviour — but it is not a
    #: judgeable answer either, so it is counted separately.
    no_answer: bool = False
    #: Per-stage latency in milliseconds, straight from ``RAGResult``. Recorded
    #: because "which model should we use" is never only a quality question,
    #: and the generate stage is the only one that changes when the model does.
    timings_ms: dict[str, float] = field(default_factory=dict)

    def inputs(self) -> dict[str, object]:
        """The judged fields, as Ragas' metrics expect to receive them."""
        return {
            "user_input": self.user_input,
            "retrieved_contexts": list(self.retrieved_contexts),
            "response": self.response,
            "reference": self.reference,
        }

    def available(self) -> frozenset[str]:
        """Which judged fields this sample actually carries (empty = absent)."""
        return frozenset(name for name, value in self.inputs().items() if value)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "user_input": self.user_input,
            "retrieved_contexts": list(self.retrieved_contexts),
            "response": self.response,
            "reference": self.reference,
            "retrieved_ids": list(self.retrieved_ids),
            "no_answer": self.no_answer,
            "timings_ms": dict(self.timings_ms),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> RagasSample:
        missing = {"case_id", "user_input"} - set(raw)
        if missing:
            raise ValueError(f"sample is missing {sorted(missing)}: {raw}")
        return cls(
            case_id=raw["case_id"],
            user_input=raw["user_input"],
            retrieved_contexts=tuple(raw.get("retrieved_contexts", ())),
            response=raw.get("response", ""),
            reference=raw.get("reference", ""),
            retrieved_ids=tuple(raw.get("retrieved_ids", ())),
            no_answer=bool(raw.get("no_answer", False)),
            timings_ms=dict(raw.get("timings_ms") or {}),
        )


@dataclass
class SampleSet:
    """Samples plus the run settings that produced them.

    A score is only comparable to another score taken at the same ``top_k`` and
    ``min_score`` against the same index, so a dump carries them. Reading a dump
    back and finding these do not match what you meant to test is the difference
    between a regression and a configuration change.
    """

    samples: list[RagasSample] = field(default_factory=list)
    index_name: str = ""
    top_k: int = 0
    min_score: float = 0.0
    chat_model: str = ""

    def __iter__(self):
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def describe(self) -> str:
        return (
            f"{len(self.samples)} sample(s) from {self.index_name or 'unknown index'} "
            f"(top_k={self.top_k}, min_score={self.min_score}, "
            f"chat_model={self.chat_model or 'none'})"
        )


def sample_from_retrieval(
    case: EvalCase, results: Sequence[RetrievalResult]
) -> RagasSample:
    """A retrieval-only sample: context, but no answer to judge."""
    return RagasSample(
        case_id=case.id,
        user_input=case.question,
        retrieved_contexts=tuple(context_texts(results)),
        reference=case.reference,
        retrieved_ids=tuple(r.document_id for r in results),
    )


def sample_from_result(case: EvalCase, result: RAGResult) -> RagasSample:
    """A full sample from a pipeline run, including the answer.

    ``RAGResult.sources`` is post-floor, so the contexts here are exactly the
    ones that reached the model. A declined answer is recorded as such and left
    out of the answer text, so no metric judges the no-answer string as if it
    were a real response.
    """
    declined = result.answer == NO_ANSWER
    return RagasSample(
        case_id=case.id,
        user_input=case.question,
        retrieved_contexts=tuple(context_texts(result.sources)),
        response="" if declined else result.answer,
        reference=case.reference,
        retrieved_ids=tuple(s.document_id for s in result.sources),
        no_answer=declined,
        timings_ms=dict(result.timings_ms),
    )


def write_samples(path: Path, sample_set: SampleSet) -> None:
    """Write a run as JSONL: a settings header line, then one line per sample."""
    header = {
        "_run": {
            "index_name": sample_set.index_name,
            "top_k": sample_set.top_k,
            "min_score": sample_set.min_score,
            "chat_model": sample_set.chat_model,
        }
    }
    lines = [json.dumps(header)]
    lines += [json.dumps(s.to_dict(), ensure_ascii=False) for s in sample_set.samples]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_samples(path: Path) -> SampleSet:
    """Read a dump written by ``write_samples``."""
    sample_set = SampleSet()
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            raw = json.loads(stripped)
            if "_run" in raw:
                run = raw["_run"]
                sample_set.index_name = run.get("index_name", "")
                sample_set.top_k = int(run.get("top_k", 0))
                sample_set.min_score = float(run.get("min_score", 0.0))
                sample_set.chat_model = run.get("chat_model", "")
                continue
            sample_set.samples.append(RagasSample.from_dict(raw))
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}:{number}: {e}") from None

    if not sample_set.samples:
        raise ValueError(f"no samples in {path}")
    return sample_set


def cases_by_id(cases: Iterable[EvalCase]) -> dict[str, EvalCase]:
    """Index golden cases so a dump can be re-joined to them by ``case_id``."""
    return {case.id: case for case in cases}

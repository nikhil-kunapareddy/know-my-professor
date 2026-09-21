"""The record a judge scores: one question, its context, the answer, the truth.

Pure: no DeepEval, no SDKs, no network. A sample is built either from a live run
or read back from a dump, and the metrics cannot tell the difference — which is
the point. Retrieval and generation are the expensive, rate-limited half of an
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

**In-memory field names are DeepEval's** (``input``, ``actual_output``,
``expected_output``, ``retrieval_context``), so building an ``LLMTestCase`` is a
construction and not a translation that can drift from what the metrics read.

**On-disk field names are not.** The JSONL keys are the ones written before this
package moved from Ragas to DeepEval, and ``from_dict`` accepts either spelling.
Dumps are the whole point of ``--from-dump``: a recorded run is re-judgeable for
the price of the judge alone, and breaking last month's dump to tidy four key
names would throw away exactly the artefact the flag exists to serve. The
mapping lives in ``_DISK_ALIASES`` and nowhere else.
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

#: DeepEval's ``LLMTestCase`` field names, which are also this dataclass's field
#: names. Metrics declare what they need in these terms via ``_required_params``
#: (see ``metrics.required_fields``), so the check for "can this sample be
#: scored" is a set operation rather than a lookup table.
JUDGED_FIELDS = ("input", "retrieval_context", "actual_output", "expected_output")

#: on-disk key -> in-memory field. Read-only compatibility with dumps written
#: before the DeepEval port; ``to_dict`` still writes the on-disk spelling.
_DISK_ALIASES = {
    "user_input": "input",
    "retrieved_contexts": "retrieval_context",
    "response": "actual_output",
    "reference": "expected_output",
}
_TO_DISK = {memory: disk for disk, memory in _DISK_ALIASES.items()}


def context_texts(results: Sequence[RetrievalResult]) -> list[str]:
    """Render retrieved chunks the way the generator's prompt renders them."""
    builder = PromptBuilder()
    return [builder.context_block(i, r) for i, r in enumerate(results, start=1)]


@dataclass(frozen=True)
class JudgedSample:
    """One golden case joined to what the system did with it.

    ``retrieved_ids`` is carried alongside the context text so a single run can
    be scored both ways: by the judge, and by the slug-based metrics in
    ``evaluation.harness``. Two evaluations of the same run should never come
    from two different runs.
    """

    case_id: str
    input: str
    retrieval_context: tuple[str, ...] = ()
    actual_output: str = ""
    expected_output: str = ""
    retrieved_ids: tuple[str, ...] = ()
    #: True when the pipeline declined to answer. Not a failure — a refusal on a
    #: genuinely out-of-scope question is correct behaviour — but it is not a
    #: judgeable answer either, so it is counted separately.
    no_answer: bool = False
    #: Per-stage latency in milliseconds, straight from ``RAGResult``. Recorded
    #: because "which model should we use" is never only a quality question,
    #: and the generate stage is the only one that changes when the model does.
    timings_ms: dict[str, float] = field(default_factory=dict)

    def inputs(self) -> dict[str, object]:
        """The judged fields, under the names DeepEval's metrics read."""
        return {
            "input": self.input,
            "retrieval_context": list(self.retrieval_context),
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
        }

    def available(self) -> frozenset[str]:
        """Which judged fields this sample actually carries (empty = absent)."""
        return frozenset(name for name, value in self.inputs().items() if value)

    def to_test_case(self):
        """Build the ``LLMTestCase`` a DeepEval metric scores.

        Imported here rather than at module scope so this module — and every
        test of it — stays free of the ``eval`` extra.
        """
        from deepeval.test_case import LLMTestCase

        return LLMTestCase(
            input=self.input,
            actual_output=self.actual_output,
            expected_output=self.expected_output,
            retrieval_context=list(self.retrieval_context),
        )

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            _TO_DISK["input"]: self.input,
            _TO_DISK["retrieval_context"]: list(self.retrieval_context),
            _TO_DISK["actual_output"]: self.actual_output,
            _TO_DISK["expected_output"]: self.expected_output,
            "retrieved_ids": list(self.retrieved_ids),
            "no_answer": self.no_answer,
            "timings_ms": dict(self.timings_ms),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> JudgedSample:
        """Read a sample, accepting either the on-disk or in-memory spelling."""
        merged = dict(raw)
        for disk, memory in _DISK_ALIASES.items():
            if disk in merged and memory not in merged:
                merged[memory] = merged[disk]

        missing = {"case_id", "input"} - set(merged)
        if missing:
            raise ValueError(f"sample is missing {sorted(missing)}: {raw}")
        return cls(
            case_id=merged["case_id"],
            input=merged["input"],
            retrieval_context=tuple(merged.get("retrieval_context", ())),
            actual_output=merged.get("actual_output", ""),
            expected_output=merged.get("expected_output", ""),
            retrieved_ids=tuple(merged.get("retrieved_ids", ())),
            no_answer=bool(merged.get("no_answer", False)),
            timings_ms=dict(merged.get("timings_ms") or {}),
        )


@dataclass
class SampleSet:
    """Samples plus the run settings that produced them.

    A score is only comparable to another score taken at the same ``top_k`` and
    ``min_score`` against the same index, so a dump carries them. Reading a dump
    back and finding these do not match what you meant to test is the difference
    between a regression and a configuration change.
    """

    samples: list[JudgedSample] = field(default_factory=list)
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


def sample_from_retrieval(case: EvalCase, results: Sequence[RetrievalResult]) -> JudgedSample:
    """A retrieval-only sample: context, but no answer to judge."""
    return JudgedSample(
        case_id=case.id,
        input=case.question,
        retrieval_context=tuple(context_texts(results)),
        expected_output=case.reference,
        retrieved_ids=tuple(r.document_id for r in results),
    )


def sample_from_result(case: EvalCase, result: RAGResult) -> JudgedSample:
    """A full sample from a pipeline run, including the answer.

    ``RAGResult.sources`` is post-floor, so the contexts here are exactly the
    ones that reached the model. A declined answer is recorded as such and left
    out of the answer text, so no metric judges the no-answer string as if it
    were a real response.
    """
    declined = result.answer == NO_ANSWER
    return JudgedSample(
        case_id=case.id,
        input=case.question,
        retrieval_context=tuple(context_texts(result.sources)),
        actual_output="" if declined else result.answer,
        expected_output=case.reference,
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
    """Read a dump written by ``write_samples``, including pre-DeepEval ones."""
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
            sample_set.samples.append(JudgedSample.from_dict(raw))
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}:{number}: {e}") from None

    if not sample_set.samples:
        raise ValueError(f"no samples in {path}")
    return sample_set


def cases_by_id(cases: Iterable[EvalCase]) -> dict[str, EvalCase]:
    """Index golden cases so a dump can be re-joined to them by ``case_id``."""
    return {case.id: case for case in cases}

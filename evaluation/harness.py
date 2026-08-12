"""Scoring for retrieval and answer quality.

Every change worth making to a RAG system — a new embedding provider, a chunking
tweak, a prompt edit, a new corpus — moves answer quality, and until it is
measured the only signal is whether one hand-typed question still looks fine.
These metrics turn "does this feel better?" into a number that can be compared
across runs.

Pure functions over recorded results: no network, no provider SDKs. The runner
(``run_eval.py``) supplies the results; this module only scores them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.llm.prompts import cited_numbers

#: Vector IDs are ``{slug}#{section_type}``.
ID_SEPARATOR = "#"


def slug_of(document_id: str) -> str:
    """The entity a chunk belongs to."""
    return document_id.split(ID_SEPARATOR, 1)[0]


def section_of(document_id: str) -> str:
    """The section type a chunk came from ("" if the id has none)."""
    _, _, section = document_id.partition(ID_SEPARATOR)
    return section


@dataclass(frozen=True)
class EvalCase:
    """One golden question and what a good answer would draw on."""

    id: str
    question: str
    expected_slugs: tuple[str, ...]
    expected_sections: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> EvalCase:
        missing = {"id", "question", "expected_slugs"} - set(raw)
        if missing:
            raise ValueError(f"golden case is missing {sorted(missing)}: {raw}")
        return cls(
            id=raw["id"],
            question=raw["question"],
            expected_slugs=tuple(raw["expected_slugs"]),
            expected_sections=tuple(raw.get("expected_sections", ())),
            notes=raw.get("notes", ""),
        )


def load_cases(path: Path) -> list[EvalCase]:
    """Read a JSONL golden file, skipping blank lines and ``#`` comments."""
    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            cases.append(EvalCase.from_dict(json.loads(stripped)))
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}:{number}: {e}") from None

    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate case ids in {path}: {duplicates}")
    return cases


@dataclass
class CaseOutcome:
    """What one case produced, plus its derived scores."""

    case: EvalCase
    retrieved_ids: list[str] = field(default_factory=list)
    answer: str | None = None

    @property
    def retrieved_slugs(self) -> list[str]:
        """Slugs in retrieval order, first occurrence only."""
        seen: list[str] = []
        for document_id in self.retrieved_ids:
            slug = slug_of(document_id)
            if slug not in seen:
                seen.append(slug)
        return seen

    @property
    def rank(self) -> int | None:
        """1-based position of the first expected slug, or None if absent."""
        for position, slug in enumerate(self.retrieved_slugs, start=1):
            if slug in self.case.expected_slugs:
                return position
        return None

    @property
    def hit(self) -> bool:
        """Did retrieval surface at least one expected professor?"""
        return self.rank is not None

    @property
    def reciprocal_rank(self) -> float:
        """1/rank — rewards putting the right professor near the top, not just in the list."""
        return 1.0 / self.rank if self.rank else 0.0

    @property
    def section_hit(self) -> bool | None:
        """Whether an expected section was retrieved (None if the case names none)."""
        if not self.case.expected_sections:
            return None
        return any(section_of(i) in self.case.expected_sections for i in self.retrieved_ids)

    @property
    def cited_slugs(self) -> list[str]:
        """Slugs the answer actually cited, resolved through the [n] markers."""
        if not self.answer:
            return []
        # Markers are 1-based over the retrieved CHUNK list, not the slug list,
        # so resolve against retrieved_ids and then dedupe.
        out: list[str] = []
        for number in sorted(cited_numbers(self.answer)):
            index = number - 1
            if 0 <= index < len(self.retrieved_ids):
                slug = slug_of(self.retrieved_ids[index])
                if slug not in out:
                    out.append(slug)
        return out

    @property
    def citation_precision(self) -> float | None:
        """Share of cited professors that were expected (None if nothing cited)."""
        cited = self.cited_slugs
        if not cited:
            return None
        correct = sum(1 for slug in cited if slug in self.case.expected_slugs)
        return correct / len(cited)


@dataclass
class EvalReport:
    """Aggregate scores across every case."""

    outcomes: list[CaseOutcome]
    top_k: int

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def recall_at_k(self) -> float:
        """Share of questions where an expected professor was retrieved at all.

        The headline number: if the right chunk never reaches the model, no
        amount of prompt work can produce a correct answer.
        """
        if not self.outcomes:
            return 0.0
        return sum(1 for o in self.outcomes if o.hit) / self.total

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank — how high up the right professor lands."""
        if not self.outcomes:
            return 0.0
        return sum(o.reciprocal_rank for o in self.outcomes) / self.total

    @property
    def section_recall(self) -> float | None:
        scored = [o for o in self.outcomes if o.section_hit is not None]
        if not scored:
            return None
        return sum(1 for o in scored if o.section_hit) / len(scored)

    @property
    def citation_precision(self) -> float | None:
        scored = [o.citation_precision for o in self.outcomes if o.citation_precision is not None]
        if not scored:
            return None
        return sum(scored) / len(scored)

    @property
    def misses(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if not o.hit]

    def format(self) -> str:
        """A short human-readable report, misses listed so they can be inspected."""
        lines = [
            "",
            f"Cases:            {self.total}",
            f"Recall@{self.top_k:<10} {self.recall_at_k:.1%}",
            f"MRR:              {self.mrr:.3f}",
        ]
        if self.section_recall is not None:
            lines.append(f"Section recall:   {self.section_recall:.1%}")
        if self.citation_precision is not None:
            lines.append(f"Citation prec.:   {self.citation_precision:.1%}")

        if self.misses:
            lines.append("")
            lines.append(f"Misses ({len(self.misses)}):")
            for outcome in self.misses:
                expected = ", ".join(outcome.case.expected_slugs)
                got = ", ".join(outcome.retrieved_slugs[:5]) or "nothing"
                lines.append(f"  [{outcome.case.id}] {outcome.case.question}")
                lines.append(f"      expected: {expected}")
                lines.append(f"      got:      {got}")
        return "\n".join(lines)

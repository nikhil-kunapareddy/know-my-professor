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
import random
from dataclasses import dataclass, field
from pathlib import Path

from core.llm.prompts import cited_numbers
from core.pipeline import NO_ANSWER
from shared.config import PEOPLE_NAMESPACE

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
    expected_slugs: tuple[str, ...] = ()
    expected_sections: tuple[str, ...] = ()
    #: Which Pinecone namespace answers this case. A query reads exactly one, so
    #: a course case run against the people namespace retrieves nothing and
    #: scores a miss that says nothing about retrieval quality. Defaults to the
    #: people corpus, which is what a case omitting the field has always meant —
    #: so the golden file needed no edit when people moved off the unnamed
    #: default partition.
    #:
    #: An explicit ``null`` is the third option and means "every namespace",
    #: i.e. the blended path ``/chat`` actually serves. Only cases that need
    #: BOTH corpora at once should use it; CLAUDE.md records that no case could
    #: express this before, which is why the blended path went unmeasured.
    namespace: str | None = PEOPLE_NAMESPACE
    #: True when the RIGHT behaviour is to decline. Without cases like these an
    #: evaluation can only reward retrieving more, so raising the relevance floor
    #: always looks free — which is exactly the question MIN_RETRIEVAL_SCORE
    #: needs answered. A negative case names no slugs.
    expect_no_answer: bool = False
    #: A ground-truth answer, in prose. Optional, and only the LLM-judged
    #: metrics use it (``evaluation.deepeval_eval``): every "was the answer right?"
    #: or "was the right context retrieved?" judgement needs something to
    #: compare against, and slugs alone cannot express a claim. Cases without
    #: one are still scored by every metric that does not need it.
    reference: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> EvalCase:
        missing = {"id", "question"} - set(raw)
        if missing:
            raise ValueError(f"golden case is missing {sorted(missing)}: {raw}")

        expect_no_answer = bool(raw.get("expect_no_answer", False))
        slugs = tuple(raw.get("expected_slugs", ()))
        # A case has to say what "right" looks like. Either it names who should
        # be retrieved, or it declares that nothing should be.
        if not slugs and not expect_no_answer:
            raise ValueError(
                f"golden case must give expected_slugs or set expect_no_answer: {raw}"
            )
        if slugs and expect_no_answer:
            raise ValueError(f"a no-answer case cannot also expect slugs: {raw}")

        return cls(
            id=raw["id"],
            question=raw["question"],
            expected_slugs=slugs,
            expected_sections=tuple(raw.get("expected_sections", ())),
            # An ABSENT key defaults to people, because the 60 original golden
            # cases predate namespaces and omit it. An EXPLICIT null is
            # different and means "search them all" -- the blended path /chat
            # actually takes, which no case could express before. The two are
            # told apart by key presence, since raw.get() collapses them.
            namespace=raw["namespace"] if "namespace" in raw else PEOPLE_NAMESPACE,
            expect_no_answer=expect_no_answer,
            reference=raw.get("reference", ""),
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


def sample_cases(cases: list[EvalCase], count: int, seed: int) -> list[EvalCase]:
    """A reproducible random subset, in the golden file's own order.

    Random rather than "the first N" because the golden file is grouped by what
    each block probes -- narrow topics, then broad, then sections, then the
    no-answer cases at the end. Taking a prefix would measure one block and
    never see a refusal. The seed is required, not optional: an experiment whose
    sample cannot be reproduced cannot be compared to the next one.

    Order is preserved so two runs with the same seed also report in the same
    order, which makes their outputs diffable.
    """
    if count >= len(cases):
        return list(cases)
    chosen = set(random.Random(seed).sample(range(len(cases)), count))
    return [case for i, case in enumerate(cases) if i in chosen]


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
    def declined(self) -> bool:
        """Did the system refuse to answer?

        Three spellings count, and the third is the one that matters. Exact
        equality with ``NO_ANSWER`` under-counts badly: asked something the
        corpus cannot support, the model reliably opens with the refusal the
        prompt asks for and then adds an honest caveat about the nearest
        material it did see — "I don't have that information in my data. The
        closest is Benjamin Gyori, who works on computational systems biology
        [4]." That is the behaviour worth having, not a failure, so a refusal is
        recognised by the prompt's own phrasing rather than by string identity.

        An empty result also counts, because a retrieval-only run never reaches
        the model: at a nonzero floor, nothing retrieved IS the refusal.
        """
        if not self.retrieved_ids:
            return True
        if not self.answer:
            return False
        normalised = " ".join(self.answer.lower().split())
        return self.answer == NO_ANSWER or (
            "i don't have" in normalised and "in my data" in normalised
        )

    @property
    def hit(self) -> bool:
        """Did the case get the outcome it asked for?

        For a normal case that means an expected professor was retrieved; for a
        no-answer case it means the system correctly declined.
        """
        if self.case.expect_no_answer:
            return self.declined
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
    #: The floor the run applied. Recorded because no-answer cases cannot fail
    #: at 0.0 — nothing is ever filtered, so every case retrieves something.
    min_score: float = 0.0

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def positives(self) -> list[CaseOutcome]:
        """Cases that expect an answer."""
        return [o for o in self.outcomes if not o.case.expect_no_answer]

    @property
    def negatives(self) -> list[CaseOutcome]:
        """Cases whose correct outcome is a refusal."""
        return [o for o in self.outcomes if o.case.expect_no_answer]

    @property
    def recall_at_k(self) -> float:
        """Share of answerable questions where an expected professor was retrieved.

        The headline number: if the right chunk never reaches the model, no
        amount of prompt work can produce a correct answer. Computed over
        positive cases only — averaging refusals into it would let a system that
        retrieves nothing score well on the half of the set that wants nothing.
        """
        scored = self.positives
        if not scored:
            return 0.0
        return sum(1 for o in scored if o.hit) / len(scored)

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank over answerable questions — how high the right professor lands."""
        scored = self.positives
        if not scored:
            return 0.0
        return sum(o.reciprocal_rank for o in scored) / len(scored)

    @property
    def no_answer_accuracy(self) -> float | None:
        """Share of no-answer cases the system correctly declined (None if there are none)."""
        scored = self.negatives
        if not scored:
            return None
        return sum(1 for o in scored if o.hit) / len(scored)

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
            f"Cases:            {self.total}"
            + (f"  ({len(self.positives)} answerable, {len(self.negatives)} no-answer)"
               if self.negatives else ""),
        ]
        # Printing "Recall@8 0.0%" for a run of nothing but refusals reads as a
        # catastrophe when it is an empty average.
        if self.positives:
            lines.append(f"Recall@{self.top_k:<10} {self.recall_at_k:.1%}")
            lines.append(f"MRR:              {self.mrr:.3f}")
        if self.section_recall is not None:
            lines.append(f"Section recall:   {self.section_recall:.1%}")
        if self.citation_precision is not None:
            lines.append(f"Citation prec.:   {self.citation_precision:.1%}")
        if self.no_answer_accuracy is not None:
            lines.append(f"Declined right:   {self.no_answer_accuracy:.1%}"
                         f"  ({len(self.negatives)} no-answer case(s))")
            if not any(o.answer for o in self.negatives):
                lines.append("  note: no answers were generated, so these were judged on retrieval")
                lines.append("        alone — a no-answer case is really decided by the generator.")
                lines.append("        Re-run with --generate to score refusals properly.")

        positive_misses = [o for o in self.misses if not o.case.expect_no_answer]
        if positive_misses:
            lines.append("")
            lines.append(f"Misses ({len(positive_misses)}):")
            for outcome in positive_misses:
                expected = ", ".join(outcome.case.expected_slugs)
                got = ", ".join(outcome.retrieved_slugs[:5]) or "nothing"
                lines.append(f"  [{outcome.case.id}] {outcome.case.question}")
                lines.append(f"      expected: {expected}")
                lines.append(f"      got:      {got}")

        wrongly_answered = [o for o in self.misses if o.case.expect_no_answer]
        if wrongly_answered:
            lines.append("")
            lines.append(f"Should have declined ({len(wrongly_answered)}):")
            for outcome in wrongly_answered:
                got = ", ".join(outcome.retrieved_slugs[:5]) or "nothing"
                lines.append(f"  [{outcome.case.id}] {outcome.case.question}")
                lines.append(f"      retrieved: {got}")
        return "\n".join(lines)

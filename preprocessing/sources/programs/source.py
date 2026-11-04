"""The programs source: catalog program pages -> requirement chunks.

The first source built on ``preprocessing/documents``. A program's tabs are
stored as IR ``Document``s by the scraper, and every section is cut by
``section_chunks`` along the page's own structure: one chunk per requirement
area ("Computer Science Requirements > Security Required Course"), tables
split between rows with their header repeated when an area outgrows the budget.

Programs are their OWN entities (``program-{path}``), in the ``courses``
namespace beside ``course-*``. Measured, not assumed (2026-09-29, the full crawl
of 1,200 programs / 14,650 chunks embedded and scored against the live top-11,
nothing upserted), because requirement tables name courses by code and title
and so could crowd course chunks out of the namespace's 11 slots:

- program questions: the right program ranks #1 every time, 0.85-0.87 against
  a best course chunk of 0.75-0.79;
- six course questions ("what does CS 3800 cover", "who teaches Discrete
  Structures", ...): no expected chunk lost a single rank. Program chunks never
  enter the top 11 of a "who teaches" question at all; on "prerequisites for
  Discrete Structures" 65 of the 73 chunks naming CS 1800 would fill ranks
  3-11, below the course's own chunk at rank 2.

A namespace of their own would have bought nothing measurable for a fourth
Pinecone query and ~11 more chunks in every /chat request.
"""

from __future__ import annotations

from shared.config import COURSES_NAMESPACE

from ...documents.config import DEFAULT_MAX_CHARS
from ...documents.ir import Document
from ..base import Chunk, SectionSpec, Source, section_chunks
from .config import ENTITY_PREFIX, LEVEL_LABELS

PROGRAM_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec("program_overview", "Program overview"),
    SectionSpec("program_requirements", "Program requirements"),
)


class ProgramSource(Source):
    name = "programs"
    prefix = "programs/"
    sections = PROGRAM_SECTIONS
    depends_on_entities = False
    namespace = COURSES_NAMESPACE

    def entity_id(self, record: dict) -> str | None:
        """``program-undergraduate-...-bscs``; the prefix keeps it apart from ``course-*``."""
        slug = record.get("slug")
        return f"{ENTITY_PREFIX}{slug}" if slug else None

    def is_ingestable(self, record: dict) -> bool:
        """A program with no requirement blocks has nothing to answer with."""
        return bool((record.get("program_requirements") or {}).get("blocks"))

    def header(self, record: dict) -> str:
        """``Computer Science, BSCS (Boston) — Undergraduate program``."""
        level = LEVEL_LABELS.get(record.get("level") or "", "Program")
        name = record.get("name") or record.get("slug") or ""
        return f"{name} — {level}"

    def to_chunks(self, record: dict) -> list[Chunk]:
        entity_id = self.entity_id(record)
        if not entity_id or not self.is_ingestable(record):
            return []

        base = {
            "program_name": record.get("name") or "",
            "program_level": record.get("level") or "",
            "catalog_college": record.get("catalog_college") or "",
            "url": record.get("url") or "",
        }
        chunks: list[Chunk] = []
        for spec in self.sections:
            stored = record.get(spec.key)
            if not stored:
                continue
            chunks.extend(
                section_chunks(
                    f"{entity_id}#{spec.key}",
                    self.header(record),
                    spec.label,
                    Document.from_dict(stored),
                    {**base, "section_type": spec.key},
                    max_chars=DEFAULT_MAX_CHARS,
                )
            )
        return chunks

"""The class-schedule source: Banner JSON -> who teaches each course.

Enriches a course entity defined by ``preprocessing/sources/courses``, exactly
as weblinks enrich a professor. It MUST mint entity ids the same way
``CourseSource`` does, or the enrichment attaches to nothing.

Lands in the same namespace as courses, not the professor one: "who teaches
Compilers?" is answered by retrieving the course, and the instructor chunk has
to be able to come back alongside the description chunk for that course.
"""

from __future__ import annotations

from ..base import (
    Chunk,
    SectionSpec,
    Source,
    content_hash,
    render_section,
)
from ..courses.source import COURSES_NAMESPACE, ENTITY_PREFIX

#: One section type. Keys must stay disjoint from every other source's; the
#: registry asserts that at import.
SECTION_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec("course_instructors", "Instructors"),
)


class SectionSource(Source):
    """Renders one ``sections/{slug}.json`` record into an instructors chunk."""

    name = "sections"
    prefix = "sections/"
    sections = SECTION_SECTIONS
    depends_on_entities = True
    namespace = COURSES_NAMESPACE

    def entity_id(self, record: dict) -> str | None:
        """The course entity this schedule belongs to — same scheme as courses."""
        slug = record.get("slug")
        return f"{ENTITY_PREFIX}{slug}" if slug else None

    def is_ingestable(self, record: dict) -> bool:
        """True only if some term actually named an instructor.

        A course can be listed with every section staffed as TBA. Embedding
        "CS 3800 Instructors:" with an empty body produces a chunk that matches
        instructor questions and answers none of them.
        """
        return any(t.get("instructors") for t in (record.get("terms") or []))

    def header(self, record: dict) -> str:
        """``CS 3800 — Theory of Computation``, matching the course chunk."""
        parts = [record.get("code") or "", record.get("title") or ""]
        return " — ".join(p for p in parts if p)

    def body(self, record: dict) -> str:
        """One line per term, naming that term's instructors.

        The term is written into the text rather than left in metadata because
        "who teaches X" and "who taught X last spring" are different questions,
        and only text reaches the model.
        """
        lines: list[str] = []
        for term in record.get("terms") or []:
            names = [n for n in (term.get("instructors") or []) if n]
            if not names:
                continue
            label = term.get("description") or term.get("code") or "Term"
            lines.append(f"- {label}: {', '.join(names)}")
        return "\n".join(lines)

    def to_chunks(self, record: dict) -> list[Chunk]:
        """One instructors chunk per course."""
        entity_id = self.entity_id(record)
        if not entity_id or not self.is_ingestable(record):
            return []

        spec = self.sections[0]
        text = render_section(self.header(record), spec.label, self.body(record))
        instructors = sorted(
            {n for t in (record.get("terms") or []) for n in (t.get("instructors") or []) if n}
        )
        return [
            Chunk(
                vector_id=f"{entity_id}#{spec.key}",
                text=text,
                metadata={
                    "course_code": record.get("code") or "",
                    "course_title": record.get("title") or "",
                    "subject": record.get("subject") or "",
                    # Filterable, so "what does <professor> teach" can be a
                    # lookup rather than a scan once intent routing exists.
                    "instructors": instructors,
                    "terms": [t.get("description") or t.get("code") or "" for t in (record.get("terms") or [])],
                    "url": record.get("url") or "",
                    "section_type": spec.key,
                    "text": text,
                    "content_hash": content_hash(text),
                },
            )
        ]

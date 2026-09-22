"""The course catalog source: parsed catalog JSON -> chunks.

Courses are their OWN entities, not an attribute of a professor. That is a
deliberate choice and the one real design question this source answers:

  - A course exists whether or not anyone we have a profile for teaches it.
    Measured on Fall 2026 CS, 95% of instructors match a professor already in
    the corpus -- but the other 5% are real people teaching real courses, and
    hanging courses off professors would silently drop them.
  - "Who teaches Compilers?" is a course-first question. Making the course the
    entity means the answer is found by looking up the course, not by scanning
    every professor for a mention.

So ``depends_on_entities`` is False and ``entity_id`` mints its own namespaced
id. ``preprocessing/sources/sections`` then enriches these entities with who
actually teaches them in the current term.
"""

from __future__ import annotations

from ..base import (
    Chunk,
    SectionSpec,
    Source,
    content_hash,
    render_section,
)

#: Sections this source emits. Deliberately ONE: prerequisites and NUpath
#: attributes are folded into the description chunk rather than given their own
#: vectors. "Prerequisite(s): CS 1800" is a short structured fact that embeds
#: poorly on its own, and a second section would add ~4,000 near-duplicate
#: low-value vectors for no retrieval gain.
COURSE_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec("course_description", "Description"),
)

#: Every course vector id starts with this, so a course can never collide with
#: a professor slug (``course-cs1800`` vs ``alex-suciu``).
ENTITY_PREFIX = "course-"

#: Pinecone namespace. Courses are a hard partition from people: a query reads
#: exactly one namespace, so ~6,500 course chunks can never crowd professors
#: out of the top-k.
COURSES_NAMESPACE = "courses"


class CourseSource(Source):
    """Renders one ``courses/{slug}.json`` record into a description chunk."""

    name = "courses"
    prefix = "courses/"
    sections = COURSE_SECTIONS
    depends_on_entities = False
    namespace = COURSES_NAMESPACE

    def entity_id(self, record: dict) -> str | None:
        """``course-cs1800``. Prefixed so courses and people share no id space."""
        slug = record.get("slug")
        return f"{ENTITY_PREFIX}{slug}" if slug else None

    def is_ingestable(self, record: dict) -> bool:
        """True if the catalog actually wrote a description for this course.

        A handful of catalog entries are placeholders with a title and nothing
        else; embedding those produces a vector that matches its own title and
        misleads every other query.
        """
        return bool((record.get("description") or "").strip())

    def header(self, record: dict) -> str:
        """``CS 1800 — Discrete Structures (4 Hours)``.

        Not ``base.header_line``: that renders ``Name (Title)`` for a person.
        A course reads better, and retrieves better, led by its code — which is
        how people actually refer to courses.
        """
        parts = [record.get("code") or "", record.get("title") or ""]
        head = " — ".join(p for p in parts if p)
        credits = (record.get("credits") or "").strip()
        return f"{head} ({credits})" if credits else head

    def body(self, record: dict) -> str:
        """Description followed by any requisite/attribute lines."""
        lines = [(record.get("description") or "").strip()]
        lines.extend(r for r in (record.get("requisites") or []) if r)
        return "\n".join(line for line in lines if line)

    def to_chunks(self, record: dict) -> list[Chunk]:
        """One description chunk per course."""
        entity_id = self.entity_id(record)
        if not entity_id or not self.is_ingestable(record):
            return []

        spec = self.sections[0]
        text = render_section(self.header(record), spec.label, self.body(record))
        return [
            Chunk(
                vector_id=f"{entity_id}#{spec.key}",
                text=text,
                metadata={
                    "course_code": record.get("code") or "",
                    "course_title": record.get("title") or "",
                    "subject": record.get("subject") or "",
                    "credits": record.get("credits") or "",
                    "url": record.get("url") or "",
                    "section_type": spec.key,
                    "text": text,
                    "content_hash": content_hash(text),
                },
            )
        ]

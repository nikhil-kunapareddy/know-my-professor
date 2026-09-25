"""The publications source: OpenAlex works -> chunks on the professor.

Enriches a professor the profiles source defined, exactly as weblinks do, so it
mints entity ids through the same ``..entities.entity_key`` and lives in the
people namespace. Two sections per professor:

- ``publication_themes`` -- Claude's short account of what the recent work is
  about. Written from titles and abstracts only; it exists because a list of
  titles embeds poorly against "who works on X?".
- ``publications`` -- the recent works themselves, verbatim from OpenAlex, so
  "what has X published lately?" is answered from citable titles rather than a
  paraphrase.

Keys must differ from weblinks' ``recent_publications``, which is a list scraped
off the professor's own website; the registry rejects a collision at import.
"""

from __future__ import annotations

from shared.config import PEOPLE_NAMESPACE

from ..base import Chunk, SectionSpec, Source, content_hash, header_line, render_section
from ..entities import college_of, entity_key

PUBLICATIONS_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec("publication_themes", "Research themes in recent publications"),
    SectionSpec("publications", "Recent publications (OpenAlex)"),
)


def format_work(work: dict) -> str:
    """``Title (2024, Nature Methods)``; the venue is omitted when unknown."""
    detail = ", ".join(str(p) for p in (work.get("year"), work.get("venue")) if p)
    title = work.get("title") or ""
    return f"{title} ({detail})" if detail else title


class PublicationsSource(Source):
    """Renders one ``publications/{entity}.json`` record into up to two chunks."""

    name = "publications"
    prefix = "publications/"
    sections = PUBLICATIONS_SECTIONS
    depends_on_entities = True
    namespace = PEOPLE_NAMESPACE

    def entity_id(self, record: dict) -> str | None:
        """The profile entity these works belong to -- same scheme as profiles."""
        return entity_key(record)

    def is_ingestable(self, record: dict) -> bool:
        """Only if a candidate was accepted as this person and had recent works.

        Records are also written for professors whose candidates were all
        rejected, so the next run can skip them without paying the model again.
        """
        return bool(record.get("works"))

    def to_chunks(self, record: dict) -> list[Chunk]:
        slug = record.get("slug")
        if not slug or not self.is_ingestable(record):
            return []
        entity_id = self.entity_id(record)

        name = record.get("professor_name") or slug
        title = record.get("professor_title") or ""
        header = header_line(name, title)
        bodies = {
            "publication_themes": record.get("themes") or "",
            "publications": [format_work(w) for w in record["works"] if w.get("title")],
        }

        chunks: list[Chunk] = []
        for spec in self.sections:
            value = bodies[spec.key]
            if not value:
                continue
            text = render_section(header, spec.label, value)
            chunks.append(
                Chunk(
                    vector_id=f"{entity_id}#{spec.key}",
                    text=text,
                    metadata={
                        "professor_slug": slug,
                        "professor_name": name,
                        "professor_title": title,
                        "college": college_of(record),
                        "url": record.get("url") or "",
                        "section_type": spec.key,
                        "text": text,
                        "content_hash": content_hash(text),
                    },
                )
            )
        return chunks

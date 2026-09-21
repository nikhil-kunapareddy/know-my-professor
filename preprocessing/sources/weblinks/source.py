"""The faculty-website enrichment source: weblinks JSON -> chunks.

Enriches a professor already defined by the profiles source, so citations point
at the professor's own site rather than the directory page.

Because it hangs off another source's entities, it MUST mint entity ids the same
way -- hence the shared ``..entities.entity_key``. Computing them differently
would attach a College of Science professor's website to the Khoury professor
with the same slug, which for the 9 people holding joint appointments is not
hypothetical.
"""

from __future__ import annotations

from ..base import (
    Chunk,
    SectionSpec,
    Source,
    content_hash,
    fallback_label,
    header_line,
    render_section,
)
from ..entities import college_of, entity_key

#: Section types the weblinks extractor produces. Keys must stay disjoint from
#: every other source's keys; the registry asserts that at import time.
WEBLINKS_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec("website_summary", "Website summary"),
    SectionSpec("current_projects", "Current projects"),
    SectionSpec("recent_publications", "Recent publications"),
    SectionSpec("students_or_lab_members", "Students and lab members"),
    SectionSpec("recent_news", "Recent news"),
)


class WeblinksSource(Source):
    """Renders one ``weblinks/{slug}.json`` record into per-section chunks."""

    name = "weblinks"
    prefix = "weblinks/"
    sections = WEBLINKS_SECTIONS
    depends_on_entities = True

    def entity_id(self, record: dict) -> str | None:
        """The profile entity this website enriches -- same scheme as profiles."""
        return entity_key(record)

    def to_chunks(self, record: dict) -> list[Chunk]:
        """One chunk per non-empty extracted section."""
        slug = record.get("slug")
        if not slug:
            return []
        entity_id = self.entity_id(record)

        name = record.get("professor_name") or slug
        title = record.get("professor_title") or ""
        header = header_line(name, title)
        labels = self.labels()

        chunks: list[Chunk] = []
        for section in record.get("sections", []):
            section_type = section.get("section_type")
            value = section.get("text")
            if not section_type or not value:
                continue
            label = labels.get(section_type) or fallback_label(section_type)
            text = render_section(header, label, value)
            chunks.append(
                Chunk(
                    vector_id=f"{entity_id}#{section_type}",
                    text=text,
                    metadata={
                        "professor_slug": slug,
                        "professor_name": name,
                        "professor_title": title,
                        # Filterable alongside the profile chunks it enriches.
                        "college": college_of(record),
                        # Each section carries the page it came from, so a
                        # citation links to the real source.
                        "url": section.get("source_url") or "",
                        "section_type": section_type,
                        "text": text,
                        "content_hash": content_hash(text),
                    },
                )
            )
        return chunks

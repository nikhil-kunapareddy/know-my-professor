"""The faculty directory profile source: scraped profile JSON -> chunks.

Profiles define the entity space (one professor per entity id); every other
source hangs off the ids this one produces.

Records from more than one college share this source, because a College of
Science biography is still a biography -- minting per-college section keys
would fork the taxonomy for no semantic gain, and the registry rejects a second
source reusing these keys anyway. What must stay unique is the entity id, which
``entity_id`` namespaces by college.
"""

from __future__ import annotations

from shared.config import PEOPLE_NAMESPACE

from ..base import (
    Chunk,
    SectionSpec,
    Source,
    content_hash,
    header_line,
    render_section,
)
from ..entities import college_of, entity_key

#: Profile accordion sections, in the order they are emitted.
PROFILE_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec("biography", "Biography"),
    SectionSpec("research_interests", "Research interests"),
    SectionSpec("education", "Education"),
    SectionSpec("areas_of_interest", "Areas of interest"),
    SectionSpec("labs_and_groups", "Labs and groups"),
    SectionSpec("projects", "Projects"),
)


class ProfileSource(Source):
    """Renders one ``profiles/{slug}.json`` record into per-section chunks."""

    name = "profiles"
    prefix = "profiles/"
    sections = PROFILE_SECTIONS
    depends_on_entities = False
    namespace = PEOPLE_NAMESPACE

    def entity_id(self, record: dict) -> str | None:
        """Globally unique id for a professor -- see ``..entities.entity_key``."""
        return entity_key(record)

    def is_ingestable(self, record: dict) -> bool:
        """True if a profile has enough content to be worth ingesting.

        The Khoury directory calls a lot of people "faculty" (staff, PhD
        students, postdocs); many have an empty profile. Requiring one of these
        three fields keeps the corpus to entries that can actually answer a
        question.
        """
        return bool(
            record.get("biography")
            or record.get("research_interests")
            or record.get("areas_of_interest")
        )

    def to_chunks(self, record: dict) -> list[Chunk]:
        """One chunk per populated profile section."""
        slug = record.get("slug")
        if not slug:
            return []
        entity_id = self.entity_id(record)

        base_metadata = {
            "professor_slug": slug,
            "professor_name": record.get("name") or "",
            "professor_title": record.get("title") or "",
            "url": record.get("url") or "",
            "campuses": record.get("campuses") or [],
            "roles": record.get("roles") or [],
            "areas_of_interest": record.get("areas_of_interest") or [],
            # Always present, even for records written before the field existed
            # -- an upsert replaces metadata wholesale, so emitting it
            # conditionally would let a later re-ingest silently strip the
            # value the backfill put on the index.
            "college": college_of(record),
        }
        # The header falls back to the slug, but the metadata name does not --
        # keeping both behaviours as-is preserves existing content hashes.
        header = header_line(
            record.get("name") or slug or "Unknown",
            record.get("title") or "",
        )

        chunks: list[Chunk] = []
        for spec in self.sections:
            value = record.get(spec.key)
            if not value:
                continue
            text = render_section(header, spec.label, value)
            chunks.append(
                Chunk(
                    vector_id=f"{entity_id}#{spec.key}",
                    text=text,
                    metadata={
                        **base_metadata,
                        "section_type": spec.key,
                        "text": text,
                        "content_hash": content_hash(text),
                    },
                )
            )
        return chunks

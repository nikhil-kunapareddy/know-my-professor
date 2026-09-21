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

from ..base import (
    Chunk,
    SectionSpec,
    Source,
    content_hash,
    header_line,
    render_section,
)
from .config import DEFAULT_COLLEGE

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

    def entity_id(self, record: dict) -> str | None:
        """Globally unique id for a professor: ``{college}-{slug}``.

        Khoury is the exception and stays a bare ``{slug}``. Its vectors predate
        multi-college support, and an id is a Pinecone primary key -- re-minting
        the Khoury ids would write 2,221 new vectors and orphan the originals
        rather than update them. Records with no ``college`` field are Khoury by
        construction (nothing else wrote profiles before the field existed).
        """
        slug = record.get("slug")
        if not slug:
            return None
        college = record.get("college") or DEFAULT_COLLEGE
        return slug if college == DEFAULT_COLLEGE else f"{college}-{slug}"

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
        }
        # Added only when the record carries it, which keeps pre-multi-college
        # Khoury records -- and the golden fixtures pinning them -- unchanged.
        # A filter therefore reads "absent means khoury"; run
        # preprocessing.ingest.backfill_college_metadata to make it explicit.
        if record.get("college"):
            base_metadata["college"] = record["college"]
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

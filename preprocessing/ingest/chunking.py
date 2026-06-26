"""Turn profile + weblinks records into per-section Chunks ready for embedding."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from shared.config import ENRICHMENT_LABELS, SECTION_FIELDS


@dataclass
class Chunk:
    """One embeddable unit: a vector ID, its text, and the metadata to store."""

    vector_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class Chunker:
    """Renders profile/enrichment records into per-section Chunks.

    Vector IDs follow ``{slug}#{section_type}``; profile section names and
    enrichment section types are disjoint, so the two sources never collide.
    """

    @staticmethod
    def content_hash(text: str) -> str:
        """Stable hash of a chunk's text, used to skip unchanged chunks on re-ingest."""
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def is_substantive(profile: dict) -> bool:
        """True if a profile has enough content to be worth ingesting."""
        return bool(
            profile.get("biography")
            or profile.get("research_interests")
            or profile.get("areas_of_interest")
        )

    def profile_to_chunks(self, profile: dict) -> list[Chunk]:
        """One chunk per populated profile section."""
        slug = profile.get("slug")
        if not slug:
            return []

        base_metadata = {
            "professor_slug": slug,
            "professor_name": profile.get("name") or "",
            "professor_title": profile.get("title") or "",
            "url": profile.get("url") or "",
            "campuses": profile.get("campuses") or [],
            "roles": profile.get("roles") or [],
            "areas_of_interest": profile.get("areas_of_interest") or [],
        }

        chunks: list[Chunk] = []
        for field_name, label in SECTION_FIELDS:
            text = self._render_section(profile, field_name, label)
            if text is None:
                continue
            chunks.append(
                Chunk(
                    vector_id=f"{slug}#{field_name}",
                    text=text,
                    metadata={
                        **base_metadata,
                        "section_type": field_name,
                        "text": text,
                        "content_hash": self.content_hash(text),
                    },
                )
            )
        return chunks

    def enrichment_to_chunks(self, enriched: dict) -> list[Chunk]:
        """Turn a weblinks/{slug}.json record into per-section Chunks.

        Each section carries its own ``source_url`` (the faculty website it came
        from) as the chunk's ``url`` metadata, so citations point at the real
        source rather than the Khoury profile page.
        """
        slug = enriched.get("slug")
        if not slug:
            return []
        name = enriched.get("professor_name") or slug
        title = enriched.get("professor_title") or ""
        header = self._header(name, title)

        chunks: list[Chunk] = []
        for section in enriched.get("sections", []):
            section_type = section.get("section_type")
            value = section.get("text")
            if not section_type or not value:
                continue
            label = ENRICHMENT_LABELS.get(section_type, section_type.replace("_", " ").title())
            text = f"{header}\n{label}:\n{self._format_body(value)}"
            chunks.append(
                Chunk(
                    vector_id=f"{slug}#{section_type}",
                    text=text,
                    metadata={
                        "professor_slug": slug,
                        "professor_name": name,
                        "professor_title": title,
                        "url": section.get("source_url") or "",
                        "section_type": section_type,
                        "text": text,
                        "content_hash": self.content_hash(text),
                    },
                )
            )
        return chunks

    # --- rendering helpers -------------------------------------------------

    @staticmethod
    def _format_body(value) -> str:
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value)
        return str(value)

    @staticmethod
    def _header(name: str, title: str) -> str:
        return f"{name}" + (f" ({title})" if title else "")

    def _render_section(self, profile: dict, field_name: str, label: str) -> str | None:
        value = profile.get(field_name)
        if not value:
            return None
        name = profile.get("name") or profile.get("slug") or "Unknown"
        header = self._header(name, profile.get("title") or "")
        return f"{header}\n{label}:\n{self._format_body(value)}"

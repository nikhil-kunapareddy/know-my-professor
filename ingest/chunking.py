"""Turn a profile dict into per-section Chunks ready for embedding."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import SECTION_FIELDS


@dataclass
class Chunk:
    vector_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def is_substantive(profile: dict) -> bool:
    return bool(
        profile.get("biography")
        or profile.get("research_interests")
        or profile.get("areas_of_interest")
    )


def render_section(profile: dict, field_name: str, label: str) -> str | None:
    value = profile.get(field_name)
    if not value:
        return None
    name = profile.get("name") or profile.get("slug") or "Unknown"
    title = profile.get("title") or ""
    header = f"{name}" + (f" ({title})" if title else "")

    if isinstance(value, list):
        body = "\n".join(f"- {item}" for item in value)
    else:
        body = str(value)
    return f"{header}\n{label}:\n{body}"


def profile_to_chunks(profile: dict) -> list[Chunk]:
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
        text = render_section(profile, field_name, label)
        if text is None:
            continue
        chunks.append(
            Chunk(
                vector_id=f"{slug}#{field_name}",
                text=text,
                metadata={**base_metadata, "section_type": field_name, "text": text},
            )
        )
    return chunks

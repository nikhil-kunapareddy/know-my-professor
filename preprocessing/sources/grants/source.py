"""The grants source: NSF and NIH awards -> one funding chunk per professor.

Enriches a professor the profiles source defined, so it mints entity ids
through ``..entities.entity_key`` and lives in the people namespace, like
weblinks and publications.

Each award line carries its agency, dates, status and amount in the TEXT, not
only in metadata: "who has active NSF funding for X" turns on exactly those
words, and only text reaches the model. Status is fixed when the record is
written, never computed at render time, so the rendering stays a pure function
of the record (tests/test_chunk_golden.py pins it).
"""

from __future__ import annotations

from shared.config import PEOPLE_NAMESPACE

from ..base import Chunk, SectionSpec, Source, content_hash, header_line, render_section
from ..entities import college_of, entity_key

GRANTS_SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec("research_funding", "Research funding (NSF and NIH awards)"),
)


def format_award(award: dict) -> str:
    """``NSF award 2552085 (Human-Centered Computing), Co-PI, 2026-07 to 2029-09,
    active, $845,345 total: "Title". Summary.``"""
    program = f" ({award['program']})" if award.get("program") else ""
    parts = [
        f"{award.get('agency', '')} award {award.get('award_id', '')}{program}",
        award.get("role") or "",
        f"{(award.get('start') or '?')[:7]} to {(award.get('end') or '?')[:7]}",
        award.get("status") or "",
    ]
    if award.get("amount") is not None:
        basis = award.get("amount_basis") or "total"
        parts.append(f"${award['amount']:,} {'total' if basis == 'total' else 'in ' + basis}")
    line = ", ".join(p for p in parts if p) + f': "{award.get("title", "")}"'
    summary = (award.get("summary") or "").strip()
    return f"{line}. {summary}" if summary else line


class GrantsSource(Source):
    """Renders one ``grants/{entity}.json`` record into a research-funding chunk."""

    name = "grants"
    prefix = "grants/"
    sections = GRANTS_SECTIONS
    depends_on_entities = True
    namespace = PEOPLE_NAMESPACE

    def entity_id(self, record: dict) -> str | None:
        """The profile entity these awards belong to -- same scheme as profiles."""
        return entity_key(record)

    def is_ingestable(self, record: dict) -> bool:
        """False once every award has aged out of the window; the record stays
        so the runner can tell "had grants" from "never had any"."""
        return bool(record.get("awards"))

    def to_chunks(self, record: dict) -> list[Chunk]:
        slug = record.get("slug")
        if not slug or not self.is_ingestable(record):
            return []

        name = record.get("professor_name") or slug
        title = record.get("professor_title") or ""
        spec = self.sections[0]
        awards = record["awards"]
        text = render_section(header_line(name, title), spec.label, [format_award(a) for a in awards])
        return [
            Chunk(
                vector_id=f"{self.entity_id(record)}#{spec.key}",
                text=text,
                metadata={
                    "professor_slug": slug,
                    "professor_name": name,
                    "professor_title": title,
                    "college": college_of(record),
                    # A chunk holds several awards but a citation links one
                    # page: the most recent award's, which leads the list.
                    "url": awards[0].get("url") or "",
                    "agencies": sorted({a.get("agency") or "" for a in awards} - {""}),
                    "section_type": spec.key,
                    "text": text,
                    "content_hash": content_hash(text),
                },
            )
        ]

"""How a professor's entity id is minted, shared by every source that uses one.

A vector id is ``{entity}#{section}``, and ``{entity}`` must be unique across
every college the corpus covers -- 9 professors hold joint appointments and
appear in two directories, so a bare slug would have one record silently
overwrite the other.

This lives outside any single source because *agreeing* on the id is the whole
point: ``ProfileSource`` defines the entity space and ``WeblinksSource`` hangs
enrichment off it, so if they computed ids differently the enrichment would
attach to the wrong professor (or to nothing).
"""

from __future__ import annotations

#: The college whose ids stay unprefixed. Khoury's vectors predate multi-college
#: support, and an id is a Pinecone primary key -- re-minting them would write
#: new vectors and orphan the originals rather than update them.
DEFAULT_COLLEGE = "khoury"


def college_of(record: dict) -> str:
    """The college a record belongs to.

    Records written before the field existed are Khoury by construction:
    nothing else wrote into this corpus at the time.
    """
    return record.get("college") or DEFAULT_COLLEGE


def entity_key(record: dict) -> str | None:
    """``{college}-{slug}``, or a bare ``{slug}`` for the default college."""
    slug = record.get("slug")
    if not slug:
        return None
    college = college_of(record)
    return slug if college == DEFAULT_COLLEGE else f"{college}-{slug}"

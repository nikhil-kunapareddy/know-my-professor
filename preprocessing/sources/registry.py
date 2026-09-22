"""The set of data sources ingest walks.

Adding a corpus means writing a ``Source`` and appending it to ``SOURCES``.
Nothing in ``preprocessing/ingest`` names a source, so no other file changes.

Import-time validation enforces the invariants that used to be comments: source
names, GCS prefixes, and section keys must all be unique. A duplicate section key
across two sources would mint the same ``{entity}#{key}`` vector ID, which
Pinecone resolves by silently overwriting -- a data-loss bug with no error, so it
is worth failing at import.
"""

from __future__ import annotations

from collections import Counter

from .base import Chunk, Source
from .courses.source import CourseSource
from .profiles.source import ProfileSource
from .schedule.source import ScheduleSource
from .weblinks.source import WeblinksSource

#: Order matters only in that entity-defining sources must be able to run
#: first; ``entity_sources()`` handles that, so this is just reading order.
SOURCES: tuple[Source, ...] = (
    ProfileSource(),
    WeblinksSource(),
    CourseSource(),
    ScheduleSource(),
)


def _validate(sources: tuple[Source, ...]) -> None:
    """Fail loudly on duplicate names, prefixes, or section keys."""
    for label, values in (
        ("source name", [s.name for s in sources]),
        ("GCS prefix", [s.prefix for s in sources]),
        ("section key", [spec.key for s in sources for spec in s.sections]),
    ):
        duplicates = sorted({v for v, n in Counter(values).items() if n > 1})
        if duplicates:
            raise ValueError(f"duplicate {label}(s) across sources: {duplicates}")

    if not any(not s.depends_on_entities for s in sources):
        raise ValueError("at least one source must define entities (depends_on_entities=False)")

    # A dependent source is scoped to the entity ids an entity-defining source
    # produced, and ids are unique only within a namespace. A dependent source
    # alone in its namespace can therefore never match anything -- it would
    # ingest zero chunks silently, which is the worst way to find out.
    entity_namespaces = {s.namespace for s in sources if not s.depends_on_entities}
    orphaned = sorted(
        {s.name for s in sources if s.depends_on_entities and s.namespace not in entity_namespaces}
    )
    if orphaned:
        raise ValueError(
            f"dependent source(s) {orphaned} are in a namespace with no entity-defining "
            f"source; their chunks could never match an entity"
        )


_validate(SOURCES)

_BY_NAME: dict[str, Source] = {s.name: s for s in SOURCES}


def get_source(name: str) -> Source:
    """The registered source called ``name``."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown source {name!r}; registered: {sorted(_BY_NAME)}") from None


def chunks_for(name: str, record: dict) -> list[Chunk]:
    """Render one record through the named source. Convenience for tests/scripts."""
    return get_source(name).to_chunks(record)


def entity_sources() -> tuple[Source, ...]:
    """Sources that define entities; these set the id space and honour --limit."""
    return tuple(s for s in SOURCES if not s.depends_on_entities)


def dependent_sources() -> tuple[Source, ...]:
    """Sources that only apply to entities another source already defined."""
    return tuple(s for s in SOURCES if s.depends_on_entities)


def all_section_types() -> tuple[str, ...]:
    """Every section key any registered source can emit."""
    return tuple(spec.key for s in SOURCES for spec in s.sections)


def section_labels() -> dict[str, str]:
    """section_type -> human label, across all sources."""
    return {spec.key: spec.label for s in SOURCES for spec in s.sections}

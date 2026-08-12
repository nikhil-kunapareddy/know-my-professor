"""The contract between a data source and ingest.

A *source* owns one GCS prefix and knows how to turn one stored record into
``Chunk`` objects. Ingest never names a source: it walks the registry, so adding
a corpus (courses, publications, ...) is a new module plus one registry entry.

``Chunk`` and the rendering helpers live here rather than in ``ingest`` so the
dependency runs one way -- ingest imports sources, never the reverse.

RENDERING IS A WIRE FORMAT. A chunk's text is hashed into ``content_hash``, and
that hash is what tells ingest whether to re-embed. Changing how text is rendered
invalidates every affected vector and costs a full re-ingest, so the helpers
below are deliberately boring and covered by tests/test_chunk_golden.py.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One embeddable unit: a vector ID, its text, and the metadata to store."""

    vector_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SectionSpec:
    """One section type a source can emit.

    ``key`` is the suffix in the ``{entity}#{key}`` vector ID and the stored
    ``section_type``; ``label`` is the human heading rendered into the text. Keys
    must be unique across ALL sources -- two sources sharing a key would produce
    colliding vector IDs, which Pinecone resolves by silent overwrite. The
    registry asserts uniqueness at import time.
    """

    key: str
    label: str


# --- rendering helpers (shared by every source) ----------------------------


def content_hash(text: str) -> str:
    """Stable hash of a chunk's text, used to skip unchanged chunks on re-ingest."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def format_body(value) -> str:
    """Render a section body: lists become bullets, everything else is str()."""
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def header_line(name: str, title: str) -> str:
    """The ``Name (Title)`` line every chunk opens with."""
    return f"{name}" + (f" ({title})" if title else "")


def render_section(header: str, label: str, value) -> str:
    """Assemble the chunk text. The one place the on-the-wire layout is decided."""
    return f"{header}\n{label}:\n{format_body(value)}"


def fallback_label(section_type: str) -> str:
    """Heading for a section type no source declared (forward compatibility)."""
    return section_type.replace("_", " ").title()


class Source(ABC):
    """A corpus of records under one GCS prefix that renders into chunks."""

    #: Registry key, also used in CLI output.
    name: str
    #: GCS prefix the records live under, e.g. ``"profiles/"``.
    prefix: str
    #: Section types this source can emit.
    sections: tuple[SectionSpec, ...] = ()
    #: True if records only make sense alongside an entity defined by another
    #: source (weblinks enrich a professor). False for a source that defines its
    #: own entities (profiles, and a future standalone courses corpus).
    depends_on_entities: bool = False

    def entity_id(self, record: dict) -> str | None:
        """The entity a record belongs to. Default: the professor slug."""
        return record.get("slug")

    def is_ingestable(self, record: dict) -> bool:
        """Whether a record carries enough content to be worth embedding."""
        return True

    @abstractmethod
    def to_chunks(self, record: dict) -> list[Chunk]:
        """Render one stored record into zero or more chunks."""
        ...

    def labels(self) -> dict[str, str]:
        """section_type -> human label for this source."""
        return {spec.key: spec.label for spec in self.sections}

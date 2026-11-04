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
from collections.abc import Sequence
from dataclasses import dataclass, field

from shared.config import PART_SEPARATOR

from ..documents.chunking import chunk_blocks
from ..documents.ir import Block, BlockKind, Document


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


# --- structure-aware sections (preprocessing/documents) ----------------------

# PART_SEPARATOR (shared.config) joins a section's id to its part number:
# ``{entity}#{section}@2``. Section keys may not contain it; the registry checks.


def part_vector_id(base_id: str, part: int) -> str:
    """The vector id of part ``part`` (1-based) of the section ``base_id``.

    Part 1 keeps the section's own id. So a section that fits in one chunk --
    every section in the index today -- keeps the id it already has, and moving
    a source onto ``section_chunks`` orphans nothing that did not grow.
    """
    return base_id if part == 1 else f"{base_id}{PART_SEPARATOR}{part}"


def _as_blocks(content) -> list[Block]:
    """Section content as blocks: a string is a paragraph, a list is a list."""
    if isinstance(content, Document):
        return list(content.blocks)
    if isinstance(content, str):
        return [Block(BlockKind.PARAGRAPH, text=content)] if content else []
    if isinstance(content, Sequence) and all(isinstance(b, Block) for b in content):
        return list(content)
    if isinstance(content, Sequence):
        return [Block(BlockKind.LIST, items=tuple(f"{item}" for item in content))] if content else []
    return [Block(BlockKind.PARAGRAPH, text=str(content))]


def section_chunks(
    base_id: str,
    header: str,
    label: str,
    content: str | Sequence | Document,
    metadata: dict,
    *,
    max_chars: int | None = None,
) -> list[Chunk]:
    """One section as chunks, cut along its structure when it outgrows ``max_chars``.

    ``content`` may be a string, a list, or blocks / a ``Document`` from
    ``preprocessing.documents``. With ``max_chars=None`` -- or whenever the
    section fits -- this returns exactly the one chunk ``render_section`` would:
    same id, same text, same ``content_hash``. When it does not fit:

    - parts after the first get ids ``{base_id}@2``, ``@3``, ... and a ``part``
      metadata field;
    - a later part's label reads ``{label} (continued)``;
    - a part under document headings has them in its label
      (``Label > Heading > Subheading``) and in ``heading_path`` metadata;
    - a part from paginated input carries ``page_start``/``page_end``. Two
      numbers, because Pinecone metadata lists may only hold strings.

    Every chunk's metadata carries its own ``text``, as every source's does:
    it is what the prompt shows the model.

    A section that shrinks leaves its old tail parts behind in Pinecone; ingest's
    ``--prune`` is what removes them.
    """
    pieces = chunk_blocks(_as_blocks(content), max_chars)
    chunks = []
    for part, piece in enumerate(pieces, 1):
        heading = " > ".join((label, *piece.path)) + (" (continued)" if piece.continued else "")
        text = render_section(header, heading, piece.body)
        # ``text`` is set per part, never inherited: the prompt builds the
        # model's context from ``metadata["text"]``, so a part carrying the
        # caller's text would show the model some other part's content.
        meta = {**metadata, "text": text, "content_hash": content_hash(text)}
        if len(pieces) > 1:
            meta["part"] = part
        if piece.path:
            meta["heading_path"] = " > ".join(piece.path)
        if piece.pages:
            meta["page_start"], meta["page_end"] = piece.pages[0], piece.pages[-1]
        chunks.append(Chunk(vector_id=part_vector_id(base_id, part), text=text, metadata=meta))
    return chunks


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
    #: own entities (profiles, courses).
    depends_on_entities: bool = False
    #: Pinecone namespace this source's vectors live in. Every registered source
    #: names one explicitly; ``None`` (the index's unnamed default partition) is
    #: only the ABC's fallback, and nothing should be left there -- an unnamed
    #: partition cannot be told apart from "the author forgot".
    #:
    #: A namespace is a hard partition: a query names exactly one, so chunks in
    #: different namespaces NEVER compete for the same top-k slots. That is the
    #: point. ~10,900 course chunks in with ~4,100 people chunks would make
    #: courses the majority of the corpus, and "who works on machine learning?"
    #: would start returning syllabi instead of people.
    namespace: str | None = None
    #: For a dependent source: the namespace holding the entities it enriches,
    #: when that is not its own. ``None`` means its own ``namespace``.
    #: Publications and grants write to ``research`` but enrich professors in
    #: ``people``, so their chunks get their own slots instead of competing
    #: with bios for the people namespace's top-k.
    entity_namespace: str | None = None

    def entity_scope(self) -> str | None:
        """The namespace whose entity ids this source's records must match."""
        return self.entity_namespace or self.namespace

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

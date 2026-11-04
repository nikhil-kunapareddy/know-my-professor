"""The canonical document IR: typed blocks, whatever the input format was.

Every tier writes the same shape, so everything downstream of parsing -- the
chunker, the review queue, the stored record -- is written once rather than per
format. A heading read off a PDF's font sizes, a heading from an HTML ``<h2>``
and a heading OCR'd off a scan are the same ``Block``.

The IR is stored, not just passed around. A scraper parses at scrape time and
writes ``Document.to_dict()`` into its GCS record; ingest rebuilds it with
``Document.from_dict()`` and chunks it. That split is the same one every source
already has (scrape -> JSON in GCS -> ingest), and it keeps the heavy parsing
dependencies (pdfplumber, tesseract) out of the ingest image entirely: this
module and ``chunking`` import nothing beyond the standard library.

``IR_VERSION`` is written into every stored document. Bump it on any change a
stored record could not be read back through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

IR_VERSION = 1


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"


class Tier(IntEnum):
    """Which parse tier produced a block, cheapest first.

    ``LAYOUT`` reads structure the input already carries -- a DOM, a PDF text
    layer -- so it is exact and free. ``OCR`` reads pixels and carries a
    confidence. ``VISION`` asks a vision-language model to describe a figure,
    which is the only tier that can say what a chart *shows*.
    """

    LAYOUT = 0
    OCR = 1
    VISION = 2


@dataclass(frozen=True)
class Block:
    """One unit of document structure.

    Which fields carry content depends on ``kind``: a heading or paragraph uses
    ``text``, a list uses ``items``, a table uses ``rows`` (the first
    ``header_rows`` of which are its header), and a figure uses ``text`` for its
    description. ``page`` and ``bbox`` exist only for paginated input; ``bbox``
    is ``(x0, top, x1, bottom)`` in PDF points, top-left origin.
    """

    kind: BlockKind
    text: str = ""
    level: int | None = None
    items: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    header_rows: int = 0
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    tier: Tier = Tier.LAYOUT
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.kind is BlockKind.HEADING and not (self.level and 1 <= self.level <= 6):
            raise ValueError(f"a heading needs a level in 1..6, got {self.level!r}")
        if not 0 <= self.header_rows <= len(self.rows):
            raise ValueError(f"header_rows={self.header_rows} but the table has {len(self.rows)} row(s)")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def is_empty(self) -> bool:
        """True when the block carries nothing worth indexing."""
        if self.kind is BlockKind.LIST:
            return not any(item.strip() for item in self.items)
        if self.kind is BlockKind.TABLE:
            return not any(cell.strip() for row in self.rows for cell in row)
        return not self.text.strip()

    def to_dict(self) -> dict:
        """JSON-safe form. Fields at their default are omitted to keep records small."""
        out: dict = {"kind": self.kind.value}
        if self.text:
            out["text"] = self.text
        if self.level is not None:
            out["level"] = self.level
        if self.items:
            out["items"] = list(self.items)
        if self.rows:
            out["rows"] = [list(row) for row in self.rows]
        if self.header_rows:
            out["header_rows"] = self.header_rows
        if self.page is not None:
            out["page"] = self.page
        if self.bbox is not None:
            out["bbox"] = [round(v, 2) for v in self.bbox]
        if self.tier is not Tier.LAYOUT:
            out["tier"] = int(self.tier)
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 4)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> Block:
        bbox = data.get("bbox")
        return cls(
            kind=BlockKind(data["kind"]),
            text=data.get("text", ""),
            level=data.get("level"),
            items=tuple(data.get("items", ())),
            rows=tuple(tuple(row) for row in data.get("rows", ())),
            header_rows=data.get("header_rows", 0),
            page=data.get("page"),
            bbox=tuple(bbox) if bbox is not None else None,
            tier=Tier(data.get("tier", Tier.LAYOUT)),
            confidence=data.get("confidence"),
        )


@dataclass(frozen=True)
class ReviewItem:
    """Content the parser would not index on its own authority.

    A low-confidence OCR paragraph, a scanned page with no OCR engine
    configured, a figure no captioner could describe. Kept on the document --
    and so in the stored record -- rather than dropped, so nothing disappears
    silently and a person can look at exactly what was held back.
    """

    reason: str
    page: int | None = None
    text: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict:
        out: dict = {"reason": self.reason}
        if self.page is not None:
            out["page"] = self.page
        if self.text:
            out["text"] = self.text
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 4)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> ReviewItem:
        return cls(
            reason=data["reason"],
            page=data.get("page"),
            text=data.get("text", ""),
            confidence=data.get("confidence"),
        )


@dataclass(frozen=True)
class PageRoute:
    """The route one page took: its extraction tier and how many figures were captioned."""

    page: int
    tier: Tier
    figures: int = 0

    def to_dict(self) -> dict:
        return {"page": self.page, "tier": int(self.tier), "figures": self.figures}

    @classmethod
    def from_dict(cls, data: dict) -> PageRoute:
        return cls(page=data["page"], tier=Tier(data["tier"]), figures=data.get("figures", 0))


@dataclass
class Document:
    """A parsed document: its blocks in reading order, plus how it was parsed."""

    source: str
    media_type: str
    blocks: list[Block] = field(default_factory=list)
    routes: list[PageRoute] = field(default_factory=list)
    review: list[ReviewItem] = field(default_factory=list)

    def tier_counts(self) -> dict[str, int]:
        """Blocks produced per tier, by name -- the one-line answer to "what did this cost?"."""
        counts: dict[str, int] = {}
        for block in self.blocks:
            name = block.tier.name.lower()
            counts[name] = counts.get(name, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "ir_version": IR_VERSION,
            "source": self.source,
            "media_type": self.media_type,
            "blocks": [b.to_dict() for b in self.blocks],
            "routes": [r.to_dict() for r in self.routes],
            "review": [r.to_dict() for r in self.review],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Document:
        version = data.get("ir_version")
        if version != IR_VERSION:
            raise ValueError(f"stored document has ir_version={version!r}; this code reads {IR_VERSION}")
        return cls(
            source=data.get("source", ""),
            media_type=data.get("media_type", ""),
            blocks=[Block.from_dict(b) for b in data.get("blocks", ())],
            routes=[PageRoute.from_dict(r) for r in data.get("routes", ())],
            review=[ReviewItem.from_dict(r) for r in data.get("review", ())],
        )

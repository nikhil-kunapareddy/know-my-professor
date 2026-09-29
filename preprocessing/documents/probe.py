"""Probe & route: decide which tier reads each page, before paying for any.

Two pure decisions, kept apart from the parsers so the rules can be read and
tested on their own:

- ``sniff_media_type`` -- what the bytes ARE. Magic bytes beat the declared
  type, because servers routinely label PDFs ``application/octet-stream`` or
  even ``text/html``.
- ``route`` -- given cheap facts about one page (how much text its text layer
  holds, how much of it is covered by images), which tier extracts it and
  whether its figures go to the vision tier.

The rule, cheapest tier that can do the job:

- extractable text           -> tier 0 (layout)
- little text, mostly image  -> tier 1 (OCR): a scan
- figures on a text page     -> tier 2 (vision) for the figures, in addition
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import MIN_TEXT_LAYER_CHARS, SCANNED_PAGE_COVERAGE
from .ir import Tier

HTML = "text/html"
PDF = "application/pdf"

_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", PDF),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)


class UnsupportedDocument(ValueError):
    """The bytes are not a format any tier can read."""


def sniff_media_type(content: bytes, declared: str | None = None) -> str:
    """The media type of ``content``: magic bytes first, then the declared type."""
    head = content[:16]
    for magic, media_type in _MAGIC:
        if head.startswith(magic):
            return media_type
    if head.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"

    declared = (declared or "").split(";", 1)[0].strip().lower()
    if declared in (HTML, "application/xhtml+xml"):
        return HTML
    # Undeclared markup. Only the start is checked, after a BOM and whitespace.
    start = content[:512].lstrip(b"\xef\xbb\xbf \t\r\n").lower()
    if start.startswith((b"<!doctype html", b"<html", b"<head", b"<body", b"<main", b"<div", b"<p")):
        return HTML
    raise UnsupportedDocument(f"unrecognised document (declared {declared or 'nothing'!r})")


def is_image(media_type: str) -> bool:
    return media_type.startswith("image/")


@dataclass(frozen=True)
class PageProbe:
    """What can be learned about a page without reading it properly."""

    page: int
    #: Readable non-space characters in the page's text layer.
    text_chars: int
    #: Fraction of the page area covered by embedded images, clipped to 1.
    image_coverage: float
    #: Images large enough to be figures rather than logos or icons.
    figures: int
    #: Glyphs in the text layer with no Unicode mapping, which pdfminer reports
    #: as ``(cid:NN)``. A page made of these has a text layer nobody can read.
    unreadable_chars: int = 0


@dataclass(frozen=True)
class Route:
    extract: Tier
    caption_figures: bool


def route(probe: PageProbe) -> Route:
    """The tier that extracts this page, and whether its figures are captioned.

    A scanned page's one big image IS the page, so it is OCR'd, never also sent
    to the vision tier as a "figure". So is a page whose text layer exists but
    is unreadable glyph ids: rendering it to pixels recovers what the layer
    cannot. A sparse page that is neither (a title page, a divider) stays on
    tier 0, since OCR would find nothing the text layer did not already have.
    """
    if probe.text_chars < MIN_TEXT_LAYER_CHARS and (
        probe.image_coverage >= SCANNED_PAGE_COVERAGE or probe.unreadable_chars >= MIN_TEXT_LAYER_CHARS
    ):
        return Route(extract=Tier.OCR, caption_figures=False)
    return Route(extract=Tier.LAYOUT, caption_figures=probe.figures > 0)

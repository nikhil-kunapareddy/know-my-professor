"""Bytes in, ``Document`` out: probe each page, route it, run its tier.

``DocumentParser`` owns no parsing itself. It sniffs the format, asks
``probe.route`` which tier each page needs, runs that tier, and merges the
results into one ``Document`` in reading order. Tiers 1 and 2 are optional
collaborators: without an OCR engine or a captioner, the pages and figures that
needed them are recorded on ``Document.review`` instead of being dropped, so a
cheap run is visibly incomplete rather than quietly so.

A failure inside one page's OCR or one figure's caption is also recorded and
the rest of the document carries on -- one unreadable figure should not cost a
200-page PDF. A failure to open the document at all still raises.
"""

from __future__ import annotations

import io

from .config import (
    FIGURE_DPI,
    MIN_FIGURE_LABEL_CHARS,
    MIN_OCR_CHARS,
    OCR_DPI,
    OCR_REVIEW_CONFIDENCE,
)
from .ir import Block, BlockKind, Document, PageRoute, ReviewItem, Tier
from .probe import HTML, PDF, UnsupportedDocument, is_image, route, sniff_media_type
from .tiers.ocr import OcrEngine
from .tiers.vision import Captioner

_POINTS_PER_INCH = 72.0


class DocumentParser:
    """Routes a document's pages through the tiers it is configured with."""

    def __init__(
        self,
        ocr: OcrEngine | None = None,
        captioner: Captioner | None = None,
        *,
        review_confidence: float = OCR_REVIEW_CONFIDENCE,
    ):
        self.ocr = ocr
        self.captioner = captioner
        self.review_confidence = review_confidence

    def parse(self, content: bytes, *, media_type: str | None = None, source: str = "") -> Document:
        """Parse ``content``. ``media_type`` is a hint; the bytes decide."""
        kind = sniff_media_type(content, media_type)
        if kind == HTML:
            from .tiers.html import parse_html

            return Document(source, HTML, parse_html(_decode(content, media_type), source))
        if kind == PDF:
            return self._parse_pdf(content, source)
        if is_image(kind):
            return self._parse_image(content, kind, source)
        raise UnsupportedDocument(f"no tier reads {kind}")  # pragma: no cover - sniff raises first

    # --- PDF -----------------------------------------------------------------

    def _parse_pdf(self, content: bytes, source: str) -> Document:
        from .tiers.pdf import PdfLayout

        doc = Document(source, PDF)
        with PdfLayout.open(content) as layout:
            for page in layout.pages:
                number = page.page_number
                decision = route(layout.probe(page))
                if decision.extract is Tier.OCR:
                    blocks = self._ocr(
                        lambda page=page: layout.render(page, OCR_DPI),
                        doc,
                        page=number,
                        scale=_POINTS_PER_INCH / OCR_DPI,
                    )
                else:
                    blocks = layout.text_blocks(page)

                captioned = 0
                if decision.caption_figures:
                    for bbox in layout.figure_boxes(page):
                        figure = self._caption(
                            lambda page=page, bbox=bbox: layout.render(page, FIGURE_DPI, bbox),
                            doc,
                            page=number,
                            bbox=bbox,
                            context=_heading_before(doc.blocks + blocks, number, bbox),
                            labels=layout.figure_text(page, bbox),
                        )
                        if figure is not None:
                            blocks = _insert_by_position(blocks, figure)
                            captioned += figure.tier is Tier.VISION

                doc.blocks.extend(blocks)
                doc.routes.append(PageRoute(number, decision.extract, captioned))
        return doc

    # --- standalone images ---------------------------------------------------

    def _parse_image(self, content: bytes, media_type: str, source: str) -> Document:
        """OCR first; a picture that turns out to hold little text is captioned instead."""
        doc = Document(source, media_type)
        png = _to_png(content, media_type)

        blocks = self._ocr(lambda: png, doc, page=None, scale=None) if self.ocr is not None else []
        if sum(len(b.text) for b in blocks) >= MIN_OCR_CHARS:
            doc.blocks.extend(blocks)
            doc.routes.append(PageRoute(1, Tier.OCR))
            return doc

        # Too little text for a scan: it is a photo, chart or diagram. The
        # caption transcribes any text itself, so the stray OCR words are not kept.
        figure = self._caption(lambda: png, doc, page=None, bbox=None, context="")
        if figure is not None:
            doc.blocks.append(figure)
        doc.routes.append(PageRoute(1, Tier.VISION, int(figure is not None)))
        return doc

    # --- tiers 1 and 2 -------------------------------------------------------

    def _ocr(self, render, doc: Document, *, page: int | None, scale: float | None) -> list[Block]:
        """OCR one rendered page into paragraphs, holding weak reads for review."""
        if self.ocr is None:
            doc.review.append(ReviewItem("page needs OCR but no OCR engine is configured", page=page))
            return []
        try:
            paragraphs = self.ocr.recognize(render())
        except Exception as e:  # noqa: BLE001 - recorded, and the document carries on
            doc.review.append(ReviewItem(f"OCR failed: {type(e).__name__}: {e}", page=page))
            return []

        blocks = []
        for p in paragraphs:
            if p.confidence < self.review_confidence:
                doc.review.append(
                    ReviewItem("low OCR confidence", page=page, text=p.text, confidence=p.confidence)
                )
                continue
            bbox = tuple(v * scale for v in p.bbox) if scale is not None else None
            blocks.append(
                Block(
                    BlockKind.PARAGRAPH,
                    text=p.text,
                    page=page,
                    bbox=bbox,
                    tier=Tier.OCR,
                    confidence=p.confidence,
                )
            )
        return blocks

    def _caption(
        self, render, doc: Document, *, page: int | None, bbox, context: str, labels: str = ""
    ) -> Block | None:
        """Describe one figure, or record why it could not be.

        ``labels`` is the text-layer text inside a vector figure. Uncaptioned,
        the figure is still indexed by those labels -- "P1 incidents per month"
        is worth finding even without a description -- and still recorded for
        review, since the labels alone do not say what the figure shows.
        """
        caption = None
        if self.captioner is None:
            doc.review.append(ReviewItem("figure not captioned: no captioner is configured", page=page))
        else:
            try:
                caption = self.captioner.describe(render(), context=context)
            except Exception as e:  # noqa: BLE001 - recorded, and the document carries on
                doc.review.append(ReviewItem(f"caption failed: {type(e).__name__}: {e}", page=page))
            else:
                if caption is None:
                    doc.review.append(ReviewItem("captioner returned nothing usable", page=page))
        if caption is not None:
            return Block(BlockKind.FIGURE, text=caption.render(), page=page, bbox=bbox, tier=Tier.VISION)
        if len(labels) >= MIN_FIGURE_LABEL_CHARS:
            return Block(BlockKind.FIGURE, text=labels, page=page, bbox=bbox)
        return None


def _decode(content: bytes, media_type: str | None) -> str:
    """HTML bytes as text, honouring a declared charset."""
    charset = "utf-8"
    for param in (media_type or "").split(";")[1:]:
        key, _, value = param.partition("=")
        if key.strip().lower() == "charset" and value.strip():
            charset = value.strip().strip('"')
    try:
        return content.decode(charset, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def _to_png(content: bytes, media_type: str) -> bytes:
    """Any image as PNG, the one format every OCR engine and captioner takes."""
    if media_type == "image/png":
        return content
    from PIL import Image

    image = Image.open(io.BytesIO(content))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _heading_before(blocks: list[Block], page: int, bbox) -> str:
    """The most recent heading above a figure, to tell the captioner where it sits.

    A heading on an earlier page always counts; one on the figure's own page
    counts only if it starts above the figure.
    """
    for block in reversed(blocks):
        if block.kind is not BlockKind.HEADING:
            continue
        if block.page != page or block.bbox is None or block.bbox[1] <= bbox[1]:
            return block.text
    return ""


def _insert_by_position(blocks: list[Block], figure: Block) -> list[Block]:
    """``blocks`` with ``figure`` placed before the first block that starts below it."""
    top = figure.bbox[1] if figure.bbox else float("inf")
    for i, block in enumerate(blocks):
        if block.bbox is not None and block.bbox[1] > top:
            return blocks[:i] + [figure] + blocks[i:]
    return blocks + [figure]

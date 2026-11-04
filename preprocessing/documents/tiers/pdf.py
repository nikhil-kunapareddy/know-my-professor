"""Tier 0 for PDF: the text layer, read as structure rather than as a string.

pdfplumber (MIT, pure Python over pdfminer.six, renders through pypdfium2)
rather than the PyMuPDF + Docling pairing layout pipelines usually reach for:
PyMuPDF is AGPL, and Docling pulls in torch, which alone would put an image
several gigabytes over Artifact Registry's free tier. pdfplumber gives what the
IR needs -- every glyph with its font size and position, ruled tables, image
boxes -- and the structure is recovered from those with plain heuristics:

- **Lines** are built from words, not taken from pdfplumber's line grouping,
  because that groups by baseline alone and so reads straight across both
  columns of a two-column page. A visual line is cut wherever the gap between
  two words is wider than any word space could be -- a column gutter. Rotated
  text (arXiv's margin stamp, say) is dropped: it is never body text.
- **Headings** are short lines set noticeably larger than the document's body
  font, or entirely bold and at least body size -- academic templates set
  section headings only ~10% larger, in bold. Levels come from ranking the
  distinct heading styles across the whole document, so the biggest is ``h1``
  wherever it appears.
- **Paragraphs** are consecutive lines with no vertical gap between them and
  no first-line indent.
- **Reading order** is top-to-bottom, except on a two-column page, where each
  band between full-width elements is read left column first.
- **Lists** are runs of lines opening with a bullet or an enumerator;
  indented lines that follow continue the item.
- **Tables** come from ``find_tables`` (ruling lines). Their text is removed
  from the line stream so it is not read a second time as prose.
- **Running headers, footers and page numbers** are dropped when the same
  margin line repeats across pages.
- **Figures** are embedded images, and also clusters of vector drawing -- most
  charts in a PDF are drawn, not embedded, so looking only at images would miss
  them. Text inside a figure region is kept out of the prose: a diagram's
  labels read as a string of fragments ("Tok 1 [SEP] Mask LM"), so they go to
  the figure instead, and to the vision tier when one is configured. A "table"
  pdfplumber finds inside a figure is the figure's drawn boxes, and is dropped.
- **Hyphenation** at a line break is mended by asking the document itself:
  "fine-" + "tuning" keeps its hyphen when "fine-tuning" appears elsewhere,
  "representa-" + "tion" loses it when "representation" does.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pdfplumber

from ..config import (
    BOLD_HEADING_RATIO,
    COLUMN_GAP_RATIO,
    FIGURE_LABEL_MARGIN,
    HEADING_SIZE_RATIO,
    INDENT_RATIO,
    MARGIN_FRACTION,
    MAX_HEADING_CHARS,
    MIN_FIGURE_COVERAGE,
    MIN_TABLE_FILL,
    MIN_VECTOR_FIGURE_OBJECTS,
    PARAGRAPH_GAP_RATIO,
    REPEATED_MARGIN_FRACTION,
    TWO_COLUMN_MIN_FRACTION,
    VECTOR_CLUSTER_GAP,
    WORD_X_TOLERANCE,
)
from ..ir import Block, BlockKind
from ..probe import PageProbe

_BULLET_RE = re.compile(
    r"^\s*(?:[•●◦‣⁃∙▪■·*\-–—]"
    r"|\(?\d{1,3}[.)]|\(?[a-z][.)])\s+"
)
_PAGE_NUMBER_RE = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE)
_CID = "(cid:"
_CAPTION_RE = re.compile(r"^\s*(fig(ure)?|table|chart|exhibit)\.?\s*[\dA-Z]", re.IGNORECASE)
_PUNCT = ".,;:!?()[]{}\"'\u201c\u201d\u2018\u2019"
#: Font-name fragments that mean a heavy weight. "Medi" is Nimbus/Times Medium,
#: which LaTeX templates use for bold.
_BOLD_MARKERS = ("bold", "black", "heavy", "semibold", "demi", "medi")


def _is_bold(fontname: str) -> bool:
    name = fontname.split("+", 1)[-1].lower()
    return any(marker in name for marker in _BOLD_MARKERS)


@dataclass
class _Line:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    bold: bool = False

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 1.0)

    @property
    def style(self) -> tuple[float, bool]:
        return (self.size, self.bold)


@dataclass
class _Element:
    """A line or a table, positioned, for reading-order sorting."""

    x0: float
    x1: float
    top: float
    bottom: float
    line: _Line | None = None
    table: Block | None = None


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


def _margin_key(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _area(b) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def _touches(a, b, gap: float = 0.0) -> bool:
    return a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]


def _center_in(line: _Line, box) -> bool:
    cx, cy = (line.x0 + line.x1) / 2, (line.top + line.bottom) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _cluster(boxes: list[tuple[float, float, float, float]], gap: float) -> list[tuple[tuple, int]]:
    """Merge boxes within ``gap`` of each other. Returns (bbox, member count) per cluster."""
    clusters: list[list] = []  # [bbox, count]
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        merged = [box, 1]
        keep = []
        for c in clusters:
            if _touches(c[0], merged[0], gap):
                merged = [_union([c[0], merged[0]]), c[1] + merged[1]]
            else:
                keep.append(c)
        # A merge can grow a cluster into ones it skipped; settle until stable.
        changed = True
        while changed:
            changed = False
            rest = []
            for c in keep:
                if _touches(c[0], merged[0], gap):
                    merged = [_union([c[0], merged[0]]), c[1] + merged[1]]
                    changed = True
                else:
                    rest.append(c)
            keep = rest
        clusters = keep + [merged]
    return [(tuple(c[0]), c[1]) for c in clusters]


def _clip(bbox, page) -> tuple[float, float, float, float] | None:
    x0, top, x1, bottom = bbox
    x0, top = max(x0, 0.0), max(top, 0.0)
    x1, bottom = min(x1, float(page.width)), min(bottom, float(page.height))
    return (x0, top, x1, bottom) if x1 > x0 and bottom > top else None


class PdfLayout:
    """One open PDF: probes its pages, reads their text layers, renders pixels."""

    def __init__(self, pdf):
        self._pdf = pdf
        self._lines: dict[int, list[_Line]] = {}
        self._regions: dict[int, tuple[list[Block], list[tuple]]] = {}
        self._body_size: float | None = None
        self._heading_levels: dict[tuple[float, bool], int] | None = None
        self._running: set[str] | None = None
        self._vocab: set[str] = set()

    @classmethod
    @contextmanager
    def open(cls, content: bytes) -> Iterator[PdfLayout]:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            yield cls(pdf)

    @property
    def pages(self) -> list:
        return list(self._pdf.pages)

    # --- probe -------------------------------------------------------------

    def probe(self, page) -> PageProbe:
        readable = unreadable = 0
        for char in page.chars:
            text = char.get("text", "")
            if text.startswith(_CID):
                unreadable += 1
            elif text.strip():
                readable += 1
        area = float(page.width * page.height) or 1.0
        covered = sum(_area(b) for b in self._image_boxes(page))
        return PageProbe(
            page=page.page_number,
            text_chars=readable,
            image_coverage=min(covered / area, 1.0),
            figures=len(self.figure_boxes(page)),
            unreadable_chars=unreadable,
        )

    def _image_boxes(self, page) -> list[tuple[float, float, float, float]]:
        boxes = []
        for image in page.images:
            if (bbox := _clip((image["x0"], image["top"], image["x1"], image["bottom"]), page)) is not None:
                boxes.append(bbox)
        return boxes

    def figure_boxes(self, page) -> list[tuple[float, float, float, float]]:
        """Figure regions on ``page`` -- images and vector drawings -- top to bottom."""
        return self._page_regions(page)[1]

    def figure_text(self, page, bbox) -> str:
        """The text-layer words inside a figure region: its labels, legend, axis titles."""
        return " ".join(line.text for line in self._page_lines(page) if _center_in(line, bbox))

    def _page_regions(self, page) -> tuple[list[Block], list[tuple]]:
        """(tables, figure regions) for a page, decided together and cached.

        Together because each is defined against the other. A cluster of vector
        drawing that one detected table covers is that table's ruling; any other
        big cluster is a figure, and the "tables" pdfplumber found inside it are
        its drawn boxes. A table that is mostly empty cells is chart gridlines.
        """
        number = page.page_number
        if number in self._regions:
            return self._regions[number]
        self._document_stats()

        area = float(page.width * page.height) or 1.0
        tables = [t for t in self._tables(page) if _fill(t) >= MIN_TABLE_FILL]

        graphics = []
        for obj in (*page.curves, *page.rects, *page.lines):
            box = _clip((obj["x0"], obj["top"], obj["x1"], obj["bottom"]), page)
            # Zero-area boxes (a horizontal rule) clip to None too; they still
            # belong to a chart's axes, so keep them at their true extent.
            if box is None and obj["x0"] <= obj["x1"] and obj["top"] <= obj["bottom"]:
                box = (obj["x0"], obj["top"], obj["x1"], obj["bottom"])
                if box[2] < 0 or box[0] > float(page.width) or box[3] < 0 or box[1] > float(page.height):
                    continue  # wholly off the page
            if box is not None and _area(box) < 0.9 * area:  # a page border or background is not a figure
                graphics.append(box)

        figures = []
        for box, members in _cluster(graphics, VECTOR_CLUSTER_GAP):
            if members < MIN_VECTOR_FIGURE_OBJECTS or _area(box) / area < MIN_FIGURE_COVERAGE:
                continue
            if any(_area(t.bbox) >= 0.8 * _area(box) and _touches(t.bbox, box) for t in tables):
                continue  # the ruling of a real table
            figures.append(box)
        figures += [b for b in self._image_boxes(page) if _area(b) / area >= MIN_FIGURE_COVERAGE]

        # An image inside a drawing (a photo with callouts) is one figure, not two.
        regions = [self._with_labels(box, page) for box, _ in _cluster(figures, 0.0)]
        tables = [t for t in tables if not any(_touches(t.bbox, r) and _area(t.bbox) < _area(r) for r in regions)]
        self._regions[number] = (tables, sorted(regions, key=lambda b: (b[1], b[0])))
        return self._regions[number]

    # --- render ------------------------------------------------------------

    def render(self, page, dpi: int, bbox: tuple[float, float, float, float] | None = None) -> bytes:
        """``page`` (or the ``bbox`` region of it) as PNG bytes at ``dpi``."""
        region = page
        if bbox is not None and (clipped := _clip(bbox, page)) is not None:
            region = page.crop(clipped)
        image = region.to_image(resolution=dpi).original
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    # --- text layer --------------------------------------------------------

    def _page_lines(self, page) -> list[_Line]:
        number = page.page_number
        if number not in self._lines:
            upright = page.filter(lambda obj: obj.get("object_type") != "char" or obj.get("upright", True))
            words = upright.extract_words(x_tolerance=WORD_X_TOLERANCE, return_chars=True)
            self._lines[number] = [line for row in _rows(words) for line in _segments(row)]
        return self._lines[number]

    def _document_stats(self) -> None:
        """Body font size, heading levels and running margin lines, over every page.

        Document-wide because a heading is only "large" relative to the body text
        of the whole document, and a running header only "repeats" across pages.
        """
        if self._heading_levels is not None:
            return
        weights: Counter[float] = Counter()
        margin_pages: Counter[str] = Counter()
        pages = self.pages
        for page in pages:
            lines = self._page_lines(page)
            for line in lines:
                weights[line.size] += len(line.text)
                self._vocab.update(t.strip(_PUNCT).lower() for t in line.text.split())
            margin = float(page.height) * MARGIN_FRACTION
            keys = {
                _margin_key(line.text)
                for line in lines
                if line.top <= margin or line.bottom >= float(page.height) - margin
            }
            margin_pages.update(keys)

        self._body_size = weights.most_common(1)[0][0] if weights else 0.0
        heading_styles = sorted(
            {
                line.style
                for number in self._lines
                for line in self._lines[number]
                if self._looks_like_heading(line)
            },
            reverse=True,
        )
        self._heading_levels = {style: min(i + 1, 6) for i, style in enumerate(heading_styles)}
        self._running = (
            {k for k, n in margin_pages.items() if n >= REPEATED_MARGIN_FRACTION * len(pages)}
            if len(pages) >= 3
            else set()
        )

    def _with_labels(self, box, page) -> tuple:
        """``box`` grown to take in the figure's own labels just outside it.

        Axis titles, tick labels and legends often sit beyond the drawn area,
        and left there they read as stray prose. A nearby line counts as a
        label only if it is set smaller than body text and is not the figure's
        caption, which belongs to the prose that cites it.
        """
        body = self._body_size or 0.0
        reach = (box[0] - FIGURE_LABEL_MARGIN, box[1] - FIGURE_LABEL_MARGIN,
                 box[2] + FIGURE_LABEL_MARGIN, box[3] + FIGURE_LABEL_MARGIN)
        labels = [
            (line.x0, line.top, line.x1, line.bottom)
            for line in self._page_lines(page)
            if line.size < body
            and _center_in(line, reach)
            and not _CAPTION_RE.match(line.text)
        ]
        return _union([box, *labels])

    def _looks_like_heading(self, line: _Line) -> bool:
        if len(line.text) > MAX_HEADING_CHARS or not any(c.isalpha() for c in line.text):
            return False
        body = self._body_size or 0.0
        return line.size >= body * HEADING_SIZE_RATIO or (line.bold and line.size >= body * BOLD_HEADING_RATIO)

    def _is_margin_noise(self, line: _Line, page) -> bool:
        margin = float(page.height) * MARGIN_FRACTION
        if not (line.top <= margin or line.bottom >= float(page.height) - margin):
            return False
        return bool(_PAGE_NUMBER_RE.match(line.text)) or _margin_key(line.text) in (self._running or set())

    def _tables(self, page) -> list[Block]:
        tables = []
        for table in page.find_tables():
            rows = tuple(tuple(_norm(cell) for cell in row) for row in table.extract() if row)
            rows = tuple(r for r in rows if any(r))
            if not rows:
                continue
            width = max(len(r) for r in rows)
            rows = tuple(r + ("",) * (width - len(r)) for r in rows)
            tables.append(
                Block(
                    BlockKind.TABLE,
                    rows=rows,
                    # A ruled table's first row is its header far more often than
                    # not, and pdfplumber cannot tell. One row alone is data.
                    header_rows=1 if len(rows) > 1 else 0,
                    page=page.page_number,
                    bbox=tuple(float(v) for v in table.bbox),
                )
            )
        return tables

    def text_blocks(self, page) -> list[Block]:
        """The blocks of one page's text layer, in reading order, figures excluded."""
        self._document_stats()
        tables, figures = self._page_regions(page)
        taken = [t.bbox for t in tables] + figures

        elements = [
            _Element(line.x0, line.x1, line.top, line.bottom, line=line)
            for line in self._page_lines(page)
            if not any(_center_in(line, box) for box in taken) and not self._is_margin_noise(line, page)
        ]
        elements += [_Element(t.bbox[0], t.bbox[2], t.bbox[1], t.bbox[3], table=t) for t in tables]
        return _Assembler(page.page_number, self._heading_levels or {}, self._vocab).run(
            _reading_order(elements, float(page.width))
        )


def _reading_order(elements: list[_Element], width: float) -> list[_Element]:
    """Top-to-bottom; on a two-column page, left column before right within each band."""
    mid, tol = width / 2, width * 0.02

    def side(e: _Element) -> str:
        if e.x1 <= mid + tol:
            return "L"
        if e.x0 >= mid - tol:
            return "R"
        return "F"

    ordered = sorted(elements, key=lambda e: (e.top, e.x0))
    lines = [e for e in ordered if e.line is not None]
    if not lines:
        return ordered
    sides = Counter(side(e) for e in lines)
    if min(sides["L"], sides["R"]) / len(lines) < TWO_COLUMN_MIN_FRACTION:
        return ordered

    out: list[_Element] = []
    left: list[_Element] = []
    right: list[_Element] = []
    for e in ordered:
        s = side(e)
        if s == "F":
            out += left + right + [e]
            left, right = [], []
        else:
            (left if s == "L" else right).append(e)
    return out + left + right


class _Assembler:
    """Folds a page's ordered lines into heading, paragraph, list and table blocks."""

    def __init__(self, page: int, heading_levels: dict[tuple[float, bool], int], vocab: set[str]):
        self.page = page
        self.heading_levels = heading_levels
        self.vocab = vocab
        self.blocks: list[Block] = []
        self._kind: BlockKind | None = None
        self._texts: list[str] = []
        self._boxes: list[tuple[float, float, float, float]] = []
        self._level: int | None = None
        self._last: _Line | None = None
        self._item_x0 = 0.0

    def run(self, elements: list[_Element]) -> list[Block]:
        for e in elements:
            if e.table is not None:
                self._flush()
                self.blocks.append(e.table)
                self._last = None
            else:
                self._line(e.line)
        self._flush()
        return self.blocks

    def _close_to_last(self, line: _Line) -> bool:
        if self._last is None:
            return False
        gap = line.top - self._last.bottom
        return -2.0 <= gap <= PARAGRAPH_GAP_RATIO * self._last.height

    def _line(self, line: _Line) -> None:
        level = self.heading_levels.get(line.style) if len(line.text) <= MAX_HEADING_CHARS else None
        bullet = _BULLET_RE.match(line.text)
        box = (line.x0, line.top, line.x1, line.bottom)

        if level is not None:
            # A heading wrapped onto a second line is still one heading.
            if not (self._kind is BlockKind.HEADING and self._level == level and self._close_to_last(line)):
                self._start(BlockKind.HEADING, level=level)
            self._texts.append(line.text)
        elif bullet:
            if self._kind is not BlockKind.LIST:
                self._start(BlockKind.LIST)
            self._texts.append(line.text[bullet.end():].strip())
            self._item_x0 = line.x0
        elif self._kind is BlockKind.LIST and self._close_to_last(line) and line.x0 > self._item_x0 + 1.0:
            # An indented line under a bullet continues that item.
            self._texts[-1] = _join(self._texts[-1], line.text, self.vocab)
        elif self._kind is BlockKind.PARAGRAPH and self._close_to_last(line) and not self._indented(line):
            self._texts[-1] = _join(self._texts[-1], line.text, self.vocab)
        else:
            self._start(BlockKind.PARAGRAPH)
            self._texts.append(line.text)
        self._boxes.append(box)
        self._last = line

    def _indented(self, line: _Line) -> bool:
        """A first-line indent: the line starts a new paragraph even with no gap above."""
        return self._last is not None and line.x0 > self._last.x0 + INDENT_RATIO * line.size

    def _start(self, kind: BlockKind, level: int | None = None) -> None:
        self._flush()
        self._kind, self._level = kind, level

    def _flush(self) -> None:
        if self._kind is None or not self._texts:
            self._kind, self._texts, self._boxes, self._level = None, [], [], None
            return
        common = {"page": self.page, "bbox": _union(self._boxes)}
        if self._kind is BlockKind.HEADING:
            block = Block(BlockKind.HEADING, text=" ".join(self._texts), level=self._level, **common)
        elif self._kind is BlockKind.LIST:
            block = Block(BlockKind.LIST, items=tuple(t for t in self._texts if t), **common)
        else:
            block = Block(BlockKind.PARAGRAPH, text=self._texts[0], **common)
        if not block.is_empty():
            self.blocks.append(block)
        self._kind, self._texts, self._boxes, self._level = None, [], [], None


def _rows(words: list[dict]) -> list[list[dict]]:
    """Words grouped into visual rows, left to right.

    A word joins the row holding a word it overlaps vertically by at least half
    the shorter height, rather than the row whose top matches its own: a
    subscript ("BERT_BASE") or a footnote marker sits a few points off its line,
    and splitting it onto a row of its own breaks the sentence around it. The
    match is word to word, not word to one band per row, because the two
    columns of a page need not share a baseline -- measured against the left
    column's band, the right column's subscripts miss.
    """

    def overlap(a: dict, b: dict) -> float:
        shared = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
        return shared / max(min(a["bottom"] - a["top"], b["bottom"] - b["top"]), 0.1)

    rows: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        best, best_overlap = None, 0.5
        for row in rows[-3:]:
            score = max(overlap(word, other) for other in row)
            if score >= best_overlap:
                best, best_overlap = row, score
        if best is None:
            rows.append([word])
        else:
            best.append(word)
    return [sorted(row, key=lambda w: w["x0"]) for row in rows]


def _segments(row: list[dict]) -> list[_Line]:
    """Cut one visual row into lines at every gap too wide to be a word space."""
    lines: list[_Line] = []
    current: list[dict] = []
    for word in row:
        if current:
            size = _median_size(current)
            if word["x0"] - current[-1]["x1"] > COLUMN_GAP_RATIO * size:
                lines.append(_line(current))
                current = []
        current.append(word)
    if current:
        lines.append(_line(current))
    return [line for line in lines if line.text]


def _median_size(words: list[dict]) -> float:
    sizes = sorted(c.get("size", 0.0) for w in words for c in w.get("chars", ()))
    return sizes[len(sizes) // 2] if sizes else 10.0


def _line(words: list[dict]) -> _Line:
    chars = [c for w in words for c in w.get("chars", ()) if c.get("text", "").strip()]
    return _Line(
        text=_norm(" ".join(w["text"] for w in words)),
        x0=min(w["x0"] for w in words),
        x1=max(w["x1"] for w in words),
        top=min(w["top"] for w in words),
        bottom=max(w["bottom"] for w in words),
        size=round(_median_size(words) * 2) / 2,  # to 0.5pt, so styles compare equal
        bold=bool(chars) and all(_is_bold(c.get("fontname", "")) for c in chars),
    )


def _fill(table: Block) -> float:
    """Fraction of a table's cells holding text."""
    cells = [c for row in table.rows for c in row]
    return sum(1 for c in cells if c) / len(cells) if cells else 0.0


def _join(previous: str, line: str, vocab: set[str] = frozenset()) -> str:
    """Join a wrapped line, mending a word hyphenated across the break.

    Whether the hyphen belongs to the word is asked of the document itself.
    Without evidence either way, a long fragment ("representa-") was split by
    the typesetter, a short one ("pre-", "co-") is a real prefix, and a single
    character ("C-") is not a word fragment at all.
    """
    if not (previous.endswith("-") and len(previous) > 1 and previous[-2].isalpha() and line[:1].islower()):
        return f"{previous} {line}"
    head = previous.rsplit(" ", 1)[-1][:-1]
    tail = line.split(" ", 1)[0].strip(_PUNCT)
    if len(head) == 1:  # "a grade of C-" / "or better": a dash, not a split word
        return f"{previous} {line}"
    if "-" in head or f"{head}-{tail}".lower() in vocab:
        return previous + line
    if f"{head}{tail}".lower() in vocab or len(head) >= 4:
        return previous[:-1] + line
    return previous + line

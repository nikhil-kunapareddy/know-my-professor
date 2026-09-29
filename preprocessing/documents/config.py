"""Thresholds for routing, review and chunking. Starting points, not measurements.

No document in the corpus has been through this pipeline yet, so none of these
has been tuned against real input. Each says what it guards and which way to
move it; re-check them against the first real source that uses them.
"""

from __future__ import annotations

# --- probe & route -----------------------------------------------------------

#: A page whose text layer holds at least this many non-space characters is
#: read from that layer (tier 0). A scanned page has none; a real text page has
#: well over a thousand. A title page or a slide may fall below it, which is
#: why the image test below also has to pass before a page is sent to OCR.
MIN_TEXT_LAYER_CHARS = 100

#: A page with too little text whose images cover at least this fraction of it
#: is treated as a scan and OCR'd (tier 1).
SCANNED_PAGE_COVERAGE = 0.6

#: An embedded image must cover at least this fraction of the page to count as
#: a figure worth captioning (tier 2). Keeps logos, icons and rules out of the
#: vision model's bill.
MIN_FIGURE_COVERAGE = 0.05

#: A standalone image whose OCR text is shorter than this is a picture rather
#: than a photographed page, so it goes on to be captioned.
MIN_OCR_CHARS = 40

#: An uncaptioned vector figure is indexed by the text inside it only when that
#: text is at least this long; a few stray labels say nothing.
MIN_FIGURE_LABEL_CHARS = 20

# --- rendering pages to pixels -------------------------------------------------

#: Tesseract's accuracy falls off sharply below ~300 DPI.
OCR_DPI = 300
#: A figure only needs to be legible to a vision model, and a smaller image is
#: cheaper to send.
FIGURE_DPI = 150

# --- review queue --------------------------------------------------------------

#: OCR paragraphs below this mean confidence are held for review instead of
#: indexed. Tesseract reports per-word confidence; this is the paragraph mean.
OCR_REVIEW_CONFIDENCE = 0.80

# --- PDF layout heuristics -----------------------------------------------------

#: A line at least this much larger than the body font size is a heading.
HEADING_SIZE_RATIO = 1.15
#: ...as is an entirely bold line at least this large. LaTeX section headings
#: are bold at ~1.1x body size, which the plain size rule alone misses.
BOLD_HEADING_RATIO = 1.0
#: ...and no line longer than this is, since a large-font paragraph is not a heading.
MAX_HEADING_CHARS = 150
#: Horizontal gap, in font sizes, beyond which two words on one baseline are in
#: different columns. A justified word space stays well under half a font size;
#: a column gutter is well over one.
COLUMN_GAP_RATIO = 1.2
#: pdfplumber's default (3pt) glues words together in tightly set PDFs
#: ("applyingpre-trainedlanguage"); 1.5pt keeps them apart.
WORD_X_TOLERANCE = 1.5
#: A line starting this many font sizes right of the previous one is a
#: paragraph's indented first line.
INDENT_RATIO = 0.5
#: Vector drawing objects (curves, rects, lines) within this many points of each
#: other belong to one figure.
VECTOR_CLUSTER_GAP = 6.0
#: ...and a cluster needs at least this many of them to be a figure rather than
#: a rule, an underline or a box around a paragraph.
MIN_VECTOR_FIGURE_OBJECTS = 8
#: Text within this many points of a figure, and smaller than body text, is the
#: figure's own labelling (axis titles, legends) rather than prose.
FIGURE_LABEL_MARGIN = 24.0
#: A "table" with fewer than this fraction of its cells filled is chart
#: gridlines or a drawn diagram, not a table.
MIN_TABLE_FILL = 0.4
#: Two lines are one paragraph when the gap between them is at most this
#: fraction of a line's height.
PARAGRAPH_GAP_RATIO = 0.8
#: Page margin, as a fraction of page height, searched for running headers,
#: footers and page numbers.
MARGIN_FRACTION = 0.07
#: A margin line repeated on at least this fraction of pages is a running
#: header or footer. Only applied to documents with 3+ pages.
REPEATED_MARGIN_FRACTION = 0.5
#: A page is two-column when at least this fraction of its lines sit wholly on
#: each side of the centre.
TWO_COLUMN_MIN_FRACTION = 0.3

# --- tier 2: vision captions ---------------------------------------------------

#: Google AI Studio free tier, the same model the weblinks extractor uses. It
#: takes image input and supports ``response_schema``.
CAPTION_MODEL = "models/gemini-3.1-flash-lite"
CAPTION_MAX_RETRIES = 6
#: 15 requests per minute on the free tier.
CAPTION_PACE_SECONDS = 4.0

# --- chunking -----------------------------------------------------------------

#: Default size budget for one chunk's body, in characters (~400 tokens at the
#: usual ~4 characters per token). Characters rather than tokens so chunking
#: needs no tokenizer and is identical everywhere it runs. Most of today's
#: corpus sits far below it (median chunk ~320 characters).
DEFAULT_MAX_CHARS = 1600

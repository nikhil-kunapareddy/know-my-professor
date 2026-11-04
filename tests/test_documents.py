"""Tiered document parsing (preprocessing/documents): IR, probe & route, the tiers,
and the parser that runs them.

PDFs are built in-test with fpdf2 so every fixture is readable source, not a
binary blob. OCR and the vision model are fakes: no tesseract binary, no API.
"""

from __future__ import annotations

import io
import json
import sys

import pytest
from fpdf import FPDF
from PIL import Image, ImageDraw

from preprocessing.documents.ir import (
    IR_VERSION,
    Block,
    BlockKind,
    Document,
    PageRoute,
    ReviewItem,
    Tier,
)
from preprocessing.documents.parser import DocumentParser
from preprocessing.documents.probe import (
    HTML,
    PDF,
    PageProbe,
    UnsupportedDocument,
    route,
    sniff_media_type,
)
from preprocessing.documents.tiers.ocr import (
    OcrEngine,
    OcrParagraph,
    OcrUnavailable,
    TesseractOcr,
    build_ocr,
    paragraphs_from_tesseract,
)
from preprocessing.documents.tiers.vision import (
    Caption,
    Captioner,
    CaptionUnavailable,
    GeminiCaptioner,
    build_captioner,
    caption_from_json,
)

# --- fixtures ---------------------------------------------------------------------


def _png(size=(400, 250), draw=None) -> bytes:
    image = Image.new("RGB", size, "white")
    if draw:
        draw(ImageDraw.Draw(image))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _bars(d):
    d.rectangle([40, 150, 90, 240], fill="blue")
    d.rectangle([140, 60, 190, 240], fill="red")


def _runbook_pdf(*, figure=True, scan=True) -> bytes:
    """Page 1: headings, a wrapped paragraph, a list, a ruled table, an image figure.
    Page 2 (optional): a full-page image with no text layer -- a scan."""
    pdf = FPDF()
    pdf.set_auto_page_break(False)
    pdf.add_page()
    pdf.set_font("Helvetica", size=20)
    pdf.set_xy(20, 20)
    pdf.cell(0, 10, "Payments Service Runbook")
    pdf.set_font("Helvetica", size=11)
    pdf.set_xy(20, 36)
    pdf.multi_cell(
        170, 6,
        "Retention is seven years for every ledger entry. Operators must confirm the replica "
        "lag before promoting a region.",
    )
    pdf.set_font("Helvetica", size=15)
    pdf.set_xy(20, 60)
    pdf.cell(0, 8, "Regional failover procedure")
    pdf.set_font("Helvetica", size=11)
    for i, text in enumerate(["- Drain traffic from the region", "- Promote the replica", "- Verify the ledger"]):
        pdf.set_xy(24, 72 + i * 7)
        pdf.cell(0, 6, text)
    pdf.set_xy(20, 100)
    with pdf.table(width=120, col_widths=(40, 40, 40)) as table:
        for row in [("Region", "RTO", "RPO"), ("us-east", "1h", "5m"), ("eu-west", "2h", "15m")]:
            cells = table.row()
            for cell in row:
                cells.cell(cell)
    if figure:
        pdf.image(io.BytesIO(_png(draw=_bars)), x=20, y=150, w=120)
    if scan:
        pdf.add_page()
        pdf.image(io.BytesIO(_png((1240, 1754))), x=0, y=0, w=210, h=297)
    return bytes(pdf.output())


def _body(pdf: FPDF, x: float, y: float, w: float, text: str, size: int = 11) -> None:
    pdf.set_font("Helvetica", size=size)
    pdf.set_xy(x, y)
    pdf.multi_cell(w, 6, text)


LEFT = "Left column text begins here and keeps going for several lines of ordinary prose. " * 3
RIGHT = "Right column text starts at the same height and must be read after the left. " * 3


# --- IR ---------------------------------------------------------------------------


def test_document_round_trips_through_json():
    doc = Document(
        "x.pdf",
        PDF,
        blocks=[
            Block(BlockKind.HEADING, text="Title", level=1, page=1, bbox=(1.0, 2.0, 3.0, 4.0)),
            Block(BlockKind.PARAGRAPH, text="OCR'd.", page=2, tier=Tier.OCR, confidence=0.93),
            Block(BlockKind.LIST, items=("a", "b")),
            Block(BlockKind.TABLE, rows=(("A", "B"), ("1", "2")), header_rows=1),
            Block(BlockKind.FIGURE, text="A chart.", tier=Tier.VISION),
        ],
        routes=[PageRoute(1, Tier.LAYOUT, 1), PageRoute(2, Tier.OCR)],
        review=[ReviewItem("low OCR confidence", page=2, text="?", confidence=0.5)],
    )
    stored = json.loads(json.dumps(doc.to_dict()))
    assert stored["ir_version"] == IR_VERSION
    assert Document.from_dict(stored) == doc
    assert doc.tier_counts() == {"layout": 3, "ocr": 1, "vision": 1}


def test_ir_rejects_malformed_blocks_and_unknown_versions():
    with pytest.raises(ValueError, match="level"):
        Block(BlockKind.HEADING, text="x")
    with pytest.raises(ValueError, match="header_rows"):
        Block(BlockKind.TABLE, rows=(("a",),), header_rows=2)
    with pytest.raises(ValueError, match="confidence"):
        Block(BlockKind.PARAGRAPH, text="x", confidence=93)
    with pytest.raises(ValueError, match="ir_version"):
        Document.from_dict({"ir_version": IR_VERSION + 1, "blocks": []})


# --- probe & route -------------------------------------------------------------


@pytest.mark.parametrize(
    "content, declared, expected",
    [
        (b"%PDF-1.7\n...", "text/html", PDF),  # magic bytes beat a wrong label
        (b"\x89PNG\r\n\x1a\n....", None, "image/png"),
        (b"\xff\xd8\xff\xe0....", "application/octet-stream", "image/jpeg"),
        (b"GIF89a....", None, "image/gif"),
        (b"II*\x00....", None, "image/tiff"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", None, "image/webp"),
        (b"<p>no doctype</p>", None, HTML),
        (b"\xef\xbb\xbf  <!DOCTYPE html><html>", None, HTML),
        (b"plain words", "text/html; charset=utf-8", HTML),
    ],
)
def test_sniff_media_type(content, declared, expected):
    assert sniff_media_type(content, declared) == expected


def test_sniff_rejects_what_no_tier_reads():
    with pytest.raises(UnsupportedDocument):
        sniff_media_type(b"just some text", "text/plain")


@pytest.mark.parametrize(
    "probe, extract, caption",
    [
        (PageProbe(1, text_chars=2000, image_coverage=0.0, figures=0), Tier.LAYOUT, False),
        (PageProbe(1, text_chars=2000, image_coverage=0.3, figures=1), Tier.LAYOUT, True),
        (PageProbe(1, text_chars=0, image_coverage=1.0, figures=1), Tier.OCR, False),  # a scan
        (PageProbe(1, text_chars=5, image_coverage=0.0, figures=0, unreadable_chars=900), Tier.OCR, False),
        (PageProbe(1, text_chars=20, image_coverage=0.1, figures=0), Tier.LAYOUT, False),  # title page
        (PageProbe(1, text_chars=40, image_coverage=0.4, figures=1), Tier.LAYOUT, True),  # chart + caption
    ],
)
def test_route_picks_the_cheapest_tier_that_can_read_the_page(probe, extract, caption):
    decision = route(probe)
    assert (decision.extract, decision.caption_figures) == (extract, caption)


# --- tier 0: HTML ------------------------------------------------------------------

PROGRAM_HTML = """<html><head><title>BSCS</title></head><body>
<nav><a href="/">Home</a> <a href="/programs">Programs</a></nav>
<main>
  <h1>Computer Science, BSCS</h1>
  <p>The Bachelor of Science in Computer Science prepares students for careers in
     software and research, combining theory with practice across the field.</p>
  <h2>Program Requirements</h2>
  <h3>Computer Science Overview</h3>
  <p>Complete the following courses with a grade of C- or better, as listed below for every student.</p>
  <table>
    <thead><tr><th>Code</th><th>Title</th><th>Hours</th></tr></thead>
    <tbody><tr><td>CS 1200</td><td>First Year Seminar</td><td>1</td></tr>
           <tr><td>CS 1800</td><td>Discrete Structures</td><td>4</td></tr></tbody>
  </table>
  <h3>Electives</h3>
  <ul><li>CS 4100 Artificial Intelligence<ul><li>Requires CS 3500</li></ul></li>
      <li>CS 4120 Natural Language Processing</li></ul>
  <figure><img src="/map.png" alt="Four-year degree map"></figure>
</main>
<footer>Copyright Northeastern University</footer></body></html>"""


def test_html_tier_keeps_headings_tables_lists_and_alt_text():
    from preprocessing.documents.tiers.html import parse_html

    blocks = parse_html(PROGRAM_HTML)
    shape = [(b.kind, b.level) for b in blocks]
    assert shape == [
        (BlockKind.HEADING, 1),
        (BlockKind.PARAGRAPH, None),
        (BlockKind.HEADING, 2),
        (BlockKind.HEADING, 3),
        (BlockKind.PARAGRAPH, None),
        (BlockKind.TABLE, None),
        (BlockKind.HEADING, 3),
        (BlockKind.LIST, None),
        (BlockKind.FIGURE, None),
    ]
    table = blocks[5]
    assert table.header_rows == 1
    assert table.rows == (("Code", "Title", "Hours"), ("CS 1200", "First Year Seminar", "1"),
                          ("CS 1800", "Discrete Structures", "4"))
    assert blocks[7].items == ("CS 4100 Artificial Intelligence", "  Requires CS 3500",
                               "CS 4120 Natural Language Processing")
    assert blocks[8].text == "Four-year degree map"
    text = " ".join(b.text for b in blocks)
    assert "Copyright" not in text and "Programs" not in text  # boilerplate stripped


def test_html_tier_restores_an_h1_trafilatura_dropped(monkeypatch):
    from preprocessing.documents.tiers import html

    monkeypatch.setattr(
        html.trafilatura, "extract", lambda *a, **k: "<doc><main><p>Only the body.</p></main></doc>"
    )
    blocks = html.parse_html("<html><body><h1>Jane  Doe</h1><p>Only the body.</p></body></html>")
    assert blocks[0] == Block(BlockKind.HEADING, text="Jane Doe", level=1)
    assert blocks[1].text == "Only the body."


def test_html_tier_returns_nothing_for_a_page_without_content(monkeypatch):
    from preprocessing.documents.tiers import html

    monkeypatch.setattr(html.trafilatura, "extract", lambda *a, **k: None)
    assert html.parse_html("<html></html>") == []


# --- tier 0: PDF -------------------------------------------------------------------


def _layout_blocks(content: bytes, page: int = 0) -> list[Block]:
    from preprocessing.documents.tiers.pdf import PdfLayout

    with PdfLayout.open(content) as layout:
        return layout.text_blocks(layout.pages[page])


def test_pdf_tier_recovers_structure_from_the_text_layer():
    blocks = _layout_blocks(_runbook_pdf(figure=False, scan=False))
    assert [(b.kind, b.level) for b in blocks] == [
        (BlockKind.HEADING, 1),
        (BlockKind.PARAGRAPH, None),
        (BlockKind.HEADING, 2),
        (BlockKind.LIST, None),
        (BlockKind.TABLE, None),
    ]
    assert blocks[0].text == "Payments Service Runbook"
    # two wrapped lines, one paragraph
    assert blocks[1].text.startswith("Retention is seven years") and blocks[1].text.endswith("promoting a region.")
    assert blocks[3].items == ("Drain traffic from the region", "Promote the replica", "Verify the ledger")
    assert blocks[4].rows == (("Region", "RTO", "RPO"), ("us-east", "1h", "5m"), ("eu-west", "2h", "15m"))
    assert blocks[4].header_rows == 1
    # the table's cells are not read a second time as prose
    assert not any("us-east" in b.text for b in blocks if b.kind is BlockKind.PARAGRAPH)
    assert all(b.page == 1 and b.bbox is not None for b in blocks)


def test_pdf_tier_reads_two_columns_left_then_right():
    pdf = FPDF()
    pdf.add_page()
    _body(pdf, 15, 30, 85, LEFT)
    _body(pdf, 110, 30, 85, RIGHT)
    blocks = _layout_blocks(bytes(pdf.output()))
    assert [b.kind for b in blocks] == [BlockKind.PARAGRAPH, BlockKind.PARAGRAPH]
    assert blocks[0].text == " ".join(LEFT.split())
    assert blocks[1].text == " ".join(RIGHT.split())


def test_pdf_tier_drops_running_headers_page_numbers_and_rotated_text():
    pdf = FPDF()
    pdf.set_auto_page_break(False)
    for n in range(1, 4):
        pdf.add_page()
        pdf.set_font("Helvetica", size=9)
        pdf.set_xy(20, 8)
        pdf.cell(0, 5, "ACME Payments - Internal Runbook")
        pdf.set_xy(100, 285)
        pdf.cell(0, 5, str(n))
        with pdf.rotation(90, x=8, y=200):
            pdf.text(8, 200, "arXiv:0000.00000v1 [cs.XX]")
        _body(pdf, 20, 40, 170, f"Body text of page {n} is the only thing that should remain here.")
    from preprocessing.documents.tiers.pdf import PdfLayout

    with PdfLayout.open(bytes(pdf.output())) as layout:
        texts = [b.text for page in layout.pages for b in layout.text_blocks(page)]
    assert texts == [f"Body text of page {n} is the only thing that should remain here." for n in (1, 2, 3)]


def test_pdf_tier_treats_a_bold_body_size_line_as_a_heading():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_xy(20, 20)
    pdf.cell(0, 6, "2.1 Related Work")
    _body(pdf, 20, 30, 170, "Plenty of prior work exists on this, and it is summarised in this paragraph. " * 4)
    blocks = _layout_blocks(bytes(pdf.output()))
    assert blocks[0] == Block(BlockKind.HEADING, text="2.1 Related Work", level=1, page=1, bbox=blocks[0].bbox)
    assert blocks[1].kind is BlockKind.PARAGRAPH


def test_pdf_tier_finds_a_vector_chart_and_keeps_its_labels_out_of_the_prose():
    pdf = FPDF()
    pdf.add_page()
    _body(pdf, 20, 20, 170, "Incidents fell sharply after the migration, as the chart below shows.")
    pdf.line(30, 140, 150, 140)  # x axis
    pdf.line(30, 60, 30, 140)  # y axis
    for i, h in enumerate([20, 45, 70, 30, 55, 10, 65, 40]):
        pdf.rect(35 + i * 14, 140 - h, 10, h, style="F")
    pdf.set_font("Helvetica", size=8)
    pdf.set_xy(60, 145)
    pdf.cell(0, 4, "P1 incidents per month")
    _body(pdf, 20, 160, 170, "Figure 1: Monthly P1 incidents.")

    from preprocessing.documents.tiers.pdf import PdfLayout

    with PdfLayout.open(bytes(pdf.output())) as layout:
        page = layout.pages[0]
        [box] = layout.figure_boxes(page)
        assert layout.probe(page).figures == 1
        assert "P1 incidents per month" in layout.figure_text(page, box)
        blocks = layout.text_blocks(page)
    assert [b.text for b in blocks] == [
        "Incidents fell sharply after the migration, as the chart below shows.",
        "Figure 1: Monthly P1 incidents.",
    ]


def test_a_ruled_table_is_a_table_not_a_figure():
    from preprocessing.documents.tiers.pdf import PdfLayout

    rows = [("Code", "Title")] + [(f"CS {1000 + i}", f"Course {i}") for i in range(12)]
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    with pdf.table(width=120) as table:
        for row in rows:
            cells = table.row()
            for cell in row:
                cells.cell(cell)
    with PdfLayout.open(bytes(pdf.output())) as layout:
        page = layout.pages[0]
        assert layout.figure_boxes(page) == []
        [block] = layout.text_blocks(page)
    assert block.kind is BlockKind.TABLE and block.rows == tuple(rows)


def test_pdf_probe_sees_the_scan_and_the_figure():
    from preprocessing.documents.tiers.pdf import PdfLayout

    with PdfLayout.open(_runbook_pdf()) as layout:
        text_page, scan_page = (layout.probe(p) for p in layout.pages)
    assert text_page.text_chars > 100 and text_page.figures == 1
    assert scan_page.text_chars == 0 and scan_page.image_coverage > 0.95


def test_row_grouping_keeps_subscripts_on_their_line():
    from preprocessing.documents.tiers.pdf import _rows

    def w(text, x0, top, bottom):
        return {"text": text, "x0": x0, "x1": x0 + 20, "top": top, "bottom": bottom, "chars": []}

    # Measured on BERT's page 3: the left column sits 3pt above the right, and
    # the BASE subscript only overlaps the right column's line.
    words = [
        w("steps", 236.6, 577.9, 588.8),
        w("BERT", 307.3, 580.9, 591.8),
        w("(L=12,", 363.4, 581.0, 591.9),
        w("BASE", 336.7, 584.9, 592.9),
        w("eters=110M)", 307.3, 594.5, 605.4),
    ]
    rows = _rows(words)
    assert [[x["text"] for x in row] for row in rows] == [["steps", "BERT", "BASE", "(L=12,"], ["eters=110M)"]]


@pytest.mark.parametrize(
    "previous, line, vocab, joined",
    [
        ("a new representa-", "tion model", set(), "a new representation model"),
        ("standard fine-", "tuning works", {"fine-tuning"}, "standard fine-tuning works"),
        ("the left-to-", "right model", set(), "the left-to-right model"),
        ("we pre-", "train it", set(), "we pre-train it"),  # short prefix, no evidence
        ("a grade of C-", "or better", set(), "a grade of C- or better"),  # not a split word
        ("plain", "words", set(), "plain words"),
    ],
)
def test_hyphens_across_line_breaks(previous, line, vocab, joined):
    from preprocessing.documents.tiers.pdf import _join

    assert _join(previous, line, vocab) == joined


# --- tier 1: OCR ----------------------------------------------------------------------


def test_tesseract_rows_group_into_paragraphs_with_mean_confidence():
    data = {
        "text": ["", "Retention", "is", "seven", "", "years.", "Scanned"],
        "conf": [-1, 96, 90, 93, -1, 91, 40],
        "block_num": [1, 1, 1, 1, 1, 1, 2],
        "par_num": [1, 1, 1, 1, 1, 1, 1],
        "left": [0, 10, 60, 80, 0, 10, 10],
        "top": [0, 10, 10, 10, 0, 30, 90],
        "width": [0, 45, 15, 30, 0, 35, 50],
        "height": [0, 12, 12, 12, 0, 12, 12],
    }
    first, second = paragraphs_from_tesseract(data)
    assert first.text == "Retention is seven years."
    assert first.confidence == pytest.approx((96 + 90 + 93 + 91) / 4 / 100)
    assert first.bbox == (10.0, 10.0, 110.0, 42.0)
    assert second == OcrParagraph("Scanned", 0.40, (10.0, 90.0, 60.0, 102.0))


def test_tesseract_fails_fast_without_its_binary(monkeypatch):
    pytest.importorskip("pytesseract")
    monkeypatch.setattr("preprocessing.documents.tiers.ocr.shutil.which", lambda _: None)
    with pytest.raises(OcrUnavailable, match="tesseract"):
        TesseractOcr()


def test_ocr_registry():
    assert build_ocr("none") is None and build_ocr("") is None and build_ocr(None) is None
    with pytest.raises(ValueError, match="registered"):
        build_ocr("textract")


# --- tier 2: vision --------------------------------------------------------------------


def test_caption_parsing_and_rendering():
    assert caption_from_json({"description": ""}) is None
    assert caption_from_json(["not", "a", "dict"]) is None
    caption = caption_from_json({"description": "A bar chart of incidents.", "text": "P1  per month",
                                 "values": ["peak: 9", " "]})
    assert caption == Caption("A bar chart of incidents.", "P1  per month", ("peak: 9",))
    assert caption.render() == "A bar chart of incidents.\nText in figure: P1 per month\nValues: peak: 9"


class _GeminiResponse:
    def __init__(self, text):
        self.text = text


class _FakeGemini:
    def __init__(self, text):
        self.text, self.calls = text, []

    def generate_content(self, parts, generation_config):
        self.calls.append((parts, generation_config))
        return _GeminiResponse(self.text)


def test_gemini_captioner_sends_the_image_with_a_schema_and_paces_itself():
    client = _FakeGemini('{"description": "Two bars.", "values": ["red: taller"]}')
    slept = []
    captioner = GeminiCaptioner(client=client, pace_seconds=4.0, sleep=slept.append)
    caption = captioner.describe(b"PNGDATA", context="Failover")

    assert caption == Caption("Two bars.", "", ("red: taller",))
    [(parts, config)] = client.calls
    assert parts[1] == {"mime_type": "image/png", "data": b"PNGDATA"}
    assert "Failover" in parts[0]
    assert config["response_mime_type"] == "application/json" and "response_schema" in config
    assert slept == [4.0]
    assert GeminiCaptioner(client=_FakeGemini("not json"), pace_seconds=0).describe(b"x") is None


def test_gemini_captioner_fails_fast_without_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(CaptionUnavailable):
        GeminiCaptioner()
    assert build_captioner("off") is None
    with pytest.raises(ValueError, match="registered"):
        build_captioner("gpt")


# --- DocumentParser: routing end to end ----------------------------------------------


class FakeOcr(OcrEngine):
    name = "fake"

    def __init__(self, paragraphs=None, error=None):
        self.paragraphs = paragraphs if paragraphs is not None else [
            OcrParagraph("Scanned incident report: the ledger was restored from backup.", 0.94, (300, 300, 900, 360)),
            OcrParagraph("smudged fragment", 0.55, (300, 400, 600, 440)),
        ]
        self.error = error
        self.images = []

    def recognize(self, image_png):
        self.images.append(image_png)
        if self.error:
            raise self.error
        return self.paragraphs


class FakeCaptioner(Captioner):
    name = "fake"

    def __init__(self, caption=Caption("A bar chart: the red bar is taller.", values=("bars: 2",)), error=None):
        self.caption, self.error, self.calls = caption, error, []

    def describe(self, image_png, context=""):
        self.calls.append(context)
        if self.error:
            raise self.error
        return self.caption


def test_parser_routes_each_page_to_its_tier():
    ocr, captioner = FakeOcr(), FakeCaptioner()
    doc = DocumentParser(ocr=ocr, captioner=captioner).parse(_runbook_pdf(), source="runbook.pdf")

    assert doc.media_type == PDF
    assert doc.routes == [PageRoute(1, Tier.LAYOUT, 1), PageRoute(2, Tier.OCR, 0)]
    assert doc.tier_counts() == {"layout": 5, "vision": 1, "ocr": 1}

    [figure] = [b for b in doc.blocks if b.kind is BlockKind.FIGURE]
    assert figure.tier is Tier.VISION and figure.page == 1
    assert figure.text == "A bar chart: the red bar is taller.\nValues: bars: 2"
    assert captioner.calls == ["Regional failover procedure"]  # the heading above it
    # placed after the table it sits below, not appended after page 2
    kinds = [b.kind for b in doc.blocks]
    assert kinds.index(BlockKind.FIGURE) == kinds.index(BlockKind.TABLE) + 1

    [scanned] = [b for b in doc.blocks if b.tier is Tier.OCR]
    assert scanned.page == 2 and scanned.confidence == 0.94
    assert scanned.bbox == pytest.approx((72.0, 72.0, 216.0, 86.4))  # 300 dpi pixels -> points
    # the weak read is held, not indexed
    assert doc.review == [ReviewItem("low OCR confidence", page=2, text="smudged fragment", confidence=0.55)]


def test_parser_without_tiers_1_and_2_records_what_it_skipped():
    doc = DocumentParser().parse(_runbook_pdf())
    assert doc.tier_counts() == {"layout": 5}
    assert [(r.page, r.reason) for r in doc.review] == [
        (1, "figure not captioned: no captioner is configured"),
        (2, "page needs OCR but no OCR engine is configured"),
    ]


def test_a_failing_tier_costs_its_page_not_the_document():
    doc = DocumentParser(ocr=FakeOcr(error=RuntimeError("boom")), captioner=FakeCaptioner(error=TimeoutError())).parse(
        _runbook_pdf()
    )
    assert doc.tier_counts() == {"layout": 5}
    reasons = [r.reason for r in doc.review]
    assert reasons == ["caption failed: TimeoutError: ", "OCR failed: RuntimeError: boom"]


def test_an_image_of_text_is_ocrd_and_a_picture_is_captioned():
    scan = DocumentParser(ocr=FakeOcr(), captioner=FakeCaptioner()).parse(_png(), source="scan.png")
    assert scan.routes == [PageRoute(1, Tier.OCR)]
    assert [b.tier for b in scan.blocks] == [Tier.OCR]

    too_little = FakeOcr([OcrParagraph("Q3", 0.99, (0, 0, 10, 10))])
    photo = DocumentParser(ocr=too_little, captioner=FakeCaptioner()).parse(_png(), source="chart.png")
    assert photo.routes == [PageRoute(1, Tier.VISION, 1)]
    assert [b.kind for b in photo.blocks] == [BlockKind.FIGURE]

    blind = DocumentParser().parse(_png())
    assert blind.blocks == [] and blind.review[0].reason.startswith("figure not captioned")


def test_non_png_images_reach_the_tiers_as_png():
    buffer = io.BytesIO()
    Image.new("RGB", (50, 50), "white").save(buffer, "JPEG")
    ocr = FakeOcr()
    DocumentParser(ocr=ocr).parse(buffer.getvalue())
    assert ocr.images[0].startswith(b"\x89PNG")


def test_parser_reads_html_bytes_with_a_declared_charset():
    html = PROGRAM_HTML.replace("Discrete Structures", "Structures Discrètes").encode("latin-1")
    doc = DocumentParser().parse(html, media_type="text/html; charset=latin-1", source="bscs")
    assert doc.media_type == HTML and doc.routes == []
    assert any("Structures Discrètes" in cell for b in doc.blocks for row in b.rows for cell in row)


def test_a_parsed_document_survives_storage():
    """Stored and read back, a document stores identically (boxes are kept to 0.01pt)."""
    doc = DocumentParser(ocr=FakeOcr(), captioner=FakeCaptioner()).parse(_runbook_pdf())
    stored = json.loads(json.dumps(doc.to_dict()))
    restored = Document.from_dict(stored)
    assert restored.to_dict() == stored
    assert [(b.kind, b.text, b.items, b.rows, b.tier) for b in restored.blocks] == [
        (b.kind, b.text, b.items, b.rows, b.tier) for b in doc.blocks
    ]


def test_ingest_rebuilds_and_chunks_documents_without_any_parsing_dependency():
    """The ingest image installs none of pdfplumber/tesseract/trafilatura/PIL/Gemini.

    Run in a subprocess with those imports blocked, so an accidental top-level
    import in ir.py, chunking.py or sources/base.py fails here rather than as a
    boot failure of the ingest Job.
    """
    import subprocess
    import textwrap

    script = textwrap.dedent('''
        import importlib.abc, sys
        BLOCKED = {"pdfplumber", "pypdfium2", "pytesseract", "PIL", "trafilatura", "bs4", "fpdf"}
        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED or name.startswith("google.generativeai"):
                    raise ImportError("blocked: " + name)
        sys.meta_path.insert(0, Blocker())
        import preprocessing.ingest.runner
        from preprocessing.documents.ir import Document
        from preprocessing.sources.base import section_chunks
        doc = Document.from_dict({"ir_version": 1, "source": "x", "media_type": "application/pdf",
            "blocks": [{"kind": "heading", "text": "T", "level": 1},
                       {"kind": "table", "rows": [["a", "b"], ["1", "2"]], "header_rows": 1}]})
        [chunk] = section_chunks("d#body", "Doc", "Body", doc, {})
        assert chunk.text == "Doc\\nBody > T:\\n| a | b |\\n| --- | --- |\\n| 1 | 2 |", chunk.text
    ''')
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

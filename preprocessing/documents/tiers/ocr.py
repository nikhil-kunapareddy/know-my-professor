"""Tier 1: OCR, for pages that are pictures of text.

An ``OcrEngine`` turns PNG bytes into paragraphs, each with a confidence, so the
parser can hold weak reads for review instead of indexing them. Engines are a
registry like every other provider here; ``build_ocr("none")`` is how a caller
says "no OCR", and pages that needed it are then recorded for review rather than
silently dropped.

The one engine shipped is Tesseract: free, local, unlimited, and it reports
per-word confidence. It needs the ``tesseract`` binary on the machine, which no
image in ``deploy/Dockerfile`` installs yet -- the first Job that OCRs needs
``apt-get install -y tesseract-ocr`` for its component (``brew install
tesseract`` locally). Construction fails fast without it, before any page is
read. Google Cloud Vision (1,000 free pages a month, and GCP) is the natural
second engine if Tesseract's accuracy is not enough.
"""

from __future__ import annotations

import io
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass


class OcrUnavailable(RuntimeError):
    """The configured OCR engine cannot run here (missing binary or package)."""


@dataclass(frozen=True)
class OcrParagraph:
    """One recognised paragraph. ``bbox`` is in the input image's pixels."""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


class OcrEngine(ABC):
    name: str

    @abstractmethod
    def recognize(self, image_png: bytes) -> list[OcrParagraph]:
        """Paragraphs in reading order. Empty when the image holds no text."""


class TesseractOcr(OcrEngine):
    """OCR through the local ``tesseract`` binary, via pytesseract."""

    name = "tesseract"

    def __init__(self, lang: str = "eng"):
        try:
            import pytesseract
        except ImportError as e:  # pragma: no cover - present wherever the extra is installed
            raise OcrUnavailable("pytesseract is not installed; pip install '.[documents]'") from e
        if shutil.which(pytesseract.pytesseract.tesseract_cmd) is None:
            raise OcrUnavailable(
                "the tesseract binary is not on PATH "
                "(apt-get install -y tesseract-ocr, or brew install tesseract)"
            )
        self._tesseract = pytesseract
        self.lang = lang

    def recognize(self, image_png: bytes) -> list[OcrParagraph]:
        from PIL import Image

        data = self._tesseract.image_to_data(
            Image.open(io.BytesIO(image_png)),
            lang=self.lang,
            output_type=self._tesseract.Output.DICT,
        )
        return paragraphs_from_tesseract(data)


def paragraphs_from_tesseract(data: dict) -> list[OcrParagraph]:
    """Group Tesseract's per-word rows into paragraphs, in the order it read them.

    Rows with confidence -1 are layout rows (page, block, line), not words.
    Kept separate from the engine so the grouping is testable without a binary.
    """
    groups: dict[tuple[int, int], dict] = {}
    order: list[tuple[int, int]] = []
    for i, word in enumerate(data.get("text", ())):
        conf = float(data["conf"][i])
        if conf < 0 or not str(word).strip():
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]))
        if key not in groups:
            groups[key] = {"words": [], "confs": [], "boxes": []}
            order.append(key)
        left, top = float(data["left"][i]), float(data["top"][i])
        groups[key]["words"].append(str(word).strip())
        groups[key]["confs"].append(conf)
        groups[key]["boxes"].append((left, top, left + float(data["width"][i]), top + float(data["height"][i])))

    paragraphs = []
    for key in order:
        g = groups[key]
        boxes = g["boxes"]
        paragraphs.append(
            OcrParagraph(
                text=" ".join(g["words"]),
                confidence=sum(g["confs"]) / len(g["confs"]) / 100.0,
                bbox=(
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                ),
            )
        )
    return paragraphs


_OCR_ENGINES: dict[str, type[OcrEngine]] = {"tesseract": TesseractOcr}
_DISABLED = {"", "none", "off"}


def build_ocr(name: str | None) -> OcrEngine | None:
    """The named OCR engine, or ``None`` for "none"/"off"/empty."""
    key = (name or "").strip().lower()
    if key in _DISABLED:
        return None
    try:
        return _OCR_ENGINES[key]()
    except KeyError:
        raise ValueError(f"unknown OCR engine {name!r}; registered: {sorted(_OCR_ENGINES)}") from None

"""Tier 2: a vision-language model describes what a figure shows.

The only tier that can read a chart as a chart. OCR on a bar chart returns its
axis labels; a caption says "P1 incidents per month, peaking at 9 in March",
which is what someone searching for it would type.

A ``Captioner`` returns three things, so a figure indexes on all of them:
a description, the text visible in it, and the key numbers with their labels.

The one captioner shipped is Gemini through Google AI Studio -- the free tier,
and the SDK and key (``GEMINI_API_KEY``) the weblinks extractor already uses.
``build_captioner("none")`` turns the tier off; figures are then recorded for
review rather than silently dropped.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from shared.retry import with_backoff

from ..config import CAPTION_MAX_RETRIES, CAPTION_MODEL, CAPTION_PACE_SECONDS

CAPTION_PROMPT = (
    "This image is a figure taken from a document that is being indexed for "
    "search. Describe it using ONLY what is visible; do not guess.\n"
    "- description: 1-3 sentences: what kind of figure it is (chart, diagram, "
    "photo, screenshot, scanned table...) and what it shows, including the main "
    "takeaway.\n"
    "- text: every legible piece of text in the image, verbatim.\n"
    "- values: the key numbers, each with its label, e.g. 'P1 incidents, peak "
    "month: 9'. Empty if there are none.\n"
)

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "text": {"type": "string"},
        "values": {"type": "array", "items": {"type": "string"}},
    },
}


class CaptionUnavailable(RuntimeError):
    """The configured captioner cannot run here (missing key or package)."""


@dataclass(frozen=True)
class Caption:
    description: str
    text: str = ""
    values: tuple[str, ...] = ()

    def render(self) -> str:
        """The figure's indexable text: description, then transcription, then numbers."""
        lines = [self.description.strip()]
        if self.text.strip():
            lines.append(f"Text in figure: {' '.join(self.text.split())}")
        if self.values:
            lines.append("Values: " + "; ".join(v.strip() for v in self.values if v.strip()))
        return "\n".join(line for line in lines if line)


class Captioner(ABC):
    name: str

    @abstractmethod
    def describe(self, image_png: bytes, context: str = "") -> Caption | None:
        """A caption for the image, or ``None`` if the model gave nothing usable."""


def _is_retryable(exc: BaseException) -> bool:
    """429 and 5xx from Google's API. Imported lazily so tests need no SDK."""
    try:
        from google.api_core import exceptions
    except ImportError:  # pragma: no cover
        return False
    return isinstance(
        exc,
        (exceptions.ResourceExhausted, exceptions.ServiceUnavailable, exceptions.InternalServerError),
    )


class GeminiCaptioner(Captioner):
    """Captions through Gemini's free tier, paced to its per-minute limit."""

    name = "gemini"

    def __init__(
        self,
        model: str = CAPTION_MODEL,
        client=None,
        pace_seconds: float = CAPTION_PACE_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.pace_seconds = pace_seconds
        self._sleep = sleep
        self._client = client
        # Checked here, not on first use: a missing key discovered mid-document
        # would turn every figure into a review item instead of failing the run.
        if client is None and not os.environ.get("GEMINI_API_KEY"):
            raise CaptionUnavailable("GEMINI_API_KEY env required for figure captions")

    @property
    def client(self):
        if self._client is None:
            import google.generativeai as genai

            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def describe(self, image_png: bytes, context: str = "") -> Caption | None:
        prompt = CAPTION_PROMPT + (f"\nThe figure appears under: {context}\n" if context else "")
        response = with_backoff(
            lambda: self.client.generate_content(
                [prompt, {"mime_type": "image/png", "data": image_png}],
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": CAPTION_SCHEMA,
                    "temperature": 0.0,
                },
            ),
            is_retryable=_is_retryable,
            max_attempts=CAPTION_MAX_RETRIES,
            label="gemini caption",
        )
        if self.pace_seconds:
            self._sleep(self.pace_seconds)
        try:
            parsed = json.loads(response.text)
        except (ValueError, AttributeError):
            return None
        return caption_from_json(parsed)


def caption_from_json(parsed) -> Caption | None:
    """A ``Caption`` from the model's JSON, or ``None`` when it has no description."""
    if not isinstance(parsed, dict):
        return None
    description = str(parsed.get("description") or "").strip()
    if not description:
        return None
    values = parsed.get("values") or []
    return Caption(
        description=description,
        text=str(parsed.get("text") or ""),
        values=tuple(str(v) for v in values if str(v).strip()) if isinstance(values, list) else (),
    )


_CAPTIONERS: dict[str, type[Captioner]] = {"gemini": GeminiCaptioner}
_DISABLED = {"", "none", "off"}


def build_captioner(name: str | None) -> Captioner | None:
    """The named captioner, or ``None`` for "none"/"off"/empty."""
    key = (name or "").strip().lower()
    if key in _DISABLED:
        return None
    try:
        return _CAPTIONERS[key]()
    except KeyError:
        raise ValueError(f"unknown captioner {name!r}; registered: {sorted(_CAPTIONERS)}") from None

"""HTML -> clean text -> structured sections.

trafilatura strips boilerplate (nav/footer/scripts) so we don't need a parser
per site; Gemini then normalizes arbitrary clean text into our fixed schema.
"""

from __future__ import annotations

import hashlib
import json
import time

import google.generativeai as genai
import trafilatura
from google.api_core.exceptions import ResourceExhausted

from shared.config import (
    EXTRACTION_PROMPT,
    EXTRACTION_SCHEMA,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_RATE_LIMIT_SLEEP_SECONDS,
    MAX_CLEAN_TEXT_CHARS,
    MIN_CLEAN_TEXT_CHARS,
    SCHEMA_VERSION,
)


class Extractor:
    """Cleans crawled pages and extracts structured sections via Gemini.

    The text-cleaning/hashing helpers are static (pure functions of their
    input); ``extract_structured`` makes the rate-limited Gemini call.
    """

    @staticmethod
    def clean_pages(pages: list[tuple[str, str]]) -> str:
        """Main-content text from each page, concatenated and URL-labeled.

        Pages are sorted by URL so the combined text — and its hash — is stable
        regardless of crawl order.
        """
        parts: list[str] = []
        for url, html in sorted(pages, key=lambda p: p[0]):
            text = trafilatura.extract(html, include_comments=False, include_tables=True)
            if text and text.strip():
                parts.append(f"[{url}]\n{text.strip()}")
        combined = "\n\n".join(parts)
        return combined[:MAX_CLEAN_TEXT_CHARS]

    @staticmethod
    def extraction_succeeded(clean_text: str) -> bool:
        """Guard: too-short text means a failed/JS-only/error page — don't cache it."""
        return len(clean_text.strip()) >= MIN_CLEAN_TEXT_CHARS

    @staticmethod
    def page_hash(clean_text: str) -> str:
        """Fingerprint of a site, mixing in SCHEMA_VERSION so a schema/prompt
        change forces re-extraction even when the page text is unchanged."""
        payload = f"{SCHEMA_VERSION}\n{clean_text}".encode("utf-8")
        return "sha256:" + hashlib.sha256(payload).hexdigest()[:32]

    def extract_structured(self, clean_text: str) -> dict:
        """Call Gemini with the response schema; return the parsed dict ({} on parse failure)."""
        model = genai.GenerativeModel(GEMINI_MODEL)
        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": EXTRACTION_SCHEMA,
            "temperature": 0.0,
        }
        prompt = EXTRACTION_PROMPT + clean_text

        delay = 2.0
        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                resp = model.generate_content(prompt, generation_config=generation_config)
                break
            except ResourceExhausted:
                if attempt == GEMINI_MAX_RETRIES:
                    raise
                print(f"    rate-limited (attempt {attempt}/{GEMINI_MAX_RETRIES}); sleeping {delay:.1f}s")
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        else:  # pragma: no cover
            raise RuntimeError("unreachable")

        time.sleep(GEMINI_RATE_LIMIT_SLEEP_SECONDS)
        try:
            return json.loads(resp.text)
        except (ValueError, AttributeError):
            print("    warning: could not parse Gemini JSON output; treating as empty")
            return {}

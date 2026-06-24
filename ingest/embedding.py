"""Gemini embedding, paced and retried for the free-tier 100 RPM limit."""

from __future__ import annotations

import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from config import (
    EMBED_MAX_RETRIES,
    EMBED_MODEL,
    EMBED_RATE_LIMIT_SLEEP_SECONDS,
    EMBED_TASK_TYPE,
)


def _embed_one(text: str) -> list[float]:
    """Embed a single text with backoff on Gemini's 100 RPM free-tier limit."""
    delay = 2.0
    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type=EMBED_TASK_TYPE,
            )
            return result["embedding"]
        except ResourceExhausted:
            if attempt == EMBED_MAX_RETRIES:
                raise
            print(f"    rate-limited (attempt {attempt}/{EMBED_MAX_RETRIES}); sleeping {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts serially, paced to stay under the 100 RPM free-tier limit."""
    embeddings: list[list[float]] = []
    for text in texts:
        embeddings.append(_embed_one(text))
        time.sleep(EMBED_RATE_LIMIT_SLEEP_SECONDS)
    return embeddings

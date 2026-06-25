"""Mistral embeddings, batched + retried for the free-tier limits.

Mistral's embeddings endpoint accepts a list of inputs per request, so we send
texts in batches (EMBED_BATCH_SIZE) rather than one-per-request. This collapses
hundreds of calls into a handful, sidestepping the per-day request cap that the
old Gemini embedding path hit.
"""

from __future__ import annotations

import os
import time

from mistralai import Mistral
from mistralai.models import SDKError

from config import (
    EMBED_BATCH_SIZE,
    EMBED_MAX_RETRIES,
    EMBED_MODEL,
    EMBED_RATE_LIMIT_SLEEP_SECONDS,
)

_client: Mistral | None = None


def _get_client() -> Mistral:
    global _client
    if _client is None:
        key = os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY env required for embedding")
        _client = Mistral(api_key=key)
    return _client


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in a single request, with backoff on 429."""
    client = _get_client()
    delay = 2.0
    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, inputs=texts)
            return [d.embedding for d in resp.data]
        except SDKError as e:
            if getattr(e, "status_code", None) != 429 or attempt == EMBED_MAX_RETRIES:
                raise
            print(f"    rate-limited (attempt {attempt}/{EMBED_MAX_RETRIES}); sleeping {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches, paced under the free-tier RPM limit."""
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        embeddings.extend(_embed_batch(batch))
        time.sleep(EMBED_RATE_LIMIT_SLEEP_SECONDS)
    return embeddings

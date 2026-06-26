"""Document-side Mistral embeddings, batched + retried for the free-tier limits.

Mistral's endpoint accepts a list of inputs per request, so texts go out in
batches (EMBED_BATCH_SIZE) rather than one-per-request — collapsing hundreds of
calls into a handful and sidestepping the per-day request cap.
"""

from __future__ import annotations

import os
import time

from mistralai import Mistral
from mistralai.models import SDKError

from shared.config import (
    EMBED_BATCH_SIZE,
    EMBED_MAX_RETRIES,
    EMBED_MODEL,
    EMBED_RATE_LIMIT_SLEEP_SECONDS,
)


class DocumentEmbedder:
    """Embeds document chunks with Mistral, batched and paced under the RPM limit."""

    def __init__(self, client: Mistral | None = None, model: str = EMBED_MODEL):
        self.model = model
        self._client = client

    @property
    def client(self) -> Mistral:
        """Lazily build the Mistral client from MISTRAL_API_KEY."""
        if self._client is None:
            key = os.environ.get("MISTRAL_API_KEY")
            if not key:
                raise RuntimeError("MISTRAL_API_KEY env required for embedding")
            self._client = Mistral(api_key=key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches, paced under the free-tier RPM limit."""
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            embeddings.extend(self._embed_batch(batch))
            time.sleep(EMBED_RATE_LIMIT_SLEEP_SECONDS)
        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in a single request, with backoff on 429."""
        delay = 2.0
        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                resp = self.client.embeddings.create(model=self.model, inputs=texts)
                return [d.embedding for d in resp.data]
            except SDKError as e:
                if getattr(e, "status_code", None) != 429 or attempt == EMBED_MAX_RETRIES:
                    raise
                print(f"    rate-limited (attempt {attempt}/{EMBED_MAX_RETRIES}); sleeping {delay:.1f}s")
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")

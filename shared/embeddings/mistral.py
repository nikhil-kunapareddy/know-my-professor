"""Mistral embeddings — the one implementation both ingest and serving use."""

from __future__ import annotations

import os
import time

from shared.config import (
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    EMBED_MAX_RETRIES,
    EMBED_MODEL,
)
from shared.retry import with_backoff

from .base import Embedder


def _is_rate_limited(exc: BaseException) -> bool:
    """True for Mistral's 429. Imported lazily so tests need no live SDK."""
    try:
        from mistralai.models import SDKError
    except ImportError:  # pragma: no cover - SDK always present where this runs
        return False
    return isinstance(exc, SDKError) and getattr(exc, "status_code", None) == 429


class MistralEmbedder(Embedder):
    """Batched Mistral embeddings.

    Mistral's endpoint takes a list of inputs per request, so texts go out in
    batches rather than one-per-request — collapsing hundreds of calls into a
    handful and staying under the free tier's request cap.

    ``pace_seconds`` is the pause held after each request. Ingest sets it to
    space out a long run; the serving path leaves it at 0 so a single query
    embed adds no latency. Making it a parameter is what lets one class serve
    both without one silently taxing the other.
    """

    api_key_env = "MISTRAL_API_KEY"

    def __init__(
        self,
        client=None,
        model: str = EMBED_MODEL,
        dim: int = EMBED_DIM,
        pace_seconds: float = 0.0,
        batch_size: int = EMBED_BATCH_SIZE,
        max_attempts: int = EMBED_MAX_RETRIES,
    ):
        self.model = model
        self.dim = dim
        self.pace_seconds = pace_seconds
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self._client = client

    @property
    def client(self):
        """Lazily build the Mistral client from MISTRAL_API_KEY."""
        if self._client is None:
            from mistralai import Mistral

            key = os.environ.get("MISTRAL_API_KEY")
            if not key:
                raise RuntimeError("MISTRAL_API_KEY env required for embedding")
            self._client = Mistral(api_key=key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches, pausing ``pace_seconds`` after each request."""
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            embeddings.extend(self._embed_batch(texts[i : i + self.batch_size]))
            if self.pace_seconds:
                time.sleep(self.pace_seconds)
        return embeddings

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one batch in a single request, backing off on 429."""

        def call():
            resp = self.client.embeddings.create(model=self.model, inputs=texts)
            return [d.embedding for d in resp.data]

        return with_backoff(
            call,
            is_retryable=_is_rate_limited,
            max_attempts=self.max_attempts,
            label="mistral embed",
        )

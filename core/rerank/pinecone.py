"""Reranking via Pinecone's hosted inference endpoint.

Note on the module name: this file is ``core.rerank.pinecone`` while the vector
SDK is the top-level ``pinecone``. They never collide here because this module
does not import the SDK at all -- ``pinecone==5.4.2``, the version pinned in the
``api`` extra, has no ``inference`` surface, so reranking would need an SDK
major-version bump. The REST endpoint is stable and ``requests`` is already
pinned across five other extras, so this calls it directly.

Chosen model is ``bge-reranker-v2-m3``: the free Starter tier covers it and
``pinecone-rerank-v0`` but not the Cohere models, and its 1024-token sequence
limit beats pinecone-rerank-v0's 512 -- long biography chunks would otherwise be
cut before they are scored.
"""

from __future__ import annotations

import os
import time

from core.retrieval.base import RetrievalResult
from shared.retry import with_backoff

from .base import Reranker

#: Pinecone's inference host, which is NOT the per-index data-plane host.
RERANK_URL = "https://api.pinecone.io/rerank"

#: Pinecone dates its API; an unset header resolves to the account's default,
#: which can move under us. Pin it.
API_VERSION = "2025-04"

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3

#: Substrings that mark a 4xx as "the allowance is gone" rather than "slow
#: down". Matched case-insensitively against the response body.
#:
#: UNVERIFIED: the exact wording Pinecone returns when the monthly rerank quota
#: is exhausted has not been seen in the wild -- it may be a 429, a 402, or a
#: 403. Uptime does not depend on getting this right (FailOpenReranker degrades
#: on ANY error), only the decision to stop retrying does. Tighten this once a
#: real exhaustion response has been observed.
_QUOTA_MARKERS = ("quota", "exceeded your", "monthly limit", "plan limit", "upgrade")


class RerankHTTPError(RuntimeError):
    """A non-2xx from the rerank endpoint, carrying enough to classify it."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"pinecone rerank returned {status_code}: {body[:200]}")
        self.status_code = status_code
        self.body = body


def _is_rate_limited(error: BaseException) -> bool:
    """True for a 429 that looks like pacing, not an exhausted allowance.

    A spent monthly quota is excluded deliberately: retrying it burns the
    request budget on calls that cannot succeed until the month rolls over.
    """
    if not isinstance(error, RerankHTTPError):
        return False
    return error.status_code == 429 and not _looks_like_quota(error)


def _looks_like_quota(error: RerankHTTPError) -> bool:
    if error.status_code in (402, 403):
        return True
    body = error.body.lower()
    return any(marker in body for marker in _QUOTA_MARKERS)


class PineconeReranker(Reranker):
    """Cross-encoder reranking over Pinecone Inference.

    ``session`` is injectable so tests never touch the network; it is the same
    seam ``MistralEmbedder``/``AnthropicGenerator`` expose as ``client``.
    """

    default_model = "bge-reranker-v2-m3"
    api_key_env = "PINECONE_API_KEY"
    max_documents = 100

    def __init__(
        self,
        session=None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_RETRIES,
        sleep=time.sleep,
    ):
        self.model = model or self.default_model
        self.timeout = timeout
        self.max_attempts = max_attempts
        # Forwarded to with_backoff, which takes it precisely so tests do not
        # spend real seconds. Monkeypatching cannot reach it: with_backoff
        # binds time.sleep as a default argument at definition time.
        self.sleep = sleep
        self._session = session

    @property
    def session(self):
        """Lazily build a requests session carrying the Pinecone headers."""
        if self._session is None:
            import requests

            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} env required for reranking")
            session = requests.Session()
            session.headers.update(
                {
                    "Api-Key": key,
                    "Content-Type": "application/json",
                    "X-Pinecone-Api-Version": API_VERSION,
                }
            )
            self._session = session
        return self._session

    def is_quota_exhausted(self, error: BaseException) -> bool:
        """Whether this error means the month's rerank allowance is spent."""
        return isinstance(error, RerankHTTPError) and _looks_like_quota(error)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Score every result against the query and return them best first."""
        if not results:
            return []
        if len(results) > self.max_documents:
            # A configuration error, not a runtime one: TOP_K * len(namespaces)
            # exceeded what the model accepts. Raised rather than silently
            # dropping the tail, and caught upstream by FailOpenReranker so a
            # misconfiguration degrades the answer instead of 502-ing it.
            raise ValueError(
                f"{self.model} accepts at most {self.max_documents} documents, got "
                f"{len(results)}; lower TOP_K (it is per namespace, so the ceiling is "
                f"{self.max_documents} / number of namespaces searched)"
            )

        scored = self._request(query, [r.metadata.get("text", "") for r in results], top_n)

        ranked: list[RetrievalResult] = []
        for entry in scored:
            result = results[entry["index"]]
            result.rerank_score = float(entry["score"])
            ranked.append(result)
        return ranked

    def _request(self, query: str, texts: list[str], top_n: int | None) -> list[dict]:
        """POST one rerank call, backing off on 429, and return its ``data``."""

        payload = {
            "model": self.model,
            "query": query,
            "documents": [{"id": str(i), "text": text} for i, text in enumerate(texts)],
            "top_n": top_n if top_n is not None else len(texts),
            "return_documents": False,
            # Chunks longer than the model's sequence limit are cut rather than
            # rejected; without this a single long biography fails the request.
            "parameters": {"truncate": "END"},
        }

        def call():
            response = self.session.post(RERANK_URL, json=payload, timeout=self.timeout)
            if response.status_code >= 400:
                raise RerankHTTPError(response.status_code, response.text)
            return response.json().get("data", [])

        return with_backoff(
            call,
            is_retryable=_is_rate_limited,
            max_attempts=self.max_attempts,
            label="pinecone rerank",
            sleep=self.sleep,
        )

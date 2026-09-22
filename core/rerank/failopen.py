"""Keep answering when the reranker cannot.

Reranking is an enhancement, not a dependency: the system answered questions
without one for its whole life, and the degraded path here is *exactly* that
behaviour -- cosine order, nothing truncated, every retrieved chunk handed to
the model. So a rerank provider that is rate-limited, out of monthly quota, or
simply unreachable must cost relevance and never availability.

The free Pinecone Starter tier allows 500 rerank requests per month and one
``/chat`` call spends one, so exhaustion is an expected operating state rather
than an exotic failure.

Wrapping rather than building this into the provider follows
``evaluation/deepeval_eval/judge.py``'s ``CachingJudge``: same interface in and
out, composed only when wanted.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable

from core.retrieval.base import RetrievalResult

from .base import Reranker

logger = logging.getLogger("kmp.rerank")


class FailOpenReranker(Reranker):
    """Wraps a ``Reranker`` so any failure degrades instead of raising.

    Three failure grades, because they deserve different responses:

    - **transient** (a blip, a 429 that outlasted its backoff): degrade this one
      request, keep the provider enabled, try again next time.
    - **quota exhausted** (``inner.is_quota_exhausted`` says so): latch off, so
      later requests skip the provider entirely rather than paying a round trip
      per request to be told no again.
    - **persistently broken** (``consecutive_failure_limit`` failures in a row,
      whatever their kind): latch off anyway. Classification decides how fast we
      stop asking; this decides that we stop asking at all, which is what makes
      an on-by-default reranker safe against a failure mode nobody has seen yet.

    ``retry_after_seconds`` re-arms the latch so a long-lived revision recovers
    when the month rolls over; 0 means stay off until the process restarts.
    """

    def __init__(
        self,
        inner: Reranker,
        retry_after_seconds: float = 3600.0,
        consecutive_failure_limit: int = 5,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.inner = inner
        self.model = inner.model
        self.default_model = inner.default_model
        self.api_key_env = inner.api_key_env
        self.max_documents = inner.max_documents
        self.retry_after_seconds = retry_after_seconds
        #: Latch after this many back-to-back failures of ANY kind, even ones
        #: ``is_quota_exhausted`` does not recognise.
        #:
        #: This is the safety net for a genuinely unknown: nobody has yet seen
        #: what Pinecone returns when the monthly rerank allowance runs out --
        #: it may be a 429, a 402, or a 403 with wording the matcher misses.
        #: Without this, an unrecognised exhaustion would be treated as
        #: transient forever: every request would pay a doomed call plus its
        #: backoff for the rest of the month. Classification decides how FAST
        #: we give up; this decides THAT we give up.
        self.consecutive_failure_limit = consecutive_failure_limit
        self._clock = clock
        self._lock = threading.Lock()
        #: Deadline after which the latch re-arms; None when enabled.
        self._disabled_until: float | None = None
        self._consecutive_failures = 0

    @property
    def degraded(self) -> bool:
        """Whether the latch is currently holding the provider off."""
        with self._lock:
            return self._disabled_until is not None

    def is_quota_exhausted(self, error: BaseException) -> bool:
        return self.inner.is_quota_exhausted(error)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank, or hand back the input untouched if that is not possible.

        The returned list is the input list when degraded -- same objects, same
        order, every ``rerank_score`` still ``None``. That is the signal the
        pipeline uses to skip the rerank cutoff, which is what stops a degraded
        provider from emptying the context and turning answers into refusals.
        """
        if not results:
            return []
        if self._latched():
            return list(results)

        try:
            ranked = self.inner.rerank(query, results, top_n)
        except Exception as error:  # noqa: BLE001 - degrading is the point
            self._degrade(error)
            return list(results)

        with self._lock:
            self._consecutive_failures = 0
        return ranked

    def _latched(self) -> bool:
        """Is the provider currently disabled? Re-arms an expired latch."""
        with self._lock:
            if self._disabled_until is None:
                return False
            if self._clock() < self._disabled_until:
                return True
            # Half-open: let exactly this request try, and re-latch if it
            # fails. The failure count resets too, or a single probe failure
            # would immediately re-trip a limit reached an hour ago.
            self._disabled_until = None
            self._consecutive_failures = 0
            logger.info(
                "reranker re-armed",
                extra={"context": {"model": self.model}},
            )
            return False

    def _degrade(self, error: BaseException) -> None:
        """Record a failure, latching off if the allowance is spent.

        Logged under the lock and only on the transition, so 80 concurrent
        requests hitting the same exhausted quota produce one WARNING rather
        than eighty -- the whole point of noticing is that the line is findable.
        """
        exhausted = self.inner.is_quota_exhausted(error)

        with self._lock:
            if self._disabled_until is not None:
                return  # already latched; stay quiet rather than log per request
            self._consecutive_failures += 1
            failures = self._consecutive_failures
            gave_up = failures >= self.consecutive_failure_limit
            if exhausted or gave_up:
                self._disabled_until = (
                    math.inf
                    if self.retry_after_seconds <= 0
                    else self._clock() + self.retry_after_seconds
                )

        latched = exhausted or gave_up
        logger.warning(
            "reranker disabled" if latched else "rerank failed, answering unranked",
            extra={
                "context": {
                    "model": self.model,
                    "error": type(error).__name__,
                    "detail": str(error)[:200],
                    "quota_exhausted": exhausted,
                    "consecutive_failures": failures,
                    # Distinguishes "we recognised the quota error" from "we
                    # gave up guessing" — the second means _QUOTA_MARKERS needs
                    # widening, and this field is how you would ever find out.
                    "latched_on": ("quota" if exhausted else "failure_limit") if latched else None,
                    "retry_after_s": self.retry_after_seconds if latched else 0,
                }
            },
        )

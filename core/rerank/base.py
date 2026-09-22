"""The reranking interface the pipeline depends on.

Mirrors ``core.llm.base.Generator``: the pipeline is written against this, so
swapping rerank providers never touches orchestration.

A reranker is a *cross-encoder*, not another embedder. Retrieval scores a query
and a chunk separately and compares the two vectors, which is what makes the
document side precomputable and the search fast. A reranker reads the query and
one chunk **together** and emits a single relevance score, so it sees word-level
interaction that cosine cannot -- at the cost of one forward pass per chunk,
which is why it runs over the handful retrieval returned rather than the corpus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.retrieval.base import RetrievalResult


class Reranker(ABC):
    """Re-scores retrieved chunks against the query, best first."""

    #: Provider-facing model identifier, resolved in ``__init__``.
    model: str
    #: Model used when the caller names none. Lives on the provider rather than
    #: in shared.config because a model id is meaningless across providers.
    default_model: str
    #: Env var holding this provider's credential, so a component can fail at
    #: startup on a missing key instead of on the first request.
    api_key_env: str | None = None
    #: Most chunks this provider will score in one call. The pipeline's
    #: ``top_k`` is per namespace, so the real ceiling is
    #: ``max_documents / len(CHAT_NAMESPACES)``.
    max_documents: int = 100

    def is_quota_exhausted(self, error: BaseException) -> bool:
        """Whether ``error`` means this provider's allowance is spent.

        Distinguishes "retry me" from "stop asking". ``FailOpenReranker`` reads
        it to decide between degrading for one request and latching off
        entirely, so a provider that cannot tell the difference should keep the
        default: treating everything as transient costs a wasted call per
        request, whereas latching wrongly silently disables reranking.
        """
        return False

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        """Return ``results`` ordered by cross-encoder relevance, best first.

        Every returned result carries a ``rerank_score``; implementations must
        not write over ``score``, which stays the cosine similarity the floor
        filters on. ``top_n`` truncates the returned list, and ``None`` keeps
        all of them.

        Returning the input unchanged (every ``rerank_score`` still ``None``) is
        a legitimate result: it is how a degraded provider reports "I could not
        rank these", and the pipeline reads it as "do not apply the cutoff".
        """
        ...

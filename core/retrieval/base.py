"""Retrieval strategy interface and result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its similarity score and metadata."""

    document_id: str
    score: float
    metadata: dict = field(default_factory=dict)


class Retriever(ABC):
    """Abstract retrieval strategy.

    Implementations take a pre-computed query embedding and return the most
    similar chunks. Keeping embedding out of the retriever lets the query
    vector be reused (and lets a future hybrid/BM25 strategy slot in behind the
    same interface).
    """

    @abstractmethod
    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[RetrievalResult]:
        """Return the top-k most similar chunks for the given query embedding.

        ``filters`` narrows the search by chunk metadata (e.g. ``section_type``)
        and ``namespace`` partitions the index. Both are optional and exist so a
        corpus holding more than one kind of entity can be queried without one
        kind drowning out the other — a person query should not compete with
        course chunks for the same top-k slots.
        """
        ...

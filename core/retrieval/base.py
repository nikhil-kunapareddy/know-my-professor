"""Retrieval strategy interface and result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


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
    def retrieve(self, query_embedding: List[float], top_k: int) -> List[RetrievalResult]:
        """Return the top-k most similar chunks for the given query embedding."""
        ...

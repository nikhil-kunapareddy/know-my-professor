"""Semantic retrieval backed by a Pinecone index."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import RetrievalResult, Retriever


class PineconeRetriever(Retriever):
    """Cosine-similarity search over a Pinecone serverless index.

    Wraps a live ``pinecone.Index`` handle; the caller is responsible for
    constructing it (so the same client can be reused across requests).
    """

    def __init__(self, index):
        self.index = index

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[RetrievalResult]:
        """Query the index and map each match to a RetrievalResult."""
        # Only pass the optional args when set: older index handles reject an
        # explicit filter=None, and an empty namespace is not the default one.
        kwargs: dict[str, Any] = {}
        if filters:
            kwargs["filter"] = dict(filters)
        if namespace:
            kwargs["namespace"] = namespace

        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            **kwargs,
        )
        matches = response.matches or []
        return [
            RetrievalResult(
                document_id=match.id,
                score=float(match.score),
                metadata=match.metadata or {},
            )
            for match in matches
        ]

"""Semantic retrieval backed by a Pinecone index."""

from __future__ import annotations

from typing import List

from .base import Retriever, RetrievalResult


class PineconeRetriever(Retriever):
    """Cosine-similarity search over a Pinecone serverless index.

    Wraps a live ``pinecone.Index`` handle; the caller is responsible for
    constructing it (so the same client can be reused across requests).
    """

    def __init__(self, index):
        self.index = index

    def retrieve(self, query_embedding: List[float], top_k: int) -> List[RetrievalResult]:
        """Query the index and map each match to a RetrievalResult."""
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
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

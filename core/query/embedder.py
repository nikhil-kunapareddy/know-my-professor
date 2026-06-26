"""Query-side embedding.

Embeds the user's question with the SAME model/dim as the ingest job (both pull
from shared.config), so query and document vectors live in the same space.
"""

from __future__ import annotations

from typing import List

from shared.config import EMBED_MODEL


class QueryEmbedder:
    """Embeds a query string into a vector using the Mistral embeddings API."""

    def __init__(self, client, model: str = EMBED_MODEL):
        self.client = client
        self.model = model

    def embed(self, text: str) -> List[float]:
        """Return the embedding vector for a single query string."""
        response = self.client.embeddings.create(model=self.model, inputs=[text])
        return response.data[0].embedding

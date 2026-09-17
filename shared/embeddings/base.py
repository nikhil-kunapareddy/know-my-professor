"""The embedding interface both sides of the system depend on.

Query vectors and document vectors must come from the same model in the same
space, or similarity search is meaningless. That used to be enforced by two
separate classes agreeing to read the same constant; here there is one
implementation per provider and both callers use it, so the guarantee is
structural rather than a convention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Turns text into vectors for one provider/model."""

    #: Provider-facing model identifier, e.g. ``"mistral-embed-2312"``.
    model: str
    #: Vector width. Must match the Pinecone index, which is fixed at creation.
    dim: int
    #: Env var holding this provider's credential, so a component can fail at
    #: startup on a missing key instead of on the first request.
    api_key_env: str | None = None

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, batched and paced as the provider requires."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Deliberately routed through ``embed_texts`` so the serving path cannot
        drift from the ingest path.
        """
        return self.embed_texts([text])[0]

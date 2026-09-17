"""Building the live system an evaluation measures.

Both runners need the same thing: the embedder, the retriever, and — when an
answer is needed — the pipeline, wired exactly the way ``serving/api`` wires
them. Building that twice is how an eval quietly stops measuring production, so
it is built once here and imported by ``run_eval`` and ``run_ragas`` alike.

Nothing in this module reads a flag or prints: it turns settings into objects,
which is what makes both CLIs thin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.llm import build_generator
from core.pipeline import RAGPipeline
from core.retrieval.base import RetrievalResult, Retriever
from core.retrieval.pinecone_retriever import PineconeRetriever
from shared.embeddings import build_embedder
from shared.embeddings.base import Embedder
from shared.settings import ApiSettings


@dataclass(frozen=True)
class LiveSystem:
    """A connected embedder + retriever, and the pipeline they can drive."""

    settings: ApiSettings
    embedder: Embedder
    retriever: Retriever

    def retrieve(
        self,
        question: str,
        top_k: int,
        min_score: float = 0.0,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Embed and search, keeping only matches at or above ``min_score``.

        The floor is applied here rather than in the retriever because that is
        where the pipeline applies it too — an eval that skipped it would score
        chunks the model is never shown.
        """
        results = self.retriever.retrieve(
            self.embedder.embed_query(question), top_k, filters=filters
        )
        return [r for r in results if r.score >= min_score]

    def pipeline(
        self,
        top_k: int,
        min_score: float,
        chat_provider: str | None = None,
        chat_model: str | None = None,
    ) -> RAGPipeline:
        """The full pipeline, including the chat provider the settings name.

        The two overrides exist for model-selection experiments: everything else
        — index, embedder, top_k, floor, prompt — is held fixed, so a difference
        in the scores is a difference between the models and not between runs.
        """
        return RAGPipeline(
            embedder=self.embedder,
            retriever=self.retriever,
            generator=build_generator(
                chat_provider or self.settings.chat_provider,
                model=chat_model or self.settings.chat_model,
            ),
            top_k=top_k,
            min_score=min_score,
        )


def connect(settings: ApiSettings | None = None) -> LiveSystem:
    """Read the environment and open a connection to the live index.

    ``pinecone`` is imported here, not at module scope, so ``--help`` still
    works in an environment without the SDK or the credentials.
    """
    from pinecone import Pinecone

    resolved = settings or ApiSettings.from_env()
    index = Pinecone(api_key=resolved.pinecone_api_key).Index(resolved.index_name)
    return LiveSystem(
        settings=resolved,
        embedder=build_embedder(resolved.embed_provider),
        retriever=PineconeRetriever(index),
    )

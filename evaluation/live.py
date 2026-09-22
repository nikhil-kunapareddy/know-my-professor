"""Building the live system an evaluation measures.

Both runners need the same thing: the embedder, the retriever, and — when an
answer is needed — the pipeline, wired exactly the way ``serving/api`` wires
them. Building that twice is how an eval quietly stops measuring production, so
it is built once here and imported by ``run_eval`` and ``run_deepeval`` alike.

Nothing in this module reads a flag or prints: it turns settings into objects,
which is what makes both CLIs thin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.llm import build_generator
from core.pipeline import RAGPipeline
from core.rerank import FailOpenReranker, build_reranker
from core.rerank.base import Reranker
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
    #: None unless RERANK_PROVIDER names one, matching serving.
    reranker: Reranker | None = None

    def retrieve(
        self,
        question: str,
        top_k: int,
        min_score: float = 0.0,
        filters: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[RetrievalResult]:
        """Embed, search, apply the floor, and rerank as the pipeline would.

        The floor is applied here rather than in the retriever because that is
        where the pipeline applies it too — an eval that skipped it would score
        chunks the model is never shown.

        Reranking is mirrored here for the same reason, and the reason is
        sharper: ``run_eval`` without ``--generate`` never constructs a
        ``RAGPipeline`` at all, it calls this method. A rerank stage that lived
        only in the pipeline would be absent from the default eval run, which
        would then report the UNRANKED baseline as though it were the reranked
        result — a measurement that silently answers the wrong question.
        """
        results = self.retriever.retrieve(
            self.embedder.embed_query(question), top_k, filters=filters, namespace=namespace
        )
        kept = [r for r in results if r.score >= min_score]
        if self.reranker is None or not kept:
            return kept

        ranked = self.reranker.rerank(question, kept, self.settings.rerank_top_n)
        if not any(r.rerank_score is not None for r in ranked):
            return ranked  # degraded; the cutoff must not fire. See RAGPipeline.
        return [
            r
            for r in ranked
            if r.rerank_score is not None and r.rerank_score >= self.settings.rerank_min_score
        ]

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
            # Without this the pipeline falls back to RAGPipeline's (None,)
            # default -- the unnamed partition, empty since the people vectors
            # moved into `people`. Serving passes settings.namespaces, so this
            # was eval-only, and latent because run_eval always overrides it
            # with answer(namespace=case.namespace). The first caller to use the
            # blended path would have got 0 chunks and no error.
            namespaces=self.settings.namespaces,
            reranker=self.reranker,
            rerank_min_score=self.settings.rerank_min_score,
            rerank_top_n=self.settings.rerank_top_n,
        )


def connect(settings: ApiSettings | None = None) -> LiveSystem:
    """Read the environment and open a connection to the live index.

    ``pinecone`` is imported here, not at module scope, so ``--help`` still
    works in an environment without the SDK or the credentials.
    """
    from pinecone import Pinecone

    resolved = settings or ApiSettings.from_env()
    index = Pinecone(api_key=resolved.pinecone_api_key).Index(resolved.index_name)
    reranker = None
    if resolved.rerank_provider:
        reranker = FailOpenReranker(
            build_reranker(resolved.rerank_provider, model=resolved.rerank_model),
            retry_after_seconds=resolved.rerank_retry_after_seconds,
        )
    return LiveSystem(
        settings=resolved,
        embedder=build_embedder(resolved.embed_provider),
        retriever=PineconeRetriever(index),
        reranker=reranker,
    )

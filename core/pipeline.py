"""RAG pipeline: embed query -> retrieve -> generate.

Orchestrates the ``core`` components behind one ``answer()`` call. It depends
only on the ``Embedder``/``Retriever``/``Generator`` interfaces and has no
web-framework dependency, so it can be unit-tested or driven from a script as
easily as from the FastAPI service — and any provider can be swapped underneath
it without this file changing.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from core.llm.base import Generator
from core.llm.prompts import SYSTEM_INSTRUCTION, PromptBuilder
from core.retrieval.base import RetrievalResult, Retriever
from shared.config import DEFAULT_TOP_K, MIN_RETRIEVAL_SCORE
from shared.embeddings.base import Embedder

NO_ANSWER = "I don't have that information in my data."


@contextmanager
def _timed(into: dict[str, float], stage: str) -> Iterator[None]:
    """Record how long ``stage`` took, in milliseconds."""
    started = time.perf_counter()
    try:
        yield
    finally:
        into[stage] = round((time.perf_counter() - started) * 1000, 1)


@dataclass
class RAGResult:
    """The answer plus the ordered sources it was grounded in.

    ``sources[i]`` corresponds to citation number ``i + 1`` in the answer text.
    ``retrieved`` counts what the index returned before the relevance floor, so
    callers can tell "nothing matched" apart from "nothing matched well enough".
    ``timings_ms`` breaks the latency down by stage — the only way to know
    whether a slow answer was the embed call, the vector search, or the model.
    """

    answer: str
    sources: list[RetrievalResult] = field(default_factory=list)
    retrieved: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    """Wires the query embedder, retriever, and generator into one flow."""

    def __init__(
        self,
        embedder: Embedder,
        retriever: Retriever,
        generator: Generator,
        prompt_builder: PromptBuilder | None = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = MIN_RETRIEVAL_SCORE,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.top_k = top_k
        self.min_score = min_score

    def answer(
        self,
        question: str,
        filters: Mapping[str, Any] | None = None,
        namespace: str | None = None,
    ) -> RAGResult:
        """Embed the question, retrieve context, and generate a cited answer.

        ``namespace`` selects which partition of the index to search. It is a
        hard choice, not a ranking hint: a Pinecone query reads exactly one
        namespace, so passing the courses namespace means professor chunks
        cannot appear at all, and vice versa. Defaults to the professor corpus.
        """
        timings: dict[str, float] = {}

        with _timed(timings, "embed"):
            query_embedding = self.embedder.embed_query(question)

        with _timed(timings, "retrieve"):
            results = self.retriever.retrieve(
                query_embedding, self.top_k, filters=filters, namespace=namespace
            )

        # Vector search always returns top_k rows, however unrelated they are, so
        # without a floor the no-answer path could never fire and the model would
        # be handed junk to cite.
        relevant = [r for r in results if r.score >= self.min_score]
        if not relevant:
            return RAGResult(answer=NO_ANSWER, retrieved=len(results), timings_ms=timings)

        user_message = self.prompt_builder.build_user_message(question, relevant)
        with _timed(timings, "generate"):
            answer = self.generator.generate(SYSTEM_INSTRUCTION, user_message)

        if not answer:
            return RAGResult(answer=NO_ANSWER, retrieved=len(results), timings_ms=timings)
        return RAGResult(
            answer=answer,
            sources=relevant,
            retrieved=len(results),
            timings_ms=timings,
        )

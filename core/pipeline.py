"""RAG pipeline: embed query -> retrieve -> generate.

Orchestrates the ``core`` components behind one ``answer()`` call. It is a plain
object with no web-framework dependency, so it can be unit-tested or driven from
a script as easily as from the FastAPI service.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from core.llm.generator import AnswerGenerator
from core.llm.prompts import SYSTEM_INSTRUCTION, PromptBuilder
from core.query.embedder import QueryEmbedder
from core.retrieval.base import Retriever, RetrievalResult
from shared.config import DEFAULT_TOP_K

NO_ANSWER = "I don't have that information in my data."


@dataclass
class RAGResult:
    """The answer plus the ordered sources it was grounded in.

    ``sources[i]`` corresponds to citation number ``i + 1`` in the answer text.
    """

    answer: str
    sources: List[RetrievalResult] = field(default_factory=list)


class RAGPipeline:
    """Wires the query embedder, retriever, and generator into one flow."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        retriever: Retriever,
        generator: AnswerGenerator,
        prompt_builder: PromptBuilder | None = None,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.top_k = top_k

    def answer(self, question: str) -> RAGResult:
        """Embed the question, retrieve context, and generate a cited answer."""
        query_embedding = self.embedder.embed(question)
        results = self.retriever.retrieve(query_embedding, self.top_k)
        if not results:
            return RAGResult(answer=NO_ANSWER, sources=[])

        user_message = self.prompt_builder.build_user_message(question, results)
        answer = self.generator.generate(SYSTEM_INSTRUCTION, user_message)
        return RAGResult(answer=answer or NO_ANSWER, sources=results)

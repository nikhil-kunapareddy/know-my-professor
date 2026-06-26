"""Unit tests for the core RAG pipeline orchestration (fully offline, fakes only)."""

from __future__ import annotations

from core.llm.prompts import PromptBuilder
from core.pipeline import NO_ANSWER, RAGPipeline
from core.retrieval.base import RetrievalResult


class _FakeEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3]


class _FakeRetriever:
    def __init__(self, results):
        self._results = results
        self.seen = None

    def retrieve(self, query_embedding, top_k):
        self.seen = (query_embedding, top_k)
        return self._results[:top_k]


class _FakeGenerator:
    def __init__(self, answer="generated answer [1]"):
        self.answer = answer
        self.last_user_message = None

    def generate(self, system_instruction, user_message):
        self.last_user_message = user_message
        return self.answer


def _result(slug, name):
    return RetrievalResult(
        document_id=f"{slug}#biography",
        score=0.9,
        metadata={"professor_name": name, "professor_title": "Prof",
                  "section_type": "biography", "url": "https://x/", "text": f"{name} bio"},
    )


def test_pipeline_returns_answer_and_ordered_sources():
    retriever = _FakeRetriever([_result("a", "Ann"), _result("b", "Bob")])
    gen = _FakeGenerator()
    pipe = RAGPipeline(_FakeEmbedder(), retriever, gen, top_k=2)

    out = pipe.answer("who works on PL?")
    assert out.answer == "generated answer [1]"
    assert [s.document_id for s in out.sources] == ["a#biography", "b#biography"]
    assert retriever.seen == ([0.1, 0.2, 0.3], 2)
    # the generator saw numbered context built from the same ordered sources
    assert "[1] Ann" in gen.last_user_message
    assert "[2] Bob" in gen.last_user_message


def test_pipeline_no_matches_short_circuits_without_calling_generator():
    gen = _FakeGenerator(answer="should not be used")
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever([]), gen, top_k=5)

    out = pipe.answer("obscure question")
    assert out.answer == NO_ANSWER
    assert out.sources == []
    assert gen.last_user_message is None  # generator never invoked


def test_pipeline_falls_back_when_generator_returns_empty():
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever([_result("a", "Ann")]),
                       _FakeGenerator(answer=""), top_k=1)
    assert pipe.answer("q").answer == NO_ANSWER


def test_prompt_builder_numbers_context_blocks():
    builder = PromptBuilder()
    msg = builder.build_user_message("q?", [_result("a", "Ann"), _result("b", "Bob")])
    assert msg.index("[1] Ann") < msg.index("[2] Bob")
    assert msg.endswith("Question: q?\n")

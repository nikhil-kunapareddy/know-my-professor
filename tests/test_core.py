"""Unit tests for the core RAG pipeline orchestration (fully offline, fakes only)."""

from __future__ import annotations

from core.llm.base import Generation
from core.llm.prompts import PromptBuilder
from core.pipeline import NO_ANSWER, RAGPipeline
from core.retrieval.base import RetrievalResult


class _FakeEmbedder:
    def __init__(self):
        self.seen: list[str] = []

    def embed_texts(self, texts):
        self.seen.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return self.embed_texts([text])[0]


class _FakeRetriever:
    def __init__(self, results):
        self._results = results
        self.seen = None
        self.seen_filters = None

    def retrieve(self, query_embedding, top_k, filters=None, namespace=None):
        self.seen = (query_embedding, top_k)
        self.seen_filters = filters
        return self._results[:top_k]


class _FakeGenerator:
    def __init__(self, answer="generated answer [1]"):
        self.answer = answer
        self.last_user_message = None

    def generate(self, system_instruction, user_message):
        self.last_user_message = user_message
        return Generation(text=self.answer, input_tokens=5272, output_tokens=420)


def _result(slug, name, score=0.9):
    return RetrievalResult(
        document_id=f"{slug}#biography",
        score=score,
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


def test_pipeline_embeds_the_query_through_the_shared_batch_path():
    """The query goes through embed_texts, the same call ingest uses."""
    embedder = _FakeEmbedder()
    pipe = RAGPipeline(embedder, _FakeRetriever([_result("a", "Ann")]), _FakeGenerator())
    pipe.answer("who works on PL?")
    assert embedder.seen == ["who works on PL?"]


def test_pipeline_no_matches_short_circuits_without_calling_generator():
    gen = _FakeGenerator(answer="should not be used")
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever([]), gen, top_k=5)

    out = pipe.answer("obscure question")
    assert out.answer == NO_ANSWER
    assert out.sources == []
    assert out.retrieved == 0
    assert gen.last_user_message is None  # generator never invoked


def test_pipeline_drops_matches_below_the_score_floor():
    """Vector search always returns rows; weak ones must not reach the model."""
    gen = _FakeGenerator(answer="should not be used")
    weak = [_result("a", "Ann", score=0.10), _result("b", "Bob", score=0.05)]
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever(weak), gen, min_score=0.35)

    out = pipe.answer("something unrelated")
    assert out.answer == NO_ANSWER
    assert out.sources == []
    assert out.retrieved == 2  # retrieved, then rejected as too weak
    assert gen.last_user_message is None


def test_pipeline_keeps_only_matches_above_the_floor():
    mixed = [_result("a", "Ann", score=0.80), _result("b", "Bob", score=0.10)]
    gen = _FakeGenerator()
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever(mixed), gen, min_score=0.35)

    out = pipe.answer("q")
    assert [s.document_id for s in out.sources] == ["a#biography"]
    assert out.retrieved == 2
    assert "Bob" not in gen.last_user_message


def test_pipeline_passes_filters_to_the_retriever():
    retriever = _FakeRetriever([_result("a", "Ann")])
    pipe = RAGPipeline(_FakeEmbedder(), retriever, _FakeGenerator())

    pipe.answer("q", filters={"section_type": "biography"})
    assert retriever.seen_filters == {"section_type": "biography"}


def test_pipeline_falls_back_when_generator_returns_empty():
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever([_result("a", "Ann")]),
                       _FakeGenerator(answer=""), top_k=1)
    out = pipe.answer("q")
    assert out.answer == NO_ANSWER
    assert out.sources == []  # no sources for a non-answer


def test_pipeline_records_stage_timings():
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever([_result("a", "Ann")]), _FakeGenerator())
    out = pipe.answer("q")
    assert set(out.timings_ms) == {"embed", "retrieve", "generate"}
    assert all(v >= 0 for v in out.timings_ms.values())


def test_prompt_builder_numbers_context_blocks():
    builder = PromptBuilder()
    msg = builder.build_user_message("q?", [_result("a", "Ann"), _result("b", "Bob")])
    assert msg.index("[1] Ann") < msg.index("[2] Bob")
    assert msg.endswith("Question: q?\n")

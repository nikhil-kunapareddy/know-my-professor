"""Unit tests for the core RAG pipeline orchestration (fully offline, fakes only)."""

from __future__ import annotations

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
        return self.answer


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


# --- reranking -------------------------------------------------------------


class _FakeReranker:
    """Reorders by a canned score map, or degrades and hands back the input.

    ``degraded=True`` models the operating state that matters most: the
    provider is configured but could not rank (out of quota, unreachable), so
    it returns the results untouched with every ``rerank_score`` still None.
    """

    def __init__(self, scores=None, degraded=False):
        self.scores = scores or {}
        self.degraded = degraded
        self.seen = None
        self.calls = 0

    def rerank(self, query, results, top_n=None):
        self.calls += 1
        self.seen = (query, [r.document_id for r in results])
        if self.degraded:
            return list(results)
        for result in results:
            result.rerank_score = self.scores.get(result.document_id, 0.0)
        ranked = sorted(results, key=lambda r: r.rerank_score, reverse=True)
        return ranked[:top_n] if top_n is not None else ranked


def test_pipeline_reorders_sources_by_rerank_score():
    """Cosine order says Ann then Bob; the cross-encoder disagrees, and wins."""
    results = [_result("a", "Ann", score=0.9), _result("b", "Bob", score=0.8)]
    reranker = _FakeReranker(scores={"a#biography": 0.1, "b#biography": 0.9})
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever(results), _FakeGenerator(),
                       top_k=2, min_score=0.0, reranker=reranker)

    out = pipe.answer("q")
    assert [s.document_id for s in out.sources] == ["b#biography", "a#biography"]
    assert out.reranked is True
    assert reranker.seen == ("q", ["a#biography", "b#biography"])


def test_pipeline_sources_match_the_order_the_prompt_used():
    """Citation [n] is resolved positionally against sources, so the two must agree."""
    gen = _FakeGenerator()
    results = [_result("a", "Ann", score=0.9), _result("b", "Bob", score=0.8)]
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever(results), gen, top_k=2, min_score=0.0,
                       reranker=_FakeReranker(scores={"a#biography": 0.1, "b#biography": 0.9}))

    out = pipe.answer("q")
    assert gen.last_user_message.index("[1] Bob") < gen.last_user_message.index("[2] Ann")
    assert [s.metadata["professor_name"] for s in out.sources] == ["Bob", "Ann"]


def test_pipeline_drops_chunks_below_the_rerank_cutoff():
    results = [_result("a", "Ann", score=0.9), _result("b", "Bob", score=0.9)]
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever(results), _FakeGenerator(),
                       top_k=2, min_score=0.0, rerank_min_score=0.5,
                       reranker=_FakeReranker(scores={"a#biography": 0.8, "b#biography": 0.2}))

    out = pipe.answer("q")
    assert [s.document_id for s in out.sources] == ["a#biography"]


def test_pipeline_rerank_cutoff_can_refuse_outright():
    """Nothing clears the bar, so the model is never asked."""
    gen = _FakeGenerator()
    results = [_result("a", "Ann", score=0.9)]
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever(results), gen, top_k=1, min_score=0.0,
                       rerank_min_score=0.5,
                       reranker=_FakeReranker(scores={"a#biography": 0.1}))

    out = pipe.answer("q")
    assert out.answer == NO_ANSWER
    assert out.retrieved == 1
    assert gen.last_user_message is None


def test_pipeline_ignores_the_rerank_cutoff_when_reranking_degraded():
    """The one that protects correctness when the free tier runs out.

    A degraded reranker returns chunks unscored. Applying a 0.5 cutoff to
    unscored chunks would drop every one of them and refuse a question the
    system can perfectly well answer — losing the reranker must cost relevance,
    not correctness.
    """
    results = [_result("a", "Ann", score=0.9), _result("b", "Bob", score=0.8)]
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever(results), _FakeGenerator(),
                       top_k=2, min_score=0.0, rerank_min_score=0.5,
                       reranker=_FakeReranker(degraded=True))

    out = pipe.answer("q")
    assert [s.document_id for s in out.sources] == ["a#biography", "b#biography"]
    assert out.answer != NO_ANSWER
    assert out.reranked is False  # how a caller tells a degraded answer apart


def test_pipeline_truncates_to_rerank_top_n():
    results = [_result(s, s.upper(), score=0.9) for s in ("a", "b", "c")]
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever(results), _FakeGenerator(),
                       top_k=3, min_score=0.0, rerank_top_n=2,
                       reranker=_FakeReranker(scores={"a#biography": 0.1,
                                                      "b#biography": 0.9,
                                                      "c#biography": 0.5}))

    out = pipe.answer("q")
    assert [s.document_id for s in out.sources] == ["b#biography", "c#biography"]


def test_pipeline_does_not_rerank_when_the_floor_left_nothing():
    """No point paying for a rerank call on an empty list."""
    reranker = _FakeReranker()
    weak = [_result("a", "Ann", score=0.01)]
    pipe = RAGPipeline(_FakeEmbedder(), _FakeRetriever(weak), _FakeGenerator(),
                       min_score=0.35, reranker=reranker)

    assert pipe.answer("q").answer == NO_ANSWER
    assert reranker.calls == 0


def test_pipeline_times_the_rerank_stage_only_when_one_runs():
    results = [_result("a", "Ann")]
    without = RAGPipeline(_FakeEmbedder(), _FakeRetriever(results), _FakeGenerator())
    assert set(without.answer("q").timings_ms) == {"embed", "retrieve", "generate"}

    with_rerank = RAGPipeline(_FakeEmbedder(), _FakeRetriever(results), _FakeGenerator(),
                              min_score=0.0, reranker=_FakeReranker())
    assert set(with_rerank.answer("q").timings_ms) == {"embed", "retrieve", "rerank", "generate"}

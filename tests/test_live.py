"""`evaluation/live.py` — the one definition of "the system under test".

Nothing covered this module before, which is how the namespace bug below
survived long enough to reach CLAUDE.md as a TODO. Everything here is offline:
`LiveSystem` takes its embedder, retriever and reranker by injection, so no
test connects to Pinecone.
"""

from __future__ import annotations

from core.retrieval.base import RetrievalResult
from evaluation.live import LiveSystem
from shared.config import COURSES_NAMESPACE, PEOPLE_NAMESPACE, RESEARCH_NAMESPACE
from shared.settings import ApiSettings


def _result(slug, score=0.9):
    return RetrievalResult(
        document_id=f"{slug}#biography", score=score, metadata={"text": f"{slug} bio"}
    )


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class _FakeRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query_embedding, top_k, filters=None, namespace=None):
        return self._results[:top_k]


class _FakeReranker:
    """Scores from a canned map, or degrades and returns the input untouched."""

    def __init__(self, scores=None, degraded=False):
        self.scores = scores or {}
        self.degraded = degraded
        self.calls = 0

    def rerank(self, query, results, top_n=None):
        self.calls += 1
        if self.degraded:
            return list(results)
        for result in results:
            result.rerank_score = self.scores.get(result.document_id, 0.0)
        ranked = sorted(results, key=lambda r: r.rerank_score, reverse=True)
        return ranked[:top_n] if top_n is not None else ranked


def _system(reranker=None, results=None, **overrides):
    return LiveSystem(
        settings=ApiSettings(pinecone_api_key="pk", **overrides),
        embedder=_FakeEmbedder(),
        retriever=_FakeRetriever(results if results is not None else []),
        reranker=reranker,
    )


def test_live_pipeline_searches_the_same_namespaces_as_serving():
    """The eval must measure the corpora production actually reads.

    `pipeline()` used to omit `namespaces=`, so it fell back to RAGPipeline's
    `(None,)` default — the unnamed partition, empty since the people vectors
    moved into `people`. Nothing raised: a namespace mismatch returns zero
    chunks, so a blended-path eval would have scored an empty index silently.
    """
    system = _system()
    pipeline = system.pipeline(top_k=11, min_score=0.35)

    assert pipeline.namespaces == system.settings.namespaces
    assert pipeline.namespaces == (PEOPLE_NAMESPACE, COURSES_NAMESPACE, RESEARCH_NAMESPACE)
    assert None not in pipeline.namespaces


def test_live_pipeline_carries_the_reranker_and_its_knobs():
    reranker = _FakeReranker()
    system = _system(reranker=reranker, rerank_min_score=0.5, rerank_top_n=8)

    pipeline = system.pipeline(top_k=11, min_score=0.35)

    assert pipeline.reranker is reranker
    assert (pipeline.rerank_min_score, pipeline.rerank_top_n) == (0.5, 8)


def test_live_retrieve_reranks_too():
    """`run_eval` without --generate never builds a RAGPipeline, it calls this.

    A rerank stage that lived only in the pipeline would be absent from the
    default eval run, which would then report the unranked baseline as though
    it were the reranked result.
    """
    reranker = _FakeReranker(scores={"b#biography": 0.9, "a#biography": 0.1})
    system = _system(reranker=reranker, results=[_result("a"), _result("b", 0.8)])

    kept = system.retrieve("q", top_k=2, min_score=0.0)

    assert [r.document_id for r in kept] == ["b#biography", "a#biography"]
    assert reranker.calls == 1


def test_live_retrieve_applies_the_rerank_cutoff():
    system = _system(
        reranker=_FakeReranker(scores={"a#biography": 0.8, "b#biography": 0.2}),
        results=[_result("a"), _result("b")],
        rerank_min_score=0.5,
    )

    kept = system.retrieve("q", top_k=2, min_score=0.0)
    assert [r.document_id for r in kept] == ["a#biography"]


def test_live_retrieve_ignores_the_cutoff_when_reranking_degraded():
    """Same guarantee the pipeline makes: a dead reranker must not empty the eval."""
    system = _system(
        reranker=_FakeReranker(degraded=True),
        results=[_result("a"), _result("b", 0.8)],
        rerank_min_score=0.5,
    )

    kept = system.retrieve("q", top_k=2, min_score=0.0)
    assert [r.document_id for r in kept] == ["a#biography", "b#biography"]


def test_live_retrieve_does_not_call_a_reranker_with_nothing_to_rank():
    reranker = _FakeReranker()
    system = _system(reranker=reranker, results=[_result("a", score=0.01)])

    assert system.retrieve("q", top_k=1, min_score=0.35) == []
    assert reranker.calls == 0


def test_live_retrieve_without_a_reranker_only_applies_the_cosine_floor():
    system = _system(results=[_result("a"), _result("b", score=0.10)])
    kept = system.retrieve("q", top_k=2, min_score=0.35)
    assert [r.document_id for r in kept] == ["a#biography"]

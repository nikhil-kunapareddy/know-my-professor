"""The rerank experiment harness (offline; no Pinecone, no reranker, no judge).

`evaluation/topk` is imported by the thing under test and must stay unmodified,
so nothing here touches it.
"""

from __future__ import annotations

from pathlib import Path

from core.retrieval.base import RetrievalResult
from evaluation.harness import EvalCase
from evaluation.rerank.experiment import (
    RerankedCase,
    corpus_of,
    read_cases,
    rerank_cases,
    retrieve_cases,
    score_pairs,
    stratum_of,
    targets_for,
    write_cases,
)
from evaluation.rerank.report import choose_cutoff, sweep_cutoffs
from evaluation.topk.experiment import JudgedChunk


def _case(case_id="people-narrow-x", namespace="people", slugs=("ann",), no_answer=False):
    return EvalCase(
        id=case_id,
        question=f"q for {case_id}?",
        expected_slugs=() if no_answer else tuple(slugs),
        namespace=namespace,
        expect_no_answer=no_answer,
    )


class _Settings:
    namespaces = ("people", "courses")
    index_name = "test-index"


class _Retriever:
    """The raw retriever — what the experiment must use."""

    def __init__(self, per_namespace):
        self.per_namespace = per_namespace
        self.seen: list[str | None] = []

    def retrieve(self, vector, k, filters=None, namespace=None):
        self.seen.append(namespace)
        return [
            RetrievalResult(document_id=d, score=s, metadata={"text": f"{d} text"})
            for d, s in self.per_namespace.get(namespace, [])
        ][:k]


class _Live:
    """Stand-in for LiveSystem, keeping its two retrieval paths DISTINCT.

    The real `LiveSystem.retrieve` applies a reranker when one is configured;
    `live.retriever.retrieve` does not. Conflating them in the fake would make
    the test below unable to tell which one the experiment called.
    """

    def __init__(self, per_namespace):
        self.settings = _Settings()
        self.retriever = _Retriever(per_namespace)
        self.embedder = self
        self.via_pipeline = 0

    @property
    def seen(self):
        return self.retriever.seen

    def embed_query(self, text):
        return [0.1, 0.2]

    def retrieve(self, *args, **kwargs):
        self.via_pipeline += 1
        return []


class _Reranker:
    def __init__(self, scores=None, degraded=False):
        self.scores = scores or {}
        self.degraded = degraded
        self.calls = 0

    def rerank(self, query, results, top_n=None):
        self.calls += 1
        if self.degraded:
            return list(results)
        for r in results:
            r.rerank_score = self.scores.get(r.document_id, 0.0)
        return sorted(results, key=lambda r: r.rerank_score, reverse=True)


# --- case classification ---------------------------------------------------


def test_blended_cases_are_recognised_by_a_null_namespace():
    assert corpus_of(_case("both-blended-cs5700", namespace=None)) == "blended"
    assert corpus_of(_case("courses-narrow-x", namespace="courses")) == "courses"


def test_the_blended_stratum_parses_here_even_though_topk_does_not_know_it():
    """evaluation.topk's STRATA is left alone, so this module carries its own."""
    from evaluation.topk.experiment import STRATA as TOPK_STRATA

    assert "blended" not in TOPK_STRATA
    assert stratum_of(_case("both-blended-cs5700", namespace=None)) == "blended"
    assert stratum_of(_case("courses-narrow-x", namespace="courses")) == "narrow"


def test_a_blended_case_searches_every_namespace():
    assert targets_for(_case(namespace=None), ("people", "courses")) == ("people", "courses")
    assert targets_for(_case(namespace="courses"), ("people", "courses")) == ("courses",)


# --- retrieval -------------------------------------------------------------


def test_retrieval_fans_out_and_sorts_the_union_by_cosine():
    live = _Live({"people": [("p1", 0.70)], "courses": [("c1", 0.90), ("c2", 0.50)]})

    cases = retrieve_cases(live, [_case("both-blended-x", namespace=None)], k=11)

    assert live.seen == ["people", "courses"]
    assert [c.document_id for c in cases[0].chunks] == ["c1", "p1", "c2"]
    assert cases[0].namespaces == ("people", "courses")


def test_a_single_namespace_case_queries_only_that_namespace():
    live = _Live({"people": [("p1", 0.7)], "courses": [("c1", 0.9)]})
    retrieve_cases(live, [_case(namespace="people")], k=11)
    assert live.seen == ["people"]


def test_retrieval_does_not_go_through_the_reranking_path():
    """`LiveSystem.retrieve` applies a reranker when one is configured.

    Using it here would rerank the BASELINE arm too, collapsing the experiment
    into comparing a thing with itself and reporting a null result that looks
    like a finding. The raw retriever must be called instead.
    """
    live = _Live({"people": [("p1", 0.7)]})

    cases = retrieve_cases(live, [_case(namespace="people")], k=11)

    assert live.via_pipeline == 0, "went through the reranking path"
    assert live.retriever.seen == ["people"]
    assert [c.document_id for c in cases[0].chunks] == ["p1"]


# --- reranking -------------------------------------------------------------


def _retrieved(*pairs):
    case = RerankedCase(case=_case(), namespaces=("people",))
    case.chunks = [
        JudgedChunk(document_id=d, score=s, rank=i, text=f"{d} text")
        for i, (d, s) in enumerate(pairs, start=1)
    ]
    return case


def test_reranking_reorders_the_same_objects():
    """Both arms must share chunk objects, or a verdict fills only one arm."""
    case = _retrieved(("a", 0.9), ("b", 0.8))
    rerank_cases(_Reranker(scores={"a": 0.1, "b": 0.9}), [case])

    assert [c.document_id for c in case.reranked] == ["b", "a"]
    assert case.reranked[0] is case.chunks[1]
    assert case.degraded is False

    case.chunks[1].relevant = True
    assert case.reranked[0].relevant is True


def test_a_degraded_reranker_is_recorded_not_silently_counted_as_no_change():
    case = _retrieved(("a", 0.9), ("b", 0.8))
    degraded = rerank_cases(_Reranker(degraded=True), [case])

    assert degraded == 1
    assert case.degraded is True
    assert [c.document_id for c in case.reranked] == ["a", "b"]
    assert case.rerank_scores == {}


def test_a_raising_reranker_degrades_rather_than_killing_the_run():
    class _Boom:
        def rerank(self, *a, **k):
            raise RuntimeError("out of quota")

    case = _retrieved(("a", 0.9))
    assert rerank_cases(_Boom(), [case]) == 1
    assert case.degraded is True


# --- the cutoff arithmetic -------------------------------------------------


def test_score_pairs_excludes_degraded_cases_and_unjudged_chunks():
    """An unjudged chunk counted as irrelevant would bias every cutoff upward."""
    good = _retrieved(("a", 0.9), ("b", 0.8))
    rerank_cases(_Reranker(scores={"a": 0.9, "b": 0.2}), [good])
    good.chunks[0].relevant = True  # b left unjudged

    dead = _retrieved(("c", 0.9))
    rerank_cases(_Reranker(degraded=True), [dead])
    dead.chunks[0].relevant = True

    assert score_pairs([good, dead]) == [(0.9, True)]


def test_the_sweep_counts_what_each_cutoff_keeps_and_drops():
    case = _retrieved(("a", 0.9), ("b", 0.8), ("c", 0.7))
    rerank_cases(_Reranker(scores={"a": 0.9, "b": 0.3, "c": 0.1}), [case])
    case.chunks[0].relevant = True
    case.chunks[1].relevant = False
    case.chunks[2].relevant = False

    at_half = next(p for p in sweep_cutoffs([case]) if p.cutoff == 0.5)
    assert (at_half.kept_relevant, at_half.kept_irrelevant) == (1, 0)
    assert (at_half.dropped_relevant, at_half.dropped_irrelevant) == (0, 2)
    assert at_half.precision == 1.0
    assert at_half.relevant_loss == 0.0


def test_the_rule_picks_the_highest_cutoff_inside_the_recall_budget():
    case = _retrieved(("a", 0.9), ("b", 0.8))
    rerank_cases(_Reranker(scores={"a": 0.9, "b": 0.1}), [case])
    case.chunks[0].relevant = True
    case.chunks[1].relevant = False

    choice = choose_cutoff(sweep_cutoffs([case]))
    assert choice.cutoff == 0.8  # the highest that still keeps the relevant one


def test_the_rule_can_answer_no_cutoff_at_all():
    """If the score does not separate the two, shipping none is the finding —
    the same shape as MIN_RETRIEVAL_SCORE turning out inert."""
    case = _retrieved(("a", 0.9), ("b", 0.8))
    rerank_cases(_Reranker(scores={"a": 0.02, "b": 0.9}), [case])
    case.chunks[0].relevant = True  # the RELEVANT chunk scored lowest
    case.chunks[1].relevant = False

    choice = choose_cutoff(sweep_cutoffs([case]))
    assert choice.cutoff is None
    assert "does not separate" in choice.reason


# --- dump / replay ---------------------------------------------------------


def test_a_recorded_run_replays_with_both_orderings_intact(tmp_path: Path):
    case = _retrieved(("a", 0.9), ("b", 0.8))
    rerank_cases(_Reranker(scores={"a": 0.1, "b": 0.9}), [case])
    case.chunks[0].relevant = False
    case.chunks[1].relevant = True

    path = tmp_path / "run.jsonl"
    write_cases(path, [case], {"k": 11, "rerank_model": "bge"})

    loaded, run = read_cases(path, {case.case.id: case.case})
    assert run["k"] == 11
    assert [c.document_id for c in loaded[0].chunks] == ["a", "b"]
    assert [c.document_id for c in loaded[0].reranked] == ["b", "a"]
    assert [c.relevant for c in loaded[0].chunks] == [False, True]


def test_replay_skips_cases_no_longer_in_the_question_set(tmp_path: Path):
    case = _retrieved(("a", 0.9))
    rerank_cases(_Reranker(scores={"a": 0.5}), [case])
    path = tmp_path / "run.jsonl"
    write_cases(path, [case], {"k": 11})

    loaded, _ = read_cases(path, {})
    assert loaded == []

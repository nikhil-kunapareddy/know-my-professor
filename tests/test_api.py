"""Unit tests for the /chat HTTP surface (offline; the pipeline is a fake).

TestClient is used WITHOUT a ``with`` block on purpose: entering the context
manager would run the lifespan, which builds real providers and demands real
API keys. These tests install a fake pipeline into the app state instead, so
they exercise routing, citation filtering, and error mapping only.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from core.pipeline import RAGResult
from core.retrieval.base import RetrievalResult
from serving.api import app as app_module
from serving.api.errors import classify_upstream_error


def _source(name: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        document_id=f"{name}#biography",
        score=score,
        metadata={
            "professor_name": name,
            "professor_title": "Prof",
            "section_type": "biography",
            "url": f"https://x/{name}",
            "text": f"{name} bio",
        },
    )


class _FakePipeline:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.seen: list[str] = []

    def answer(self, question, filters=None):
        self.seen.append(question)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def client():
    """A client whose app state holds a replaceable fake pipeline."""
    app_module.state.clear()
    yield TestClient(app_module.app)
    app_module.state.clear()


def _install(pipeline) -> None:
    app_module.state["pipeline"] = pipeline


# --- health / readiness ----------------------------------------------------


def test_health_is_liveness_only(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_starting_before_the_pipeline_exists(client):
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "starting"


def test_ready_is_503_when_the_index_is_unreachable(client):
    class _DeadIndex:
        def describe_index_stats(self):
            raise RuntimeError("connection refused")

    _install(_FakePipeline())
    app_module.state["index"] = _DeadIndex()
    app_module.state["ready_until"] = 0.0

    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "index_unreachable"


def test_ready_is_200_when_the_index_answers(client):
    class _LiveIndex:
        def describe_index_stats(self):
            return {"total_vector_count": 42}

    _install(_FakePipeline())
    app_module.state["index"] = _LiveIndex()
    app_module.state["ready_until"] = 0.0

    assert client.get("/ready").status_code == 200


# --- routing ---------------------------------------------------------------


@pytest.mark.parametrize("path", ["/chat", "/v1/chat"])
def test_both_the_versioned_route_and_the_legacy_alias_answer(client, path):
    """The frontend deploys separately, so the unversioned path must keep working."""
    _install(_FakePipeline(RAGResult(answer="Ann works on PL [1].", sources=[_source("Ann")])))

    resp = client.post(path, json={"question": "who does PL?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Ann works on PL [1]."


def test_short_question_is_rejected_by_validation(client):
    _install(_FakePipeline(RAGResult(answer="x")))
    assert client.post("/v1/chat", json={"question": "a"}).status_code == 422


def test_request_id_is_echoed_back(client):
    _install(_FakePipeline(RAGResult(answer="hi")))
    resp = client.post("/v1/chat", json={"question": "hello?"}, headers={"X-Request-ID": "abc123"})
    assert resp.headers["X-Request-ID"] == "abc123"


def test_a_request_id_is_generated_when_absent(client):
    _install(_FakePipeline(RAGResult(answer="hi")))
    resp = client.post("/v1/chat", json={"question": "hello?"})
    assert resp.headers.get("X-Request-ID")


# --- citations -------------------------------------------------------------


def test_only_cited_sources_are_returned(client):
    """Eight retrieved chunks behind a two-source answer must not all be listed."""
    sources = [_source("Ann"), _source("Bob"), _source("Cy")]
    _install(_FakePipeline(RAGResult(answer="Ann [1] and Cy [3] do PL.", sources=sources)))

    citations = client.post("/v1/chat", json={"question": "who does PL?"}).json()["citations"]
    assert [c["number"] for c in citations] == [1, 3]
    assert [c["professor_name"] for c in citations] == ["Ann", "Cy"]


def test_citation_numbers_are_not_renumbered(client):
    """[3] in the prose must keep pointing at citation 3."""
    sources = [_source("Ann"), _source("Bob"), _source("Cy")]
    _install(_FakePipeline(RAGResult(answer="Only Cy [3].", sources=sources)))

    citations = client.post("/v1/chat", json={"question": "q?"}).json()["citations"]
    assert [c["number"] for c in citations] == [3]


def test_an_uncited_answer_still_returns_provenance(client):
    sources = [_source("Ann"), _source("Bob")]
    _install(_FakePipeline(RAGResult(answer="No brackets here.", sources=sources)))

    citations = client.post("/v1/chat", json={"question": "q?"}).json()["citations"]
    assert [c["number"] for c in citations] == [1, 2]


def test_no_answer_returns_no_citations(client):
    _install(_FakePipeline(RAGResult(answer="I don't have that information in my data.", retrieved=8)))
    body = client.post("/v1/chat", json={"question": "q?"}).json()
    assert body["citations"] == []


# --- error mapping ---------------------------------------------------------


class _RateLimited(Exception):
    status_code = 429


def test_rate_limit_maps_to_429(client):
    _install(_FakePipeline(error=_RateLimited("quota exhausted for key sk-SECRET")))
    resp = client.post("/v1/chat", json={"question": "q?"})

    assert resp.status_code == 429
    assert resp.json()["error"] == "rate_limited"


def test_generic_failure_maps_to_502(client):
    _install(_FakePipeline(error=RuntimeError("boom")))
    resp = client.post("/v1/chat", json={"question": "q?"})

    assert resp.status_code == 502
    assert resp.json()["error"] == "upstream_error"


def test_upstream_error_text_never_reaches_the_client(client):
    """Provider messages can carry internal detail; they belong in logs only."""
    secret = "https://internal.provider/v1?key=sk-SECRET-abc"
    _install(_FakePipeline(error=RuntimeError(secret)))

    resp = client.post("/v1/chat", json={"question": "q?"})
    assert secret not in resp.text
    assert "sk-SECRET" not in resp.text


def test_error_body_carries_the_request_id_for_log_correlation(client):
    _install(_FakePipeline(error=RuntimeError("boom")))
    resp = client.post("/v1/chat", json={"question": "q?"}, headers={"X-Request-ID": "trace-me"})
    assert resp.json()["request_id"] == "trace-me"


# --- classifier ------------------------------------------------------------


@pytest.mark.parametrize(
    "exc,expected_status,expected_code",
    [
        (_RateLimited("x"), 429, "rate_limited"),
        (type("ResourceExhausted", (Exception,), {})(), 429, "rate_limited"),
        (type("RateLimitError", (Exception,), {})(), 429, "rate_limited"),
        (TimeoutError(), 504, "upstream_timeout"),
        (type("ReadTimeout", (Exception,), {})(), 504, "upstream_timeout"),
        (type("BadRequest", (Exception,), {"status_code": 400})(), 502, "upstream_rejected"),
        (RuntimeError("x"), 502, "upstream_error"),
    ],
)
def test_classifier_categorises_provider_errors(exc, expected_status, expected_code):
    classified = classify_upstream_error(exc)
    assert (classified.status_code, classified.code) == (expected_status, expected_code)


# --- reranker wiring in lifespan -------------------------------------------


class _Stub:
    """Stands in for any provider the lifespan builds."""

    default_model = "stub"
    max_documents = 100

    def __init__(self, name="stub"):
        self.model = name
        self.api_key_env = None

    def rerank(self, query, results, top_n=None):
        return list(results)


@pytest.fixture
def booted(monkeypatch):
    """Runs the real lifespan with every provider and Pinecone stubbed out."""
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    monkeypatch.setattr(app_module, "build_embedder", lambda *a, **k: _Stub("embed"))
    monkeypatch.setattr(app_module, "build_generator", lambda *a, **k: _Stub("chat"))
    monkeypatch.setattr(
        app_module, "Pinecone", lambda **k: type("C", (), {"Index": lambda s, n: object()})()
    )
    app_module.state.clear()
    yield
    app_module.state.clear()


def test_lifespan_builds_no_reranker_by_default(booted, monkeypatch):
    """An existing revision sets no RERANK_* var and must keep working."""
    monkeypatch.delenv("RERANK_PROVIDER", raising=False)
    called = []
    monkeypatch.setattr(app_module, "build_reranker", lambda *a, **k: called.append(a))

    with TestClient(app_module.app):
        assert app_module.state["pipeline"].reranker is None
    assert called == []


def test_lifespan_wraps_the_reranker_so_it_can_fail_open(booted, monkeypatch):
    monkeypatch.setenv("RERANK_PROVIDER", "pinecone")
    monkeypatch.setenv("RERANK_MIN_SCORE", "0.5")
    monkeypatch.setattr(app_module, "build_reranker", lambda *a, **k: _Stub("rerank"))

    with TestClient(app_module.app):
        pipeline = app_module.state["pipeline"]
        assert isinstance(pipeline.reranker, app_module.FailOpenReranker)
        assert pipeline.reranker.model == "rerank"
        assert pipeline.rerank_min_score == 0.5


def test_chat_logs_whether_the_answer_was_reranked(client, caplog):
    """False while a reranker IS configured means it degraded — the only way
    to tell which answers were served without one."""
    _install(_FakePipeline(RAGResult(answer="hi [1]", sources=[_source("Ann")], reranked=False)))

    with caplog.at_level(logging.INFO, logger="kmp.api"):
        assert client.post("/v1/chat", json={"question": "who does PL?"}).status_code == 200

    ok = [r for r in caplog.records if r.getMessage() == "chat ok"]
    assert ok and ok[0].context["reranked"] is False

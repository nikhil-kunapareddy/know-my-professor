"""Reranking: the registry, the Pinecone provider, and the fail-open wrapper.

Fully offline. The provider is driven through an injected fake session, so no
test spends a request from the 500/month free tier.
"""

from __future__ import annotations

import logging
import threading

import pytest

from core.rerank import FailOpenReranker, available_rerankers, build_reranker
from core.rerank.pinecone import RERANK_URL, PineconeReranker, RerankHTTPError
from core.retrieval.base import RetrievalResult


def _result(slug, score=0.9, text=None):
    return RetrievalResult(
        document_id=f"{slug}#biography",
        score=score,
        metadata={"text": text if text is not None else f"{slug} bio"},
    )


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeSession:
    """Records posts and replays canned responses (or raises them)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.posts: list[dict] = []

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        nxt = self.responses.pop(0) if self.responses else _Resp(payload={"data": []})
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


def _ranked(*pairs):
    return _Resp(payload={"data": [{"index": i, "score": s} for i, s in pairs]})


# --- registry --------------------------------------------------------------


def test_build_reranker_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown rerank provider"):
        build_reranker("does-not-exist")


def test_registry_lists_the_pinecone_provider():
    assert "pinecone" in available_rerankers()


def test_reranker_declares_its_credential_and_default_model():
    """Reuses the Pinecone key, so enabling reranking needs no new secret."""
    reranker = build_reranker("pinecone", session=object())
    assert reranker.api_key_env == "PINECONE_API_KEY"
    assert reranker.model == PineconeReranker.default_model == "bge-reranker-v2-m3"


def test_an_explicit_model_overrides_the_provider_default():
    assert build_reranker("pinecone", session=object(), model="pinecone-rerank-v0").model == (
        "pinecone-rerank-v0"
    )


# --- the Pinecone provider -------------------------------------------------


def test_rerank_posts_the_documents_and_maps_scores_back():
    session = _FakeSession(_ranked((1, 0.91), (0, 0.12)))
    reranker = PineconeReranker(session=session)
    results = [_result("a"), _result("b")]

    ranked = reranker.rerank("who works on compilers?", results)

    assert [r.document_id for r in ranked] == ["b#biography", "a#biography"]
    assert [r.rerank_score for r in ranked] == [0.91, 0.12]
    sent = session.posts[0]
    assert sent["url"] == RERANK_URL
    assert sent["json"]["query"] == "who works on compilers?"
    assert [d["text"] for d in sent["json"]["documents"]] == ["a bio", "b bio"]
    assert sent["json"]["parameters"] == {"truncate": "END"}


def test_rerank_does_not_overwrite_the_cosine_score():
    """MIN_RETRIEVAL_SCORE filters on `score`; the two live on different scales."""
    results = [_result("a", score=0.77)]
    PineconeReranker(session=_FakeSession(_ranked((0, 0.02)))).rerank("q", results)
    assert results[0].score == 0.77
    assert results[0].rerank_score == 0.02


def test_rerank_passes_top_n_through():
    session = _FakeSession(_ranked((0, 0.9)))
    PineconeReranker(session=session).rerank("q", [_result("a"), _result("b")], top_n=1)
    assert session.posts[0]["json"]["top_n"] == 1


def test_rerank_of_nothing_calls_no_api():
    session = _FakeSession()
    assert PineconeReranker(session=session).rerank("q", []) == []
    assert session.posts == []


def test_rerank_refuses_more_documents_than_the_model_accepts():
    """A config error (TOP_K too high), raised rather than silently truncated."""
    reranker = PineconeReranker(session=_FakeSession())
    with pytest.raises(ValueError, match="at most 100 documents"):
        reranker.rerank("q", [_result(str(i)) for i in range(101)])


def test_http_error_carries_its_status():
    session = _FakeSession(_Resp(status_code=500, text="boom"))
    with pytest.raises(RerankHTTPError) as caught:
        PineconeReranker(session=session, max_attempts=1).rerank("q", [_result("a")])
    assert caught.value.status_code == 500


@pytest.mark.parametrize(
    "status,body,exhausted",
    [
        (429, "rate limit exceeded, slow down", False),
        (429, "You have exceeded your monthly quota", True),
        (402, "payment required", True),
        (403, "plan limit reached", True),
        (500, "internal error", False),
    ],
)
def test_quota_exhaustion_is_told_apart_from_pacing(status, body, exhausted):
    """Only an exhausted allowance should stop the caller retrying."""
    reranker = PineconeReranker(session=_FakeSession())
    assert reranker.is_quota_exhausted(RerankHTTPError(status, body)) is exhausted


def test_a_non_http_error_is_never_read_as_quota():
    assert PineconeReranker(session=_FakeSession()).is_quota_exhausted(TimeoutError()) is False


def test_a_transient_429_is_retried():
    slept: list[float] = []
    session = _FakeSession(_Resp(status_code=429, text="slow down"), _ranked((0, 0.5)))

    ranked = PineconeReranker(
        session=session, max_attempts=2, sleep=slept.append
    ).rerank("q", [_result("a")])

    assert len(session.posts) == 2
    assert slept  # backed off rather than hammering
    assert ranked[0].rerank_score == 0.5


def test_an_exhausted_quota_is_not_retried():
    """Retrying spends the budget on calls that cannot succeed this month."""
    session = _FakeSession(_Resp(status_code=429, text="monthly quota exceeded"))
    with pytest.raises(RerankHTTPError):
        PineconeReranker(session=session, max_attempts=5).rerank("q", [_result("a")])
    assert len(session.posts) == 1


# --- fail open -------------------------------------------------------------


class _Inner:
    """A reranker that succeeds, or raises a chosen error."""

    default_model = "fake-rerank"
    api_key_env = "FAKE_KEY"
    max_documents = 100

    def __init__(self, error=None, quota=False):
        self.model = self.default_model
        self.error = error
        self.quota = quota
        self.calls = 0

    def is_quota_exhausted(self, error):
        return self.quota

    def rerank(self, query, results, top_n=None):
        self.calls += 1
        if self.error:
            raise self.error
        for i, result in enumerate(results):
            result.rerank_score = 1.0 - i / 100
        return list(results)


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_fail_open_passes_a_working_reranker_through():
    wrapped = FailOpenReranker(_Inner())
    ranked = wrapped.rerank("q", [_result("a"), _result("b")])
    assert [r.rerank_score for r in ranked] == [1.0, 0.99]
    assert wrapped.degraded is False


def test_a_failure_returns_the_input_unchanged_and_in_order():
    """Degraded mode IS today's production behaviour: cosine order, nothing cut."""
    wrapped = FailOpenReranker(_Inner(error=TimeoutError("upstream")))
    results = [_result("a"), _result("b"), _result("c")]

    out = wrapped.rerank("q", results)

    assert [r.document_id for r in out] == [r.document_id for r in results]
    assert all(r.rerank_score is None for r in out)


def test_a_transient_failure_does_not_latch():
    inner = _Inner(error=TimeoutError("blip"), quota=False)
    wrapped = FailOpenReranker(inner)

    wrapped.rerank("q", [_result("a")])
    wrapped.rerank("q", [_result("a")])

    assert wrapped.degraded is False
    assert inner.calls == 2  # tried again, as it should


def test_an_exhausted_quota_latches_and_stops_calling():
    inner = _Inner(error=RuntimeError("out of quota"), quota=True)
    wrapped = FailOpenReranker(inner)

    wrapped.rerank("q", [_result("a")])
    assert wrapped.degraded is True

    wrapped.rerank("q", [_result("a")])
    assert inner.calls == 1  # the second request never touched the network


def test_the_latch_re_arms_after_the_retry_window():
    clock = _Clock()
    inner = _Inner(error=RuntimeError("out of quota"), quota=True)
    wrapped = FailOpenReranker(inner, retry_after_seconds=3600, clock=clock)

    wrapped.rerank("q", [_result("a")])
    clock.now += 3599
    wrapped.rerank("q", [_result("a")])
    assert inner.calls == 1  # still inside the window

    clock.now += 2
    wrapped.rerank("q", [_result("a")])
    assert inner.calls == 2  # half-open: one probe allowed through


def test_retry_after_zero_latches_until_the_process_restarts():
    clock = _Clock()
    inner = _Inner(error=RuntimeError("out of quota"), quota=True)
    wrapped = FailOpenReranker(inner, retry_after_seconds=0, clock=clock)

    wrapped.rerank("q", [_result("a")])
    clock.now += 10_000_000
    wrapped.rerank("q", [_result("a")])
    assert inner.calls == 1


def test_rerank_of_nothing_short_circuits_before_the_latch():
    inner = _Inner()
    assert FailOpenReranker(inner).rerank("q", []) == []
    assert inner.calls == 0


def test_the_quota_warning_is_logged_once_under_concurrency(caplog):
    """80 threads hitting the same dead quota must not write 80 warnings.

    The line only helps if it is findable, which is what the lock is for.
    """
    inner = _Inner(error=RuntimeError("out of quota"), quota=True)
    wrapped = FailOpenReranker(inner)

    with caplog.at_level(logging.WARNING, logger="kmp.rerank"):
        threads = [
            threading.Thread(target=wrapped.rerank, args=("q", [_result("a")]))
            for _ in range(16)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    disabled = [r for r in caplog.records if r.getMessage() == "reranker disabled"]
    assert len(disabled) == 1
    assert disabled[0].context["quota_exhausted"] is True


def test_repeated_failures_latch_even_when_unrecognised():
    """The safety net for a failure mode nobody has seen yet.

    Nobody knows what Pinecone returns when the monthly rerank allowance runs
    out. If `is_quota_exhausted` misses it, every request would pay a doomed
    call plus its backoff for the rest of the month — unacceptable now that
    reranking is on by default. Classification decides how fast we give up;
    this decides that we give up at all.
    """
    inner = _Inner(error=RuntimeError("some unfamiliar 4xx"), quota=False)
    wrapped = FailOpenReranker(inner, consecutive_failure_limit=3)

    for _ in range(3):
        wrapped.rerank("q", [_result("a")])
    assert wrapped.degraded is True
    assert inner.calls == 3

    wrapped.rerank("q", [_result("a")])
    assert inner.calls == 3  # latched; no further calls


def test_a_success_resets_the_failure_run():
    """Only CONSECUTIVE failures count, or a flaky month eventually latches."""
    inner = _Inner(quota=False)
    wrapped = FailOpenReranker(inner, consecutive_failure_limit=3)

    inner.error = RuntimeError("blip")
    wrapped.rerank("q", [_result("a")])
    wrapped.rerank("q", [_result("a")])

    inner.error = None
    wrapped.rerank("q", [_result("a")])  # recovers

    inner.error = RuntimeError("blip")
    wrapped.rerank("q", [_result("a")])
    wrapped.rerank("q", [_result("a")])
    assert wrapped.degraded is False  # run restarted, only 2 since the success


def test_the_log_says_which_rule_latched_it(caplog):
    """"failure_limit" means _QUOTA_MARKERS missed a real exhaustion and needs
    widening — this field is the only way that would ever be noticed."""
    wrapped = FailOpenReranker(
        _Inner(error=RuntimeError("mystery"), quota=False), consecutive_failure_limit=1
    )
    with caplog.at_level(logging.WARNING, logger="kmp.rerank"):
        wrapped.rerank("q", [_result("a")])

    record = next(r for r in caplog.records if r.getMessage() == "reranker disabled")
    assert record.context["latched_on"] == "failure_limit"
    assert record.context["quota_exhausted"] is False


def test_a_quota_error_latches_immediately_not_after_the_limit():
    """A recognised exhaustion should not cost four more doomed calls first."""
    inner = _Inner(error=RuntimeError("out of quota"), quota=True)
    wrapped = FailOpenReranker(inner, consecutive_failure_limit=5)

    wrapped.rerank("q", [_result("a")])
    assert wrapped.degraded is True
    assert inner.calls == 1

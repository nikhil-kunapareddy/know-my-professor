"""Unit tests for the frontend transport and formatting.

Neither module under test imports Streamlit — the transport is plain requests and
the formatting is plain strings — so these run anywhere the package installs.
"""

from __future__ import annotations

import pytest
import requests

from serving.frontend.api_client import ChatClient, ChatError
from serving.frontend.formatting import format_citation, sources_label
from shared.schemas import Citation


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, raises_json=False):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self._raises_json = raises_json

    def json(self):
        if self._raises_json:
            raise ValueError("not json")
        return self._payload


class _FakeSession:
    """Returns queued responses (or raises queued exceptions) in order."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        outcome = self.outcomes.pop(0) if self.outcomes else self.outcomes
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(*outcomes, retries=1) -> ChatClient:
    client = ChatClient(base_url="https://api.example/", session=_FakeSession(*outcomes), retries=retries)
    client.sleep = lambda _seconds: None  # no real waiting in tests
    return client


# --- happy path ------------------------------------------------------------


def test_ask_parses_a_valid_response():
    payload = {
        "answer": "Ann works on PL [1].",
        "citations": [{"number": 1, "professor_name": "Ann", "professor_title": "Prof",
                       "section_type": "biography", "url": "https://x/", "score": 0.9}],
    }
    client = _client(_FakeResponse(payload=payload))
    result = client.ask("who does PL?")

    assert result.answer == "Ann works on PL [1]."
    assert result.citations[0].professor_name == "Ann"


def test_ask_targets_the_versioned_endpoint():
    client = _client(_FakeResponse(payload={"answer": "hi"}))
    client.ask("q?")
    assert client.session.calls[0]["url"] == "https://api.example/v1/chat"


def test_base_url_trailing_slash_does_not_double_up():
    assert ChatClient(base_url="https://api.example///").chat_url == "https://api.example/v1/chat"


def test_a_timeout_is_always_sent():
    client = _client(_FakeResponse(payload={"answer": "hi"}))
    client.ask("q?")
    assert client.session.calls[0]["timeout"] == 60


# --- failures --------------------------------------------------------------


def test_timeout_becomes_a_readable_error():
    client = _client(requests.Timeout(), requests.Timeout(), retries=1)
    with pytest.raises(ChatError) as excinfo:
        client.ask("q?")
    assert "too long" in excinfo.value.message
    assert len(client.session.calls) == 2  # retried once


def test_connection_error_becomes_a_readable_error():
    client = _client(requests.ConnectionError(), requests.ConnectionError())
    with pytest.raises(ChatError, match="Couldn't reach the API"):
        client.ask("q?")


def test_api_detail_is_surfaced_to_the_user():
    """The API already writes user-safe detail text; don't discard it."""
    body = {"error": "rate_limited", "detail": "Upstream is rate limiting. Try again shortly.",
            "request_id": "abc123"}
    client = _client(
        _FakeResponse(429, body, {"X-Request-ID": "abc123"}),
        _FakeResponse(429, body, {"X-Request-ID": "abc123"}),
    )
    with pytest.raises(ChatError) as excinfo:
        client.ask("q?")

    assert excinfo.value.message == "Upstream is rate limiting. Try again shortly."
    assert excinfo.value.request_id == "abc123"


def test_retryable_status_is_retried_then_succeeds():
    ok = {"answer": "recovered", "citations": []}
    client = _client(_FakeResponse(503, {}), _FakeResponse(200, ok))
    assert client.ask("q?").answer == "recovered"
    assert len(client.session.calls) == 2


def test_client_error_is_not_retried():
    client = _client(_FakeResponse(400, {"detail": "bad question"}))
    with pytest.raises(ChatError, match="bad question"):
        client.ask("q?")
    assert len(client.session.calls) == 1  # 4xx is the caller's fault; no retry


def test_unparseable_body_is_reported_not_crashed():
    client = _client(_FakeResponse(200, raises_json=True))
    with pytest.raises(ChatError, match="doesn't understand"):
        client.ask("q?")


def test_response_missing_required_fields_is_reported():
    """A field rename on the API must fail here, not render as a blank."""
    client = _client(_FakeResponse(200, {"reply": "wrong field name"}))
    with pytest.raises(ChatError, match="doesn't understand"):
        client.ask("q?")


# --- formatting (pure; no Streamlit needed) --------------------------------


def test_citation_without_a_url_renders_as_text_not_a_dead_link():
    line = format_citation(Citation(number=1, professor_name="Ann", section_type="biography"))
    assert "no link" in line
    assert "]()" not in line


def test_citation_with_a_url_links_to_the_source():
    line = format_citation(
        Citation(number=2, professor_name="Bob", professor_title="Prof",
                 section_type="website_summary", url="https://bob.example/", score=0.83)
    )
    assert "[source](https://bob.example/)" in line
    assert "website summary" in line
    assert "0.83" in line


def test_citation_tolerates_a_missing_name():
    assert "Unknown" in format_citation(Citation(number=1))


def test_sources_label_is_singular_for_one():
    assert sources_label(1) == "Source (1)"
    assert sources_label(3) == "Sources (3)"

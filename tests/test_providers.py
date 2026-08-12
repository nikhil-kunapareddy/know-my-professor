"""Unit tests for the provider seams: registries, backoff, and the embedder."""

from __future__ import annotations

import pytest

from core.llm import available_generators, build_generator
from shared.config import DEFAULT_CHAT_MODEL, EMBED_DIM, EMBED_MODEL
from shared.embeddings import available_embedders, build_embedder
from shared.embeddings.mistral import MistralEmbedder
from shared.retry import with_backoff

# --- embedding registry ----------------------------------------------------


def test_build_embedder_uses_the_configured_model_and_dim():
    embedder = build_embedder()
    assert embedder.model == EMBED_MODEL
    assert embedder.dim == EMBED_DIM
    assert "mistral" in available_embedders()


def test_build_embedder_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown embedding provider"):
        build_embedder("does-not-exist")


def test_build_embedder_rejects_a_dim_that_the_index_cannot_hold():
    """A provider with a different vector width needs a new index, not a crash mid-ingest."""
    with pytest.raises(ValueError, match="requires a new"):
        build_embedder("mistral", dim=512)


# --- generation registry ---------------------------------------------------


def test_build_generator_uses_the_configured_model():
    generator = build_generator(client=object())
    assert generator.model == DEFAULT_CHAT_MODEL
    assert "llama" in available_generators()


def test_build_generator_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown chat provider"):
        build_generator("does-not-exist")


def test_providers_declare_their_credential_env_var():
    """Lets the API fail at boot on a missing key instead of on first request."""
    assert build_embedder().api_key_env == "MISTRAL_API_KEY"
    assert build_generator(client=object()).api_key_env == "LLAMA_API_KEY"


# --- backoff ---------------------------------------------------------------


class _Boom(Exception):
    def __init__(self, status_code=None):
        super().__init__("boom")
        self.status_code = status_code


def test_with_backoff_retries_then_succeeds():
    slept: list[float] = []
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _Boom(429)
        return "ok"

    result = with_backoff(
        flaky,
        is_retryable=lambda e: getattr(e, "status_code", None) == 429,
        max_attempts=5,
        sleep=slept.append,
    )
    assert result == "ok"
    assert attempts["n"] == 3
    assert slept == [2.0, 4.0]  # doubling


def test_with_backoff_reraises_non_retryable_immediately():
    slept: list[float] = []
    attempts = {"n": 0}

    def always_401():
        attempts["n"] += 1
        raise _Boom(401)

    with pytest.raises(_Boom):
        with_backoff(
            always_401,
            is_retryable=lambda e: getattr(e, "status_code", None) == 429,
            max_attempts=5,
            sleep=slept.append,
        )
    assert attempts["n"] == 1  # no retries for a permanent error
    assert slept == []


def test_with_backoff_gives_up_and_raises_after_max_attempts():
    slept: list[float] = []

    with pytest.raises(_Boom):
        with_backoff(
            lambda: (_ for _ in ()).throw(_Boom(429)),
            is_retryable=lambda e: True,
            max_attempts=3,
            sleep=slept.append,
        )
    assert len(slept) == 2  # slept between the 3 attempts, then re-raised


def test_with_backoff_caps_the_delay():
    slept: list[float] = []
    with pytest.raises(_Boom):
        with_backoff(
            lambda: (_ for _ in ()).throw(_Boom(429)),
            is_retryable=lambda e: True,
            max_attempts=8,
            initial_delay=10.0,
            max_delay=25.0,
            sleep=slept.append,
        )
    assert max(slept) == 25.0


# --- the Mistral embedder --------------------------------------------------


class _FakeEmbeddingsAPI:
    def __init__(self, dim=4):
        self.dim = dim
        self.calls: list[list[str]] = []

    def create(self, model, inputs):
        self.calls.append(list(inputs))
        vectors = [type("D", (), {"embedding": [0.1] * self.dim})() for _ in inputs]
        return type("R", (), {"data": vectors})()


class _FakeMistral:
    def __init__(self, dim=4):
        self.embeddings = _FakeEmbeddingsAPI(dim)


def test_embed_texts_batches_by_batch_size():
    client = _FakeMistral()
    embedder = MistralEmbedder(client=client, batch_size=2)

    vectors = embedder.embed_texts(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert client.embeddings.calls == [["a", "b"], ["c", "d"], ["e"]]


def test_embed_query_reuses_the_document_path():
    """One code path for both sides is what keeps query and document vectors aligned."""
    client = _FakeMistral()
    embedder = MistralEmbedder(client=client)

    vector = embedder.embed_query("who works on PL?")
    assert vector == [0.1] * 4
    assert client.embeddings.calls == [["who works on PL?"]]


def test_serving_path_does_not_pace(monkeypatch):
    """A paced embedder would add a full second to every /chat request."""
    slept: list[float] = []
    monkeypatch.setattr("shared.embeddings.mistral.time.sleep", slept.append)

    MistralEmbedder(client=_FakeMistral()).embed_query("q")
    assert slept == []

    MistralEmbedder(client=_FakeMistral(), pace_seconds=1.0).embed_texts(["q"])
    assert slept == [1.0]

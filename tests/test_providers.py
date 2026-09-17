"""Unit tests for the provider seams: registries, backoff, and the embedder."""

from __future__ import annotations

import pytest

from core.llm import available_generators, build_generator
from core.llm.anthropic import AnthropicGenerator
from core.llm.llama import LlamaGenerator
from shared.config import EMBED_DIM, EMBED_MODEL
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


def test_build_generator_defaults_to_anthropic():
    generator = build_generator(client=object())
    assert isinstance(generator, AnthropicGenerator)
    assert generator.model == AnthropicGenerator.default_model
    assert {"anthropic", "llama"} <= set(available_generators())


def test_each_provider_supplies_its_own_default_model():
    """A model id is provider-specific, so it cannot live in shared.config."""
    assert build_generator("llama", client=object()).model == LlamaGenerator.default_model
    assert build_generator("anthropic", client=object()).model != LlamaGenerator.default_model


def test_an_explicit_model_overrides_the_provider_default():
    assert build_generator("anthropic", client=object(), model="claude-sonnet-5").model == (
        "claude-sonnet-5"
    )


def test_build_generator_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown chat provider"):
        build_generator("does-not-exist")


def test_providers_declare_their_credential_env_var():
    """Lets the API fail at boot on a missing key instead of on first request."""
    assert build_embedder().api_key_env == "MISTRAL_API_KEY"
    assert build_generator(client=object()).api_key_env == "ANTHROPIC_API_KEY"
    assert build_generator("llama", client=object()).api_key_env == "LLAMA_API_KEY"


# --- the Anthropic generator -----------------------------------------------


class _FakeBlock:
    def __init__(self, type_, text=""):
        self.type = type_
        self.text = text


class _FakeMessages:
    def __init__(self, blocks, stop_reason="end_turn"):
        self._blocks, self._stop_reason = blocks, stop_reason
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return type("R", (), {"content": self._blocks, "stop_reason": self._stop_reason})()


class _FakeAnthropic:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.messages = _FakeMessages(blocks, stop_reason)


def test_anthropic_generate_joins_text_and_drops_thinking_blocks():
    """Thinking is on by default on this model, so content is not all answer."""
    client = _FakeAnthropic([
        _FakeBlock("thinking", "the user wants PL people"),
        _FakeBlock("text", "Jan Vitek works on PL [1]."),
        _FakeBlock("text", " Also Amal Ahmed [2]."),
    ])

    answer = AnthropicGenerator(client=client).generate("sys", "who does PL?")
    assert answer == "Jan Vitek works on PL [1]. Also Amal Ahmed [2]."


def test_anthropic_generate_treats_a_refusal_as_no_answer():
    """A refusal is a 200 with stop_reason, not an exception; "" hits NO_ANSWER."""
    client = _FakeAnthropic([], stop_reason="refusal")
    assert AnthropicGenerator(client=client).generate("sys", "q") == ""


def test_anthropic_generate_never_sends_sampling_params():
    """temperature/top_p are removed on the Claude 5 family and return a 400."""
    client = _FakeAnthropic([_FakeBlock("text", "hi")])
    AnthropicGenerator(client=client).generate("sys", "q")

    sent = client.messages.kwargs
    assert "temperature" not in sent and "top_p" not in sent and "top_k" not in sent
    assert sent["system"] == "sys"
    assert sent["messages"] == [{"role": "user", "content": "q"}]
    assert sent["output_config"] == {"effort": "low"}


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


# --- effort is Claude-5-only ----------------------------------------------


def test_effort_is_sent_only_to_models_that_accept_it():
    """Anthropic 400s on effort for anything outside the Claude 5 family."""
    from core.llm.anthropic import supports_effort

    assert supports_effort("claude-opus-5")
    assert supports_effort("claude-sonnet-5")
    assert supports_effort("claude-fable-5-1")
    assert not supports_effort("claude-haiku-4-5-20251001")
    assert not supports_effort("claude-3-5-sonnet-20241022")


def test_an_older_model_is_called_without_output_config():
    client = _FakeAnthropic([_FakeBlock("text", "hi")])
    generator = AnthropicGenerator(client=client, model="claude-haiku-4-5-20251001")
    generator.generate("sys", "q")

    assert "output_config" not in client.messages.kwargs, (
        "sending effort to a non-Claude-5 model is a hard 400"
    )
    assert generator.effort is None


def test_effort_can_be_switched_off_explicitly():
    client = _FakeAnthropic([_FakeBlock("text", "hi")])
    AnthropicGenerator(client=client, effort=None).generate("sys", "q")
    assert "output_config" not in client.messages.kwargs

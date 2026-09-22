"""Unit tests for env-var settings and the Pinecone dimension guard."""

from __future__ import annotations

import pytest

from preprocessing.ingest.pinecone_store import PineconeStore
from shared.config import DEFAULT_TOP_K, EMBED_DIM, PINECONE_DEFAULT_INDEX
from shared.settings import (
    ApiSettings,
    IngestSettings,
    MissingSettingError,
    require_env,
)

# --- require_env -----------------------------------------------------------


def test_require_env_returns_the_value(monkeypatch):
    monkeypatch.setenv("KMP_TEST_KEY", "value")
    assert require_env("KMP_TEST_KEY") == "value"


def test_require_env_rejects_missing_and_empty(monkeypatch):
    monkeypatch.delenv("KMP_TEST_KEY", raising=False)
    with pytest.raises(MissingSettingError, match="KMP_TEST_KEY"):
        require_env("KMP_TEST_KEY")

    monkeypatch.setenv("KMP_TEST_KEY", "")
    with pytest.raises(MissingSettingError):
        require_env("KMP_TEST_KEY")


# --- ApiSettings -----------------------------------------------------------


def test_api_settings_fall_back_to_config_defaults(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    for name in ("EMBED_PROVIDER", "CHAT_PROVIDER", "CHAT_MODEL",
                 "PINECONE_INDEX_NAME", "TOP_K", "MIN_RETRIEVAL_SCORE"):
        monkeypatch.delenv(name, raising=False)

    settings = ApiSettings.from_env()
    assert settings.embed_provider == "mistral"
    assert settings.chat_provider == "anthropic"
    # None, not a model id: the provider supplies its own default.
    assert settings.chat_model is None
    assert settings.index_name == PINECONE_DEFAULT_INDEX
    assert settings.top_k == DEFAULT_TOP_K


def test_api_settings_read_overrides(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    monkeypatch.setenv("TOP_K", "3")
    monkeypatch.setenv("MIN_RETRIEVAL_SCORE", "0.5")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "other-index")

    settings = ApiSettings.from_env()
    assert (settings.top_k, settings.min_score, settings.index_name) == (3, 0.5, "other-index")


def test_api_settings_fail_fast_without_a_pinecone_key(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(MissingSettingError, match="PINECONE_API_KEY"):
        ApiSettings.from_env()


def test_numeric_settings_reject_garbage(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    monkeypatch.setenv("TOP_K", "eight")
    with pytest.raises(MissingSettingError, match="must be an integer"):
        ApiSettings.from_env()


# --- IngestSettings --------------------------------------------------------


def test_ingest_settings_prefer_cli_over_env(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    monkeypatch.setenv("KMP_GCS_BUCKET", "env-bucket")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "env-index")

    settings = IngestSettings.from_env(bucket="cli-bucket", index_name="cli-index")
    assert (settings.bucket, settings.index_name) == ("cli-bucket", "cli-index")

    fallback = IngestSettings.from_env()
    assert (fallback.bucket, fallback.index_name) == ("env-bucket", "env-index")


# --- Pinecone dimension guard ---------------------------------------------


class _Described:
    def __init__(self, dimension):
        self.dimension = dimension


def test_assert_dimension_accepts_a_matching_index():
    PineconeStore.assert_dimension(_Described(EMBED_DIM), "idx", EMBED_DIM)


def test_assert_dimension_rejects_a_mismatched_index():
    """The old 3072-dim index must not silently accept 1024-dim vectors."""
    with pytest.raises(ValueError, match="immutable"):
        PineconeStore.assert_dimension(_Described(3072), "know-my-professor", 1024)


def test_assert_dimension_reads_dict_descriptions():
    with pytest.raises(ValueError):
        PineconeStore.assert_dimension({"dimension": 512}, "idx", 1024)


def test_assert_dimension_tolerates_an_unreadable_description():
    """An SDK shape change should not block an otherwise valid ingest."""
    PineconeStore.assert_dimension(object(), "idx", 1024)


# --- rerank settings -------------------------------------------------------


def test_rerank_is_off_unless_a_provider_is_named(monkeypatch):
    """Absence must be the default: CI never sets env vars, so an existing
    revision has none of these and has to keep booting and behaving as before."""
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    for name in ("RERANK_PROVIDER", "RERANK_MODEL", "RERANK_MIN_SCORE",
                 "RERANK_TOP_N", "RERANK_RETRY_AFTER_SECONDS"):
        monkeypatch.delenv(name, raising=False)

    settings = ApiSettings.from_env()
    assert settings.rerank_provider is None
    assert settings.rerank_model is None
    assert settings.rerank_top_n is None
    # 0.0, not 0.5: a cross-encoder score is a different scale from cosine and
    # this corpus's distribution has not been measured. Picking a number first
    # is how MIN_RETRIEVAL_SCORE ended up inert.
    assert settings.rerank_min_score == 0.0


def test_rerank_settings_read_overrides(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    monkeypatch.setenv("RERANK_PROVIDER", "pinecone")
    monkeypatch.setenv("RERANK_MODEL", "pinecone-rerank-v0")
    monkeypatch.setenv("RERANK_MIN_SCORE", "0.5")
    monkeypatch.setenv("RERANK_TOP_N", "8")
    monkeypatch.setenv("RERANK_RETRY_AFTER_SECONDS", "60")

    settings = ApiSettings.from_env()
    assert settings.rerank_provider == "pinecone"
    assert settings.rerank_model == "pinecone-rerank-v0"
    assert (settings.rerank_min_score, settings.rerank_top_n) == (0.5, 8)
    assert settings.rerank_retry_after_seconds == 60.0


def test_rerank_top_n_distinguishes_unset_from_zero(monkeypatch):
    """None keeps every chunk; 0 would keep none, so they cannot share a value."""
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    monkeypatch.setenv("RERANK_TOP_N", "0")
    assert ApiSettings.from_env().rerank_top_n == 0

    monkeypatch.setenv("RERANK_TOP_N", "")
    assert ApiSettings.from_env().rerank_top_n is None


def test_rerank_numeric_settings_reject_garbage(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pk")
    monkeypatch.setenv("RERANK_TOP_N", "eleven")
    with pytest.raises(MissingSettingError, match="must be an integer"):
        ApiSettings.from_env()

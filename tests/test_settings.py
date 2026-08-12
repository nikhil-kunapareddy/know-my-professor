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
    for name in ("EMBED_PROVIDER", "CHAT_PROVIDER", "LLAMA_CHAT_MODEL",
                 "PINECONE_INDEX_NAME", "TOP_K", "MIN_RETRIEVAL_SCORE"):
        monkeypatch.delenv(name, raising=False)

    settings = ApiSettings.from_env()
    assert settings.embed_provider == "mistral"
    assert settings.chat_provider == "llama"
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

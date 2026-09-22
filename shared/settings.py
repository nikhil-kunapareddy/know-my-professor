"""Env-var wiring, read and validated in one place per component.

Environment reading used to be scattered across ``app.py`` lifespan, the ingest
runner's ``main``, and each provider's constructor, so a missing or mistyped var
surfaced at different times in different components — sometimes mid-run, after
work had already been paid for.

Plain dataclasses rather than pydantic-settings on purpose: ``shared`` is
imported by all five deployables, and the three batch Jobs deliberately do not
install pydantic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from shared.config import (
    CHAT_NAMESPACES,
    DEFAULT_CHAT_PROVIDER,
    DEFAULT_EMBED_PROVIDER,
    DEFAULT_TOP_K,
    MIN_RETRIEVAL_SCORE,
    PINECONE_DEFAULT_CLOUD,
    PINECONE_DEFAULT_INDEX,
    PINECONE_DEFAULT_REGION,
)


class MissingSettingError(RuntimeError):
    """A required environment variable is unset or empty."""


def require_env(name: str) -> str:
    """Return ``name`` from the environment, or fail with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise MissingSettingError(f"missing required env var: {name}")
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise MissingSettingError(f"env var {name} must be an integer, got {raw!r}") from None


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise MissingSettingError(f"env var {name} must be a number, got {raw!r}") from None


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Comma-separated env var -> tuple, blanks dropped. Empty means default.

    Order is preserved: it is the order the prompt presents the corpora in.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class ApiSettings:
    """Everything the /chat service reads from the environment."""

    embed_provider: str = DEFAULT_EMBED_PROVIDER
    chat_provider: str = DEFAULT_CHAT_PROVIDER
    #: None means "let the chosen provider pick its own default_model".
    chat_model: str | None = None
    index_name: str = PINECONE_DEFAULT_INDEX
    #: Index partitions /chat searches, one query each, each with its own top_k.
    #: Every name must match a partition ingest actually wrote; a mismatch is
    #: silent (empty results from that partition, not an error).
    namespaces: tuple[str, ...] = CHAT_NAMESPACES
    top_k: int = DEFAULT_TOP_K
    min_score: float = MIN_RETRIEVAL_SCORE
    pinecone_api_key: str = ""

    @classmethod
    def from_env(cls) -> ApiSettings:
        return cls(
            embed_provider=os.environ.get("EMBED_PROVIDER") or DEFAULT_EMBED_PROVIDER,
            chat_provider=os.environ.get("CHAT_PROVIDER") or DEFAULT_CHAT_PROVIDER,
            chat_model=os.environ.get("CHAT_MODEL") or None,
            index_name=os.environ.get("PINECONE_INDEX_NAME") or PINECONE_DEFAULT_INDEX,
            namespaces=_csv_env("PINECONE_NAMESPACES", CHAT_NAMESPACES),
            top_k=_int_env("TOP_K", DEFAULT_TOP_K),
            min_score=_float_env("MIN_RETRIEVAL_SCORE", MIN_RETRIEVAL_SCORE),
            pinecone_api_key=require_env("PINECONE_API_KEY"),
        )


@dataclass(frozen=True)
class IngestSettings:
    """Everything the ingest Job reads from the environment."""

    bucket: str = ""
    index_name: str = PINECONE_DEFAULT_INDEX
    cloud: str = PINECONE_DEFAULT_CLOUD
    region: str = PINECONE_DEFAULT_REGION
    embed_provider: str = DEFAULT_EMBED_PROVIDER
    pinecone_api_key: str = ""

    @classmethod
    def from_env(cls, bucket: str | None = None, index_name: str | None = None) -> IngestSettings:
        """CLI flags win over env vars, which win over the defaults."""
        return cls(
            bucket=bucket or require_env("KMP_GCS_BUCKET"),
            index_name=index_name or os.environ.get("PINECONE_INDEX_NAME") or PINECONE_DEFAULT_INDEX,
            cloud=os.environ.get("PINECONE_CLOUD") or PINECONE_DEFAULT_CLOUD,
            region=os.environ.get("PINECONE_REGION") or PINECONE_DEFAULT_REGION,
            embed_provider=os.environ.get("EMBED_PROVIDER") or DEFAULT_EMBED_PROVIDER,
            pinecone_api_key=require_env("PINECONE_API_KEY"),
        )

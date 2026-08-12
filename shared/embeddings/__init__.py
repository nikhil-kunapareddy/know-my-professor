"""Embedding providers.

Adding one means writing an ``Embedder`` and adding a line to ``_EMBEDDERS``.
Nothing else in the codebase names a provider.
"""

from __future__ import annotations

import importlib

from shared.config import DEFAULT_EMBED_PROVIDER, EMBED_DIM

from .base import Embedder

#: provider name -> "module:ClassName". Resolved lazily so importing this
#: package never drags in an SDK the running component didn't install.
_EMBEDDERS: dict[str, str] = {
    "mistral": "shared.embeddings.mistral:MistralEmbedder",
}


def available_embedders() -> tuple[str, ...]:
    """Names accepted by ``build_embedder``."""
    return tuple(sorted(_EMBEDDERS))


def build_embedder(provider: str | None = None, **kwargs) -> Embedder:
    """Construct the named embedding provider (default: shared.config).

    Guards the invariant that outlives any single run: the vector width must
    match the Pinecone index, whose dimension is fixed when it is created.
    Switching to a provider with a different width is legitimate but means a new
    index and a full re-ingest, so it fails here rather than part-way through an
    upsert with the embedding bill already paid.
    """
    name = provider or DEFAULT_EMBED_PROVIDER
    try:
        target = _EMBEDDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown embedding provider {name!r}; available: {available_embedders()}"
        ) from None

    module_path, _, class_name = target.partition(":")
    embedder = getattr(importlib.import_module(module_path), class_name)(**kwargs)

    if embedder.dim != EMBED_DIM:
        raise ValueError(
            f"embedder {name!r} produces {embedder.dim}-dim vectors but the configured "
            f"index expects {EMBED_DIM}. Changing embedding width requires a new "
            f"Pinecone index and a full re-ingest — update EMBED_DIM and "
            f"PINECONE_DEFAULT_INDEX together."
        )
    return embedder


__all__ = ["Embedder", "available_embedders", "build_embedder"]

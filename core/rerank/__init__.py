"""Rerank providers.

Adding one means writing a ``Reranker`` and adding a line to ``_RERANKERS``.
``serving/api`` builds whatever the settings name and never imports a provider
module directly.

Unlike embedding and generation, reranking is **optional**: there is no default
provider and no null implementation. ``build_reranker`` always returns a real
``Reranker`` or raises, exactly like ``build_generator`` -- "off" is expressed
by the caller not calling this at all, which keeps every build_* function in the
codebase honest about its return type.
"""

from __future__ import annotations

import importlib

from .base import Reranker
from .failopen import FailOpenReranker

#: provider name -> "module:ClassName". Resolved lazily so importing this
#: package never drags in an SDK the running component didn't install.
_RERANKERS: dict[str, str] = {
    "pinecone": "core.rerank.pinecone:PineconeReranker",
}


def available_rerankers() -> tuple[str, ...]:
    """Names accepted by ``build_reranker``."""
    return tuple(sorted(_RERANKERS))


def build_reranker(provider: str, **kwargs) -> Reranker:
    """Construct the named rerank provider.

    ``provider`` is required and has no default, because there is no rerank
    provider the system falls back to -- reranking is off unless a deployment
    asks for it by name.
    """
    try:
        target = _RERANKERS[provider]
    except KeyError:
        raise ValueError(
            f"unknown rerank provider {provider!r}; available: {available_rerankers()}"
        ) from None

    module_path, _, class_name = target.partition(":")
    return getattr(importlib.import_module(module_path), class_name)(**kwargs)


__all__ = ["FailOpenReranker", "Reranker", "available_rerankers", "build_reranker"]

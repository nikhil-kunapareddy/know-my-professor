"""Chat/generation providers.

Adding one means writing a ``Generator`` and adding a line to ``_GENERATORS``.
``serving/api`` builds whatever the settings name and never imports a provider
module directly.
"""

from __future__ import annotations

import importlib

from shared.config import DEFAULT_CHAT_PROVIDER

from .base import Generator

#: provider name -> "module:ClassName". Resolved lazily so importing this
#: package never drags in an SDK the running component didn't install.
_GENERATORS: dict[str, str] = {
    "llama": "core.llm.llama:LlamaGenerator",
}


def available_generators() -> tuple[str, ...]:
    """Names accepted by ``build_generator``."""
    return tuple(sorted(_GENERATORS))


def build_generator(provider: str | None = None, **kwargs) -> Generator:
    """Construct the named chat provider (default: shared.config)."""
    name = provider or DEFAULT_CHAT_PROVIDER
    try:
        target = _GENERATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown chat provider {name!r}; available: {available_generators()}"
        ) from None

    module_path, _, class_name = target.partition(":")
    return getattr(importlib.import_module(module_path), class_name)(**kwargs)


__all__ = ["Generator", "available_generators", "build_generator"]

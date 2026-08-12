"""The generation interface the pipeline depends on.

Mirrors ``core.retrieval.base.Retriever``: the pipeline is written against this,
so swapping chat providers never touches orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Generator(ABC):
    """Produces an answer from a system instruction plus a user message."""

    #: Provider-facing model identifier.
    model: str
    #: Env var holding this provider's credential, so a component can fail at
    #: startup on a missing key instead of on the first request.
    api_key_env: str | None = None

    @abstractmethod
    def generate(self, system_instruction: str, user_message: str) -> str:
        """Return the model's answer text, or "" if it produced none."""
        ...

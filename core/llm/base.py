"""The generation interface the pipeline depends on.

Mirrors ``core.retrieval.base.Retriever``: the pipeline is written against this,
so swapping chat providers never touches orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Generation:
    """One model response, plus the accounting needed to compare providers.

    ``generate`` used to return a bare ``str``, which discarded ``usage`` and
    ``stop_reason`` even though the SDK hands both back on every call. Token
    counts and refusals were therefore unmeasurable.

    **Returned, never stored on the generator.** ``serving/api/app.py`` builds one
    generator in ``lifespan`` and dispatches through ``run_in_threadpool`` at
    Cloud Run concurrency 80, so a ``last_usage`` attribute would be read by the
    wrong request. Frozen for the same reason.

    ``output_tokens`` is what the provider bills, and on a thinking model it
    **includes thinking tokens** — they cannot be separated, and with
    ``display: "omitted"`` (the default on Claude Opus 5 and Sonnet 5) the
    thinking blocks come back empty so they cannot be counted here either. Cost
    must be computed from this field; counting the visible ``text`` instead
    understates a high-effort model roughly twofold.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    #: Provider-reported reason generation ended. ``None`` when unmapped.
    stop_reason: str | None = None
    #: Populated only alongside a refusal; providers report nothing otherwise.
    refusal_category: str | None = None

    @property
    def refused(self) -> bool:
        """Whether the model declined on safety grounds.

        Distinct from an empty answer. Both reach the caller as no text, but a
        refusal is the provider's safety classifier and an empty answer is the
        model having nothing to say — conflating them scores a refusal as an
        honest "I don't have that information".
        """
        return self.stop_reason == "refusal"

    @property
    def truncated(self) -> bool:
        """Whether the answer was cut off at ``max_tokens``.

        Worth separating: a truncated answer is a *measurement* failure that
        otherwise scores as a bad answer.
        """
        return self.stop_reason == "max_tokens"


class Generator(ABC):
    """Produces an answer from a system instruction plus a user message."""

    #: Provider-facing model identifier, resolved in ``__init__``.
    model: str
    #: Model used when the caller names none. Lives on the provider rather than
    #: in shared.config because a model id is meaningless across providers.
    default_model: str
    #: Env var holding this provider's credential, so a component can fail at
    #: startup on a missing key instead of on the first request.
    api_key_env: str | None = None

    @abstractmethod
    def generate(self, system_instruction: str, user_message: str) -> Generation:
        """Return the model's answer and its accounting.

        ``Generation.text`` is "" when the model produced no answer, which is the
        pipeline's no-answer case.
        """
        ...

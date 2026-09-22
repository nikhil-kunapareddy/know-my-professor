"""Answer generation via the Anthropic Messages API.

Note on the module name: this file is ``core.llm.anthropic`` while the SDK is the
top-level ``anthropic``. Python 3 resolves ``import anthropic`` below absolutely,
so the two never collide — but do not switch that to a relative import.
"""

from __future__ import annotations

import os

from .base import Generation, Generator

#: /chat runs under REQUEST_BUDGET_SECONDS (45s by default), and answering
#: "who works on X" from eight retrieved chunks is not a reasoning-heavy task.
#: Low effort keeps thinking shallow so the budget is not spent deliberating.
#: Raise this if answer quality turns out to need it — measure with
#: ``python -m evaluation.run_eval`` rather than guessing.
DEFAULT_EFFORT = "low"

#: ``effort`` exists only on the Claude 5 family. Sending it to anything else is
#: a hard 400 — "This model does not support the effort parameter" — so
#: CHAT_MODEL=claude-haiku-4-5-20251001 used to fail on every request, despite
#: CHAT_MODEL being documented as the way to switch models without a rebuild.
#: Matching on the family rather than listing ids keeps new 5-series models
#: working without an edit here.
_EFFORT_MODEL_PREFIXES = ("claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-haiku-5")


def supports_effort(model: str) -> bool:
    """Whether this model accepts ``output_config={"effort": ...}``."""
    return any(model.startswith(prefix) for prefix in _EFFORT_MODEL_PREFIXES)


#: A ceiling, not a target: cited answers run a few hundred tokens. It is set
#: high because hitting the cap truncates mid-sentence, and unused headroom
#: costs nothing (billing is on tokens produced).
DEFAULT_MAX_TOKENS = 16000


class AnthropicGenerator(Generator):
    """Wraps ``anthropic.Anthropic``.

    Two deliberate differences from ``LlamaGenerator``:

    - **No ``temperature``.** Sampling parameters are removed on the Claude 5
      family and return a 400. Determinism is not available as a knob here;
      ``effort`` is the closest equivalent lever.
    - **``effort`` is Claude-5-only.** It is dropped for any other model, which
      is what makes ``CHAT_MODEL`` usable across families rather than a 400.
    - **Thinking is on.** Claude Opus 5 runs adaptive thinking whenever
      ``thinking`` is omitted, so ``response.content`` carries thinking blocks
      alongside text ones and ``generate`` must join only the text. Disabling
      thinking is a documented footgun on this model (it can emit a tool call or
      a ``<thinking>`` tag as visible prose); lowering ``effort`` is the
      supported way to trade depth for latency.
    """

    default_model = "claude-opus-5"
    api_key_env = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        client=None,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str | None = DEFAULT_EFFORT,
    ):
        self.model = model or self.default_model
        self.max_tokens = max_tokens
        # None means "never send it". Otherwise it is sent only where the model
        # accepts it, so an older model can be selected with CHAT_MODEL.
        self.effort = effort if supports_effort(self.model) else None
        self._client = client

    @property
    def client(self):
        """Lazily build the client from ANTHROPIC_API_KEY."""
        if self._client is None:
            import anthropic

            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} env required for generation")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def generate(self, system_instruction: str, user_message: str) -> Generation:
        """Return the model's answer text and its accounting.

        A refusal arrives as a 200 with ``stop_reason == "refusal"``, not as an
        exception, so it is checked explicitly; "" is the pipeline's no-answer case.
        The refusal is now *recorded* rather than only flattened to "", because a
        safety decline and an honest "I don't have that information" are otherwise
        indistinguishable downstream.

        No ``fallbacks`` parameter is set. Server-side fallback would silently
        re-run a declined request on another model and return it under this
        model's name, which would corrupt any per-model measurement.
        """
        options = {"output_config": {"effort": self.effort}} if self.effort else {}
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_instruction,
            messages=[{"role": "user", "content": user_message}],
            **options,
        )

        stop_reason = getattr(response, "stop_reason", None)
        # stop_details is populated ONLY when stop_reason is "refusal" and is
        # None for every other stop reason, so it must be guarded before reading.
        details = getattr(response, "stop_details", None)
        usage = getattr(response, "usage", None)

        if stop_reason == "refusal":
            text = ""
        else:
            # content interleaves thinking and text blocks; only the latter is answer.
            text = "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()

        return Generation(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            stop_reason=stop_reason,
            refusal_category=getattr(details, "category", None),
        )

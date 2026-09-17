"""Answer generation via the Anthropic Messages API.

Note on the module name: this file is ``core.llm.anthropic`` while the SDK is the
top-level ``anthropic``. Python 3 resolves ``import anthropic`` below absolutely,
so the two never collide — but do not switch that to a relative import.
"""

from __future__ import annotations

import os

from .base import Generator

#: /chat runs under REQUEST_BUDGET_SECONDS (45s by default), and answering
#: "who works on X" from eight retrieved chunks is not a reasoning-heavy task.
#: Low effort keeps thinking shallow so the budget is not spent deliberating.
#: Raise this if answer quality turns out to need it — measure with
#: ``python -m evaluation.run_eval`` rather than guessing.
DEFAULT_EFFORT = "low"

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
        effort: str = DEFAULT_EFFORT,
    ):
        self.model = model or self.default_model
        self.max_tokens = max_tokens
        self.effort = effort
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

    def generate(self, system_instruction: str, user_message: str) -> str:
        """Return the model's answer text (empty string if the model returns none).

        A refusal arrives as a 200 with ``stop_reason == "refusal"``, not as an
        exception, so it is checked explicitly; "" is the pipeline's no-answer case.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_instruction,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": user_message}],
        )

        if getattr(response, "stop_reason", None) == "refusal":
            return ""

        # content interleaves thinking and text blocks; only the latter is answer.
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

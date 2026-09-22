"""Answer generation via Meta's native Llama API."""

from __future__ import annotations

import os

from .base import Generation, Generator


class LlamaGenerator(Generator):
    """Wraps ``LlamaAPIClient``.

    Uses the NATIVE Llama API — response text lives at
    ``completion_message.content.text``, not the OpenAI-compatible shape — because
    the provisioned key is authorized only for the native endpoint. Confining that
    detail to this class is the whole point of the ``Generator`` interface.
    """

    default_model = "Llama-4-Maverick-17B-128E-Instruct-FP8"
    api_key_env = "LLAMA_API_KEY"

    def __init__(self, client=None, model: str | None = None, temperature: float = 0.0):
        self.model = model or self.default_model
        self.temperature = temperature
        self._client = client

    @property
    def client(self):
        """Lazily build the client from LLAMA_API_KEY."""
        if self._client is None:
            from llama_api_client import LlamaAPIClient

            key = os.environ.get("LLAMA_API_KEY")
            if not key:
                raise RuntimeError("LLAMA_API_KEY env required for generation")
            self._client = LlamaAPIClient(api_key=key)
        return self._client

    def generate(self, system_instruction: str, user_message: str) -> Generation:
        """Return the model's answer text and its accounting."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
        )
        prompt_tokens, completion_tokens = _token_metrics(response)
        return Generation(
            text=(response.completion_message.content.text or "").strip(),
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            # The native API reports no stop reason in the shape Anthropic does,
            # so refusals are not distinguishable here. Left None rather than
            # guessed: this provider is not a candidate in the model-selection
            # experiment, and a fabricated value would read as a real one.
            stop_reason=None,
        )


def _token_metrics(response) -> tuple[int, int]:
    """Pull prompt/completion counts out of the native API's metrics list.

    Llama reports usage as ``metrics: list[{metric, value, unit}]`` rather than
    Anthropic's ``usage`` object, so the counts are looked up by substring on the
    metric name. Missing metrics read 0 — the field is optional on the response.
    """
    totals = {"prompt": 0, "completion": 0}
    for entry in getattr(response, "metrics", None) or []:
        name = (getattr(entry, "metric", "") or "").lower()
        for key in totals:
            if key in name:
                totals[key] = int(getattr(entry, "value", 0) or 0)
    return totals["prompt"], totals["completion"]

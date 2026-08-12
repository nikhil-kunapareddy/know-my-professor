"""Answer generation via Meta's native Llama API."""

from __future__ import annotations

import os

from shared.config import DEFAULT_CHAT_MODEL

from .base import Generator


class LlamaGenerator(Generator):
    """Wraps ``LlamaAPIClient``.

    Uses the NATIVE Llama API — response text lives at
    ``completion_message.content.text``, not the OpenAI-compatible shape — because
    the provisioned key is authorized only for the native endpoint. Confining that
    detail to this class is the whole point of the ``Generator`` interface.
    """

    api_key_env = "LLAMA_API_KEY"

    def __init__(self, client=None, model: str = DEFAULT_CHAT_MODEL, temperature: float = 0.0):
        self.model = model
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

    def generate(self, system_instruction: str, user_message: str) -> str:
        """Return the model's answer text (empty string if the model returns none)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
        )
        return (response.completion_message.content.text or "").strip()

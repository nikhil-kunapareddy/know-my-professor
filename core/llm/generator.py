"""Answer generation via Meta's native Llama API."""

from __future__ import annotations

from shared.config import DEFAULT_CHAT_MODEL


class AnswerGenerator:
    """Generates a grounded answer from a system instruction + user message.

    Wraps a ``LlamaAPIClient``. We use the NATIVE Llama API (response text lives
    at ``completion_message.content.text``), not the OpenAI-compat endpoint —
    the provisioned key is only authorized for the native API.
    """

    def __init__(self, client, model: str = DEFAULT_CHAT_MODEL):
        self.client = client
        self.model = model

    def generate(self, system_instruction: str, user_message: str) -> str:
        """Return the model's answer text (empty string if the model returns none)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )
        return (response.completion_message.content.text or "").strip()

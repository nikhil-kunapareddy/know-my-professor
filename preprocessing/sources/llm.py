"""The Claude call shared by the enrichment sources (publications, grants).

One lazily built client, the project's one retry policy, and a running token
count so each job can report what it spent. Structured calls use
``output_config.format``, which constrains generation to the schema: the reply
is valid JSON by construction, except when ``max_tokens`` cuts it short.

The profiles LLM parser predates this and keeps its own copy of the call; it is
pinned by its own tests and nothing here needed to move it.
"""

from __future__ import annotations

import json
import os
import threading

from shared.retry import with_backoff

DEFAULT_MODEL = "claude-haiku-4-5"
MAX_RETRIES = 6


def is_retryable(exc: BaseException) -> bool:
    """True for Anthropic's 429 and 5xx. Imported lazily so tests need no SDK."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - SDK always present where this runs
        return False
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)):
        return True
    return isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500


class Claude:
    """A thread-safe Claude caller that counts the tokens it spends."""

    def __init__(self, client=None, model: str = DEFAULT_MODEL, label: str = "claude"):
        self.model = model
        self.label = label
        self._client = client
        self._lock = threading.Lock()
        self.calls = self.input_tokens = self.output_tokens = 0

    @property
    def client(self):
        """Built from ANTHROPIC_API_KEY on first use. ``max_retries=0``: retries
        are ``with_backoff``'s job, and two layers would compound."""
        if self._client is None:
            from anthropic import Anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(f"ANTHROPIC_API_KEY env required for {self.label}")
            self._client = Anthropic(max_retries=0)
        return self._client

    def _create(self, **kwargs):
        response = with_backoff(
            lambda: self.client.messages.create(model=self.model, **kwargs),
            is_retryable=is_retryable,
            max_attempts=MAX_RETRIES,
            label=self.label,
        )
        usage = getattr(response, "usage", None)
        with self._lock:
            self.calls += 1
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        return response

    @staticmethod
    def _text(response) -> str:
        return next((b.text for b in response.content if b.type == "text"), "")

    def json(self, prompt: str, schema: dict, max_tokens: int) -> dict | None:
        """One structured call. None if truncated or unparseable -- never half a dict."""
        response = self._create(
            max_tokens=max_tokens,
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "max_tokens":
            return None
        try:
            parsed = json.loads(self._text(response))
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def text(self, prompt: str, max_tokens: int) -> str:
        """One plain-text call, stripped. Empty if the model was cut off."""
        response = self._create(max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        if response.stop_reason == "max_tokens":
            return ""
        return self._text(response).strip()

    def usage_line(self) -> str:
        return (
            f"{self.label}: {self.calls} call(s), {self.input_tokens:,} input / "
            f"{self.output_tokens:,} output tokens ({self.model})"
        )

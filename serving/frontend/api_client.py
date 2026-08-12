"""HTTP client for the /chat API.

Separated from the Streamlit page so the transport can be tested without a
browser session: every test in tests/test_frontend.py drives this class with a
fake transport.

Responses are parsed into the shared pydantic models rather than poked at as
dicts, so a field rename on the API side fails here loudly instead of rendering
as a blank line in the UI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
from pydantic import ValidationError

from shared.schemas import API_VERSION, ChatResponse

DEFAULT_TIMEOUT_SECONDS = 60
#: One retry only: the API is already slow (a model call), so a long retry chain
#: just keeps the user waiting past the point they would have refreshed.
DEFAULT_RETRIES = 1
RETRY_BACKOFF_SECONDS = 1.0

#: Statuses worth retrying: transient upstream conditions, not user error.
_RETRYABLE_STATUSES = {429, 502, 503, 504}


class ChatError(Exception):
    """A user-presentable failure.

    ``message`` is safe to show; ``request_id`` (when the API supplied one) lets
    a report be matched to the server-side log line.
    """

    def __init__(self, message: str, request_id: str = "", retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.request_id = request_id
        self.retryable = retryable


@dataclass
class ChatClient:
    """Talks to the /chat API and returns a validated ChatResponse."""

    base_url: str
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    retries: int = DEFAULT_RETRIES
    session: requests.Session | None = None
    sleep = staticmethod(time.sleep)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        if self.session is None:
            self.session = requests.Session()

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/{API_VERSION}/chat"

    def ask(self, question: str) -> ChatResponse:
        """Ask a question. Raises ChatError with a message fit to display."""
        last: ChatError | None = None

        for attempt in range(self.retries + 1):
            try:
                return self._attempt(question)
            except ChatError as e:
                last = e
                if not e.retryable or attempt == self.retries:
                    raise
                self.sleep(RETRY_BACKOFF_SECONDS)

        raise last  # pragma: no cover - loop always returns or raises

    def _attempt(self, question: str) -> ChatResponse:
        try:
            response = self.session.post(
                self.chat_url,
                json={"question": question},
                timeout=self.timeout,
            )
        except requests.Timeout as e:
            raise ChatError(
                "The API took too long to respond. Try again.", retryable=True
            ) from e
        except requests.RequestException as e:
            raise ChatError(
                "Couldn't reach the API. Check your connection and try again.",
                retryable=True,
            ) from e

        request_id = response.headers.get("X-Request-ID", "")

        if response.status_code >= 400:
            raise ChatError(
                self._explain(response),
                request_id=request_id,
                retryable=response.status_code in _RETRYABLE_STATUSES,
            )

        try:
            return ChatResponse.model_validate(response.json())
        except (ValueError, ValidationError) as e:
            raise ChatError(
                "The API returned a response this app doesn't understand.",
                request_id=request_id,
            ) from e

    @staticmethod
    def _explain(response: requests.Response) -> str:
        """A human-readable line for an error status.

        Prefers the API's own ``detail`` (already written for end users and
        scrubbed of upstream text) and falls back to a generic message.
        """
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None

        if detail:
            return str(detail)
        if response.status_code == 429:
            return "The service is busy right now. Try again in a moment."
        if response.status_code >= 500:
            return "The API had a problem answering. Try again."
        return f"The API rejected the request (HTTP {response.status_code})."

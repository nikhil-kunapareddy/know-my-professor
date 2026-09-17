"""Mapping upstream provider failures onto HTTP status codes.

The service previously answered every failure with ``502`` and the upstream
exception's text in the body. That is wrong twice: a caller cannot tell "you are
being rate limited, retry later" from "the model is broken", and provider error
strings can carry internal detail (request URLs, quota identifiers, occasionally
key fragments) straight to an unauthenticated client.

Classification is duck-typed rather than provider-specific, so a newly added
provider is categorised correctly without editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifiedError:
    """How to answer the client, with nothing sensitive in it."""

    status_code: int
    code: str
    detail: str


_RATE_LIMIT_HINTS = ("ratelimit", "resourceexhausted", "toomanyrequests", "quota")
_TIMEOUT_HINTS = ("timeout", "deadline")


def classify_upstream_error(exc: BaseException) -> ClassifiedError:
    """Categorise a provider exception without echoing its message."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    name = type(exc).__name__.lower()

    if status == 429 or any(hint in name for hint in _RATE_LIMIT_HINTS):
        return ClassifiedError(
            status_code=429,
            code="rate_limited",
            detail="Upstream model provider is rate limiting. Try again shortly.",
        )

    if isinstance(exc, TimeoutError) or any(hint in name for hint in _TIMEOUT_HINTS):
        return ClassifiedError(
            status_code=504,
            code="upstream_timeout",
            detail="The answer took too long to generate. Try again.",
        )

    if isinstance(status, int) and 400 <= status < 500:
        return ClassifiedError(
            status_code=502,
            code="upstream_rejected",
            detail="The model provider rejected the request.",
        )

    return ClassifiedError(
        status_code=502,
        code="upstream_error",
        detail="The answer pipeline failed. The failure has been logged.",
    )

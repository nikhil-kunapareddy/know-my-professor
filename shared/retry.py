"""Provider-agnostic retry with exponential backoff.

Every API client this project talks to (Mistral, Gemini, Llama) signals "slow
down" with its own exception type, so the retry *policy* used to be copy-pasted
next to each one. The policy lives here; each caller supplies only the predicate
that recognises its provider's rate-limit error.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def with_backoff(
    operation: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool],
    max_attempts: int,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    label: str = "request",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``operation`` until it succeeds, doubling the wait after each retry.

    Re-raises immediately if ``is_retryable`` says the error is not transient,
    and re-raises the final attempt's error once ``max_attempts`` is spent —
    so a genuine outage still fails loudly instead of hanging.

    ``sleep`` is injectable so tests don't spend real seconds.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except BaseException as e:  # noqa: BLE001 - re-raised unless retryable
            if not is_retryable(e) or attempt == max_attempts:
                raise
            print(f"    {label} rate-limited (attempt {attempt}/{max_attempts}); sleeping {delay:.1f}s")
            sleep(delay)
            delay = min(delay * 2, max_delay)

    raise AssertionError("unreachable")  # pragma: no cover

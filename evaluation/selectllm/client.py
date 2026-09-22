"""One structured Anthropic call, cached on disk.

No evaluation framework. `client.messages.parse()` with a schema is enforced
server-side, so a validated object comes back and there is no JSON left to fail
to parse — which is the failure DeepEval hits 1 call in 35 by hand-rolling it.
``topk/judge.py`` proved the shape: 900 calls, $9.62, zero errors.

The cache is keyed on everything that could change an answer, so editing a
prompt invalidates correctly while a re-run costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from shared.retry import with_backoff

from .arms import PRICING

CACHE_DIR = Path(__file__).resolve().parents[2] / ".selectllm_cache"
MAX_TOKENS = 8000
T = TypeVar("T", bound=BaseModel)

_client = None
_client_lock = threading.Lock()


def client():
    """One lazily built client, guarded so threads do not race the constructor."""
    global _client
    with _client_lock:
        if _client is None:
            import anthropic

            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY env required")
            _client = anthropic.Anthropic(api_key=key)
        return _client


class Spend:
    """Running cost, so a run can stop at a cap instead of past it."""

    def __init__(self, cap_usd: float):
        self.cap_usd = cap_usd
        self.usd = 0.0
        self.calls = 0
        self.cached = 0
        self._lock = threading.Lock()

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        price = PRICING[model]
        with self._lock:
            self.usd += (input_tokens * price.input + output_tokens * price.output) / 1e6
            self.calls += 1
            if self.usd > self.cap_usd:
                raise SpendCapExceeded(
                    f"spend ${self.usd:.2f} exceeded the ${self.cap_usd:.2f} cap "
                    f"after {self.calls} calls"
                )

    def hit(self) -> None:
        with self._lock:
            self.cached += 1

    def __str__(self) -> str:
        return (
            f"${self.usd:.2f} over {self.calls} calls "
            f"(+{self.cached} cache hits, cap ${self.cap_usd:.0f})"
        )


class SpendCapExceeded(RuntimeError):
    """Raised instead of quietly spending past the authorised cap."""


class JudgeRefusal(RuntimeError):
    """The judge declined. Recorded, never silently scored as a verdict."""


def _is_retryable(exc: BaseException) -> bool:
    """Retry transport and truncation, never a policy decline.

    A refusal is the judge's safety classifier deciding about this exact prompt,
    so retrying it just pays twice for the same answer. A missing parsed output,
    by contrast, is usually a max_tokens truncation and does succeed on retry.
    """
    if isinstance(exc, (JudgeRefusal, SpendCapExceeded)):
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        return True
    return status in {408, 409, 429} or status >= 500


def ask(
    schema: type[T],
    *,
    model: str,
    system: str,
    user: str,
    version: str,
    spend: Spend,
    effort: str | None = "low",
    cache_dir: Path | None = CACHE_DIR,
) -> T:
    """Return a schema-validated response, reading the disk cache first.

    A refusal raises rather than returning a default: a declined call scored as
    a verdict is a silent wrong answer, which is worse than a loud missing one.
    """
    path = None
    if cache_dir is not None:
        digest = hashlib.sha256(
            "\x00".join([version, model, str(effort), system, user]).encode()
        ).hexdigest()
        path = cache_dir / f"{digest}.json"
        if path.exists():
            try:
                spend.hit()
                return schema.model_validate_json(path.read_text())
            except (OSError, ValueError):
                pass  # an unreadable entry is a miss, not an error

    def call() -> T:
        options = {"output_config": {"effort": effort}} if effort else {}
        response = client().messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            output_format=schema,
            messages=[{"role": "user", "content": user}],
            **options,
        )
        if response.stop_reason == "refusal":
            raise JudgeRefusal(
                f"declined: {getattr(response.stop_details, 'category', None)}"
            )
        parsed = next(
            (b.parsed_output for b in response.content
             if getattr(b, "parsed_output", None)),
            None,
        )
        if parsed is None:
            # Almost always a max_tokens truncation of the schema.
            raise RuntimeError(f"no parsed output (stop={response.stop_reason})")
        spend.add(model, response.usage.input_tokens, response.usage.output_tokens)
        return parsed

    result = with_backoff(call, is_retryable=_is_retryable, max_attempts=4, label="ask")

    if path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.model_dump()))
    return result

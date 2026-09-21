"""Profile HTML -> clean text -> structured ``Profile``, via Claude.

The Khoury directory renders profile sections as ``div.accordion-item`` blocks,
so ``profile_parser.ProfileParser`` can read them exactly and for free. The other
Northeastern college directories run a different template (the ``nu-*`` design
system) with no equivalent structure, and each would need its own selectors.

This parser sidesteps that: trafilatura strips boilerplate down to main content,
and Claude normalizes arbitrary clean text into the same ``Profile`` shape. One
implementation covers every non-Khoury template, present and future.

Only ever used for colleges whose ``College.parser`` is ``"llm"``. Khoury keeps
the accordion parser, because its chunk text is pinned byte-for-byte by
``tests/test_chunk_golden.py`` and an LLM cannot reproduce it.
"""

from __future__ import annotations

import json
import os

import trafilatura

from shared.retry import with_backoff

from .config import (
    EXTRACT_MAX_CLEAN_CHARS,
    EXTRACT_MAX_RETRIES,
    EXTRACT_MAX_TOKENS,
    EXTRACT_MIN_CLEAN_CHARS,
    EXTRACT_MODEL,
    EXTRACT_PROMPT,
    EXTRACTION_SCHEMA,
)
from .models import Profile


class ThinProfilePage(Exception):
    """The page yielded too little text to be a real profile.

    Raised rather than returning an empty ``Profile`` so the runner can skip the
    record instead of writing a hollow one that ingest would later filter out
    anyway -- and, more importantly, so a JS-only or error page is never cached
    as a legitimately-empty profile.
    """


def _is_retryable(exc: BaseException) -> bool:
    """True for Anthropic's 429 and 5xx. Imported lazily so tests need no SDK."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - SDK always present where this runs
        return False
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)):
        return True
    return isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500


class LlmProfileParser:
    """Parses any faculty profile page into a ``Profile`` using Claude.

    Signature-compatible with ``ProfileParser`` (``parse(url, html)``) so the
    runner can hold either one and never branch on which it has.
    """

    def __init__(self, client=None, model: str = EXTRACT_MODEL):
        self.model = model
        self._client = client

    @property
    def client(self):
        """Lazily build the Anthropic client from ANTHROPIC_API_KEY.

        ``max_retries=0`` because retries are this project's one policy, in
        ``shared.retry.with_backoff``. Leaving the SDK's own retries on would
        compound with it -- 6 backoff attempts each retried twice is 18 calls
        against a rate limit we are already backing off from.
        """
        if self._client is None:
            from anthropic import Anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY env required for profile extraction")
            self._client = Anthropic(max_retries=0)
        return self._client

    @staticmethod
    def clean(html: str) -> str:
        """Main-content text of a profile page, truncated to the token bound."""
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        return (text or "").strip()[:EXTRACT_MAX_CLEAN_CHARS]

    def parse(self, url: str, html: str) -> Profile:
        """Extract one profile. Raises ``ThinProfilePage`` if the page is empty."""
        clean_text = self.clean(html)
        if len(clean_text) < EXTRACT_MIN_CLEAN_CHARS:
            raise ThinProfilePage(f"{len(clean_text)} chars of usable text")

        fields = self._extract(clean_text)
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        return Profile(
            slug=slug,
            url=url,
            name=fields.get("name") or None,
            pronouns=None,
            title=fields.get("title") or None,
            photo_url=None,
            roles=[],
            campuses=[],
            websites=[],
            google_scholar=None,
            areas_of_interest=fields.get("areas_of_interest") or [],
            contact=[],
            research_interests=fields.get("research_interests") or [],
            education=fields.get("education") or [],
            biography=fields.get("biography") or None,
            labs_and_groups=fields.get("labs_and_groups") or [],
            projects=fields.get("projects") or [],
            raw_aside={},
        )

    def _extract(self, clean_text: str) -> dict:
        """One structured-output call, backing off on 429/5xx.

        ``output_config.format`` is the Anthropic equivalent of Gemini's
        ``response_schema``: the server constrains generation to the schema, so
        the response is valid JSON by construction rather than by hope.
        """

        def call():
            return self.client.messages.create(
                model=self.model,
                max_tokens=EXTRACT_MAX_TOKENS,
                output_config={"format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA}},
                messages=[{"role": "user", "content": EXTRACT_PROMPT + clean_text}],
            )

        response = with_backoff(
            call,
            is_retryable=_is_retryable,
            max_attempts=EXTRACT_MAX_RETRIES,
            label="claude profile extract",
        )

        if response.stop_reason == "max_tokens":
            # A truncated structured response is not valid JSON. Surfacing it as
            # a thin page skips the record rather than writing half a profile.
            raise ThinProfilePage("extraction hit max_tokens")

        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except ValueError as e:
            raise ThinProfilePage(f"unparseable extraction: {e}") from e

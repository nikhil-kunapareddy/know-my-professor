"""Tiered profile parsing: exact DOM rules first, Claude only when they come back hollow.

Khoury's accordion parser is exact and free, but it is selectors against one
template. When the template changes, it does not fail -- it returns a profile
with a name and nothing else, the scraper stores it, and ingest's substantive
filter quietly drops the professor. Nothing errors; the corpus just shrinks.

``TieredProfileParser`` keeps tier 0 in charge and adds a tier-2 fallback:

1. Parse with the DOM parser. A profile with a name and substantive content
   (the same test ingest applies) is returned untouched, so a working template
   produces byte-identical records and costs nothing.
2. Otherwise, run the LLM parser on the same HTML. If it finds substance, the
   two are merged: every field the DOM parser did read wins (its header fields
   are exact), and the gaps are filled from the extraction.
3. If the LLM finds nothing either -- a genuinely empty staff page -- or fails,
   the DOM result stands. The fallback never makes a record worse, and a thin
   page never reaches the API: ``LlmProfileParser`` rejects it on length first.

``Profile.parser_tier`` records which path produced each record.
"""

from __future__ import annotations

from dataclasses import asdict, fields

from .llm_parser import LlmProfileParser, ThinProfilePage
from .models import Profile
from .profile_parser import ProfileParser
from .source import ProfileSource

_SOURCE = ProfileSource()


def has_substance(profile: Profile) -> bool:
    """Named, and ingestable by the same rule ingest applies."""
    return bool(profile.name) and _SOURCE.is_ingestable(asdict(profile))


class TieredProfileParser:
    """``parse(url, html)`` like either parser it wraps."""

    def __init__(self, primary: ProfileParser, fallback: LlmProfileParser):
        self.primary = primary
        self.fallback = fallback

    def parse(self, url: str, html: str) -> Profile:
        exact = self.primary.parse(url, html)
        if has_substance(exact):
            return exact
        try:
            extracted = self.fallback.parse(url, html)
        except ThinProfilePage:
            return exact
        except Exception as e:  # noqa: BLE001 - the exact result is still a valid record
            print(f"    warning: LLM fallback failed for {url}: {type(e).__name__}: {e}")
            return exact
        if not _SOURCE.is_ingestable(asdict(extracted)):
            return exact
        return merge(exact, extracted)


def merge(exact: Profile, extracted: Profile) -> Profile:
    """Fields the DOM parser read win; its gaps are filled from the extraction."""
    values = {f.name: getattr(exact, f.name) or getattr(extracted, f.name) for f in fields(Profile)}
    values.update(slug=exact.slug, url=exact.url, parser_tier="accordion+llm")
    return Profile(**values)

"""Banner self-service client.

Banner will not return instructors in bulk: ``searchResults`` always reports an
empty ``faculty`` list, and the real data only comes from
``getFacultyMeetingTimes`` one section at a time. That single fact is what makes
this source expensive (~9,700 requests for a full term) and is why it is
separate from the catalog scrape.
"""

from __future__ import annotations

import requests

from .config import (
    BANNER_BASE,
    COMPLETED_TERM_MARKER,
    REQUEST_TIMEOUT_SECONDS,
    SCRAPER_USER_AGENT,
    SEARCH_PAGE_SIZE,
    TERM_FETCH_COUNT,
)


class BannerClient:
    """Reads terms, sections, and per-section instructors from Banner."""

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": SCRAPER_USER_AGENT})
        self._term: str | None = None

    def _get(self, path: str, **params):
        r = self.session.get(f"{BANNER_BASE}/{path}", params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        r.raise_for_status()
        return r.json()

    def start(self) -> None:
        """Establish the session cookie Banner requires before any search."""
        self.session.get(
            f"{BANNER_BASE}/classSearch/classSearch", timeout=REQUEST_TIMEOUT_SECONDS
        ).raise_for_status()

    def terms(self) -> list[dict]:
        """Every term Banner exposes, newest first."""
        return self._get(
            "classSearch/getTerms", searchTerm="", offset=1, max=TERM_FETCH_COUNT
        )

    @staticmethod
    def is_active(term: dict) -> bool:
        """True for a term that is current or upcoming, not an archive.

        Banner has no "is current" field; the ``(View Only)`` suffix on the
        description is the distinction it actually exposes.
        """
        return COMPLETED_TERM_MARKER not in (term.get("description") or "")

    def active_terms(self) -> list[dict]:
        """Terms worth scraping — everything not yet completed."""
        return [t for t in self.terms() if self.is_active(t)]

    def select_term(self, term_code: str) -> None:
        """Banner scopes searches to a term held in session state, not a param."""
        self.session.post(
            f"{BANNER_BASE}/term/search",
            params={"mode": "search"},
            data={"term": term_code},
            timeout=REQUEST_TIMEOUT_SECONDS,
        ).raise_for_status()
        self._term = term_code

    def sections(self, term_code: str, subject: str | None = None) -> list[dict]:
        """Every section in a term (optionally one subject), paging as needed."""
        if self._term != term_code:
            self.select_term(term_code)
        out: list[dict] = []
        offset = 0
        while True:
            params = {
                "txt_term": term_code,
                "pageOffset": offset,
                "pageMaxSize": SEARCH_PAGE_SIZE,
            }
            if subject:
                params["txt_subject"] = subject
            payload = self._get("searchResults/searchResults", **params)
            rows = payload.get("data") or []
            out.extend(rows)
            total = payload.get("totalCount") or 0
            offset += len(rows)
            if not rows or offset >= total:
                return out

    def instructors(self, term_code: str, crn: str) -> list[dict]:
        """Instructors for one section. The expensive call — one per section."""
        payload = self._get(
            "searchResults/getFacultyMeetingTimes", term=term_code, courseReferenceNumber=crn
        )
        return [f for mt in (payload.get("fmt") or []) for f in (mt.get("faculty") or [])]

    @staticmethod
    def display_to_name(display_name: str) -> str:
        """``"Suciu, Alexandru"`` -> ``"Alexandru Suciu"``.

        Banner writes surname-first; the professor corpus stores natural order,
        and matching the two is what lets a course cite a person we know about.
        """
        parts = [p.strip() for p in (display_name or "").split(",")]
        return f"{parts[1]} {parts[0]}" if len(parts) == 2 and parts[1] else (display_name or "")

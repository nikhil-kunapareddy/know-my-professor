"""Catalog HTML -> course records.

CourseLeaf renders every course as a ``div.courseblock`` holding a title line,
a description, and zero or more "extra" lines (prerequisites, corequisites,
NUpath attributes). That structure is identical across all ~227 subjects, so
unlike the faculty directories this needs no per-page LLM extraction -- and
shouldn't have one: 6,500 deterministic parses for free beats 6,500 API calls.
"""

from __future__ import annotations

import time

import requests
from bs4 import BeautifulSoup

from .config import (
    CATALOG_INDEX,
    COURSE_TITLE_RE,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    SCRAPER_USER_AGENT,
    SUBJECT_HREF_RE,
)


class CatalogFetcher:
    """Fetches catalog pages and parses them into course records."""

    def __init__(self, delay: float = REQUEST_DELAY_SECONDS):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SCRAPER_USER_AGENT})

    def fetch(self, url: str) -> str:
        response = self.session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text

    def discover_subjects(self) -> list[str]:
        """Every subject code the catalog index links to, e.g. ``cs``, ``math``."""
        soup = BeautifulSoup(self.fetch(CATALOG_INDEX), "html.parser")
        subjects = {
            m.group(1)
            for a in soup.find_all("a", href=True)
            if (m := SUBJECT_HREF_RE.search(a["href"]))
        }
        return sorted(subjects)

    def fetch_subject(self, subject: str) -> list[dict]:
        """All course records for one subject, paced by the politeness delay."""
        time.sleep(self.delay)
        url = f"{CATALOG_INDEX}{subject}/"
        return self.parse_courses(self.fetch(url), subject, url)

    @staticmethod
    def parse_courses(html: str, subject: str, url: str = "") -> list[dict]:
        """Parse every ``div.courseblock`` on a subject page."""
        soup = BeautifulSoup(html, "html.parser")
        records: list[dict] = []
        for block in soup.select("div.courseblock"):
            record = CatalogFetcher._parse_block(block, subject, url)
            if record:
                records.append(record)
        return records

    @staticmethod
    def _text(el) -> str:
        """Visible text with non-breaking spaces normalised.

        CourseLeaf writes course codes as ``CS\xa02000`` inside requisite lines.
        Left alone that lands in the chunk text, which feeds ``content_hash`` —
        so it would both read oddly and make the stored hash depend on an
        invisible character.
        """
        return el.get_text(" ", strip=True).replace("\xa0", " ") if el else ""

    @staticmethod
    def _parse_block(block, subject: str, url: str) -> dict | None:
        """One courseblock -> a record, or None if the title line is unreadable."""
        title_el = block.select_one("p.courseblocktitle")
        if not title_el:
            return None
        m = COURSE_TITLE_RE.match(CatalogFetcher._text(title_el))
        if not m:
            return None
        subject_code, number, title, credits = m.groups()

        desc_el = block.select_one("p.cb_desc")
        extras = [t for el in block.select("p.courseblockextra") if (t := CatalogFetcher._text(el))]
        return {
            "code": f"{subject_code} {number}",
            # The ingest-side entity id; lowercase and unspaced so it is stable
            # whether the catalog writes "CS 1800" or "CS1800".
            "slug": f"{subject_code}{number}".lower(),
            "subject": subject_code,
            "subject_slug": subject,
            "number": number,
            "title": title.strip(),
            "credits": (credits or "").strip(),
            "description": CatalogFetcher._text(desc_el),
            # Prerequisites / corequisites / NUpath attributes. Kept as lines on
            # the same record rather than their own section: each is a short
            # structured fact that embeds poorly alone ("Prerequisite(s): CS 1800")
            # and would add thousands of near-duplicate low-value vectors.
            "requisites": extras,
            "url": url,
        }

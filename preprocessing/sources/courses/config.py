"""Configuration owned by the courses source.

The catalog is the *entity-defining* half of course data: it lists every course
Northeastern publishes, university-wide, and is stable between terms. Who
teaches a course is term-scoped and lives in Banner -- see
``preprocessing/sources/sections``.
"""

from __future__ import annotations

import re

CATALOG_BASE = "https://catalog.northeastern.edu"
CATALOG_INDEX = f"{CATALOG_BASE}/course-descriptions/"

#: Subject pages look like /course-descriptions/cs/ — the index links to ~227.
SUBJECT_HREF_RE = re.compile(r"/course-descriptions/([a-z0-9-]+)/$")

#: "CS 1800.  Discrete Structures.  (4 Hours)" -> code, title, credits.
#: The separator is a literal period-then-spaces, and titles contain periods
#: themselves ("Prof. Practice"), so the number is what anchors the split.
COURSE_TITLE_RE = re.compile(
    r"^\s*([A-Z]{2,5})\s+(\d{3,4}[A-Z]?)\.\s+(.*?)\.\s*(?:\(([^)]*)\))?\s*$"
)

SCRAPER_USER_AGENT = (
    "KhouryFacultyScraper/0.1 (personal research project; "
    "contact: kunapareddy.s@northeastern.edu)"
)
#: robots.txt for catalog.northeastern.edu disallows only /wp-admin/-style paths
#: and sets no Crawl-delay; 1s is our own politeness floor.
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30

#: A description shorter than this is a stub the catalog never filled in.
MIN_DESCRIPTION_CHARS = 40

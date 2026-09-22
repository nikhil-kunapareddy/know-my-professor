"""Configuration owned by the sections source.

Banner is the *term-scoped* half of course data. The catalog says what CS 3800
is; Banner says who is teaching it this term. They are separate sources because
they change on completely different cadences -- the catalog is stable for a
year, Banner turns over every term -- and because Banner costs ~9,700 requests
against the catalog's ~227.
"""

from __future__ import annotations

BANNER_BASE = "https://nubanner.neu.edu/StudentRegistrationSsb/ssb"

#: Banner marks a finished term by appending this to its description. It is the
#: only signal distinguishing "registration is open" from "archive", and using
#: it means the scraper needs no hardcoded term list: Spring 2027 is picked up
#: the day Banner publishes it, and a term drops out once it completes.
COMPLETED_TERM_MARKER = "(View Only)"

#: Banner exposes ~60 terms, nearly all archived. Fetch enough to reach the
#: active ones without paging the whole history.
TERM_FETCH_COUNT = 30

#: Sections per search-results page. Banner accepts large pages; this keeps the
#: section listing to ~20 requests instead of ~200.
SEARCH_PAGE_SIZE = 500

SCRAPER_USER_AGENT = (
    "KhouryFacultyScraper/0.1 (personal research project; "
    "contact: kunapareddy.s@northeastern.edu)"
)
#: nubanner.neu.edu publishes no robots.txt (404) and sets no Crawl-delay, so
#: this is purely our own politeness floor. One instructor lookup per section
#: over ~9,700 sections is the bulk of the run, so the value matters.
REQUEST_DELAY_SECONDS = 0.3
REQUEST_TIMEOUT_SECONDS = 40
#: Concurrent instructor lookups. Fetching stays serialized behind the Pacer
#: regardless; this only overlaps network latency.
DEFAULT_WORKERS = 8

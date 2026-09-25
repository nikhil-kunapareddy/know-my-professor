"""Configuration owned by the profiles source.

Lives here rather than in ``shared/config.py`` because nothing outside this
package needs it -- ``shared`` is reserved for constants that BOTH the serving
and data sides must agree on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Re-exported: callers here think in terms of colleges, and the id scheme that
# names one is shared with every other source (see ..entities).

KHOURY_BASE = "https://www.khoury.northeastern.edu"
KHOURY_LISTING = f"{KHOURY_BASE}/people/"
PROFILE_URL_RE = re.compile(rf"^{re.escape(KHOURY_BASE)}/people/[a-z0-9-]+/$")

SCRAPER_USER_AGENT = (
    "KhouryFacultyScraper/0.1 (personal research project; "
    "contact: kunapareddy.s@northeastern.edu)"
)
SCRAPER_REQUEST_DELAY_SECONDS = 1.0
SCRAPER_REQUEST_TIMEOUT_SECONDS = 30

# Local-only output dir (gitignored); GCS is the source of truth in production.
LOCAL_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data"


# --- Colleges -------------------------------------------------------------
#
# The corpus started as Khoury only. Other Northeastern colleges publish the
# same kind of directory under /people/, but NOT the same HTML: Khoury renders
# profile sections as ``div.accordion-item`` blocks, while the newer college
# sites (College of Science and friends) use the ``nu-*`` design system with no
# equivalent structure. Rather than hand-write a DOM parser per template, a
# college declares which extraction strategy it needs:
#
#   "accordion" -> profile_parser.ProfileParser (exact, free, Khoury-only)
#   "llm"       -> llm_parser.LlmProfileParser  (trafilatura + Claude, any HTML)
#
# Discovery differs too. Khoury and CoS are found by walking the paginated
# /people/ listing. Every other college is found from its Yoast sitemap: their
# listings render a curated subset (D'Amore-McKim's showed 12 of 512 people,
# CAMD's 10 of 498), while the sitemap enumerates every published profile.
#
# robots.txt for every host below was checked to allow its profile paths
# (2026-09-20; coe, cssh, bouve, law 2026-09-23). camd, cssh, bouve and law ask
# for a 10s crawl delay, which ``crawl_delay`` honours.


@dataclass(frozen=True)
class College:
    """One faculty directory: where it lives and how to read a profile page."""

    key: str
    base: str
    parser: str = "llm"
    crawl_delay: float = SCRAPER_REQUEST_DELAY_SECONDS
    #: URL segments profile pages live under. Most directories use /people/;
    #: CSSH (/faculty/ + /person/), Bouvé (/directory/) and Law (/faculty/) don't.
    profile_paths: tuple[str, ...] = ("people",)
    #: Yoast sitemap stems to discover profiles from: ``"faculty"`` selects
    #: ``faculty-sitemap.xml`` and the ``faculty-sitemap2.xml``... pages Yoast
    #: splits it into past 1,000 entries. Empty means walk the listing instead.
    sitemaps: tuple[str, ...] = ()

    @property
    def listing(self) -> str:
        """The paginated directory index (listing-walk colleges only)."""
        return f"{self.base}/{self.profile_paths[0]}/"

    @property
    def sitemap_index(self) -> str:
        return f"{self.base}/sitemap_index.xml"

    @property
    def profile_url_re(self) -> re.Pattern[str]:
        """Matches a profile page, excluding the listing and its /page/N/ links."""
        paths = "|".join(map(re.escape, self.profile_paths))
        return re.compile(rf"^{re.escape(self.base)}/(?:{paths})/[a-z0-9-]+/$")

    @property
    def sitemap_url_re(self) -> re.Pattern[str]:
        """Matches this college's profile sitemaps within its sitemap index."""
        stems = "|".join(map(re.escape, self.sitemaps))
        return re.compile(rf"^{re.escape(self.base)}/(?:{stems})-sitemap\d*\.xml$")


#: Every directory the scraper knows how to walk. ``khoury`` MUST stay first and
#: keep the accordion parser: its records define the existing entity space and
#: its chunk text is pinned byte-for-byte by tests/test_chunk_golden.py.
COLLEGES: tuple[College, ...] = (
    College("khoury", KHOURY_BASE, parser="accordion"),
    College("cos", "https://cos.northeastern.edu"),
    College("damore-mckim", "https://damore-mckim.northeastern.edu", sitemaps=("dmsb_person",)),
    College("camd", "https://camd.northeastern.edu", crawl_delay=10.0, sitemaps=("camd_person",)),
    College("coe", "https://coe.northeastern.edu", sitemaps=("faculty",)),
    College(
        "cssh", "https://cssh.northeastern.edu", crawl_delay=10.0,
        profile_paths=("faculty", "person"), sitemaps=("faculty", "person"),
    ),
    College(
        "bouve", "https://bouve.northeastern.edu", crawl_delay=10.0,
        profile_paths=("directory",), sitemaps=("bouve_person",),
    ),
    College(
        "law", "https://law.northeastern.edu", crawl_delay=10.0,
        profile_paths=("faculty",), sitemaps=("faculty",),
    ),
)

COLLEGES_BY_KEY: dict[str, College] = {c.key: c for c in COLLEGES}

# --- LLM extraction (non-accordion colleges) ------------------------------

EXTRACT_MODEL = "claude-haiku-4-5"
EXTRACT_MAX_TOKENS = 2000
EXTRACT_MAX_RETRIES = 6
EXTRACT_MAX_CLEAN_CHARS = 20_000
EXTRACT_MIN_CLEAN_CHARS = 120  # shorter than this is a nav-only / JS-only page

EXTRACT_PROMPT = (
    "You are reading one faculty profile page from a university directory.\n"
    "Extract only what the page actually states; never infer or invent.\n"
    "Use an empty string or empty list for anything the page does not mention.\n"
    "Do not copy navigation, site chrome, or boilerplate into any field.\n\n"
    "Page text:\n\n"
)

#: Mirrors the fields of ``models.Profile`` that ``ProfileSource`` renders.
#: ``additionalProperties: false`` + ``required`` is what makes the Anthropic
#: structured-output guarantee exact rather than best-effort.
EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name", "title", "biography", "research_interests",
        "education", "areas_of_interest", "labs_and_groups", "projects",
    ],
    "properties": {
        "name": {"type": "string"},
        "title": {"type": "string"},
        "biography": {"type": "string"},
        "research_interests": {"type": "array", "items": {"type": "string"}},
        "education": {"type": "array", "items": {"type": "string"}},
        "areas_of_interest": {"type": "array", "items": {"type": "string"}},
        "labs_and_groups": {"type": "array", "items": {"type": "string"}},
        "projects": {"type": "array", "items": {"type": "string"}},
    },
}

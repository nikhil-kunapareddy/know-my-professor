"""Shared constants for the weblinks crawl/extract job.

This job is the second crawl stage: it follows the personal/lab `website` link
found inside each Khoury profile, extracts clean text, and uses Gemini to turn
that into structured sections stored under gs://$KMP_GCS_BUCKET/weblinks/.
"""

from __future__ import annotations

USER_AGENT = (
    "KhouryFacultyWeblinks/0.1 (personal research project; "
    "contact: kunapareddy.s@northeastern.edu)"
)
REQUEST_DELAY_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 30

# One-hop crawl: from the homepage, follow at most a few same-host links whose
# href or anchor text mentions one of these. Keeps the crawl shallow and cheap.
ONE_HOP_KEYWORDS = [
    "research", "publication", "publications", "bio", "about",
    "project", "projects", "group", "lab", "labs", "people",
    "students", "news", "cv",
]
MAX_PAGES_PER_PROFESSOR = 5  # homepage + up to 4 one-hop pages

# Success guard: an extraction shorter than this is treated as a failed/empty
# crawl (JS-only page, error page, Cloudflare challenge) and is NOT cached.
MIN_CLEAN_TEXT_CHARS = 200
# Token bound: truncate combined clean text before sending to Gemini.
MAX_CLEAN_TEXT_CHARS = 20_000

# Gemini extraction (Google AI Studio free tier).
# gemini-3.1-flash-lite: 500 RPD (covers ~296 faculty-with-sites in one pass),
# 15 RPM, 250K TPM. Still a Gemini model, so response_schema works (unlike Gemma).
GEMINI_MODEL = "models/gemini-3.1-flash-lite"
# 60s / 15 RPM = 4s min spacing to stay under the per-minute cap.
GEMINI_RATE_LIMIT_SLEEP_SECONDS = 4.0
GEMINI_MAX_RETRIES = 6

# Bumping this forces re-extraction even when a page's text is unchanged, so
# prompt/schema changes propagate to already-crawled professors. The stored
# page_hash mixes this in (see extract.page_hash).
SCHEMA_VERSION = "v1"

# Structured-output schema for Gemini (responseSchema). Keys must match the
# section types ingest knows about (ingest/config.py ENRICHMENT_LABELS).
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "website_summary": {"type": "string"},
        "current_projects": {"type": "array", "items": {"type": "string"}},
        "recent_publications": {"type": "array", "items": {"type": "string"}},
        "students_or_lab_members": {"type": "array", "items": {"type": "string"}},
        "recent_news": {"type": "array", "items": {"type": "string"}},
    },
}
SECTION_TYPES = list(EXTRACTION_SCHEMA["properties"].keys())

EXTRACTION_PROMPT = (
    "You are extracting structured facts from a university faculty member's "
    "personal or research-lab website. Use ONLY information present in the text "
    "below — do not infer, guess, or add anything. If a field is not clearly "
    "stated, return an empty string or empty list for it.\n\n"
    "Extract:\n"
    "- website_summary: 2-4 sentence factual summary of the person's research focus.\n"
    "- current_projects: ongoing projects/systems explicitly described.\n"
    "- recent_publications: paper titles (with year if shown).\n"
    "- students_or_lab_members: names of advised students or lab members.\n"
    "- recent_news: dated announcements/news items.\n\n"
    "WEBSITE TEXT:\n"
)

# GCS layout
PROFILE_PREFIX = "profiles/"
WEBLINKS_PREFIX = "weblinks/"

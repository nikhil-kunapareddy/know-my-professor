"""Configuration owned by the weblinks source (crawl + Gemini extraction).

Kept next to the code that uses it instead of in ``shared/config.py``: the crawl
budget, the extraction prompt, and the response schema mean nothing to the
serving side.
"""

from __future__ import annotations

from .source import WEBLINKS_SECTIONS

WEBLINKS_USER_AGENT = (
    "KhouryFacultyWeblinks/0.1 (personal research project; "
    "contact: kunapareddy.s@northeastern.edu)"
)
WEBLINKS_REQUEST_DELAY_SECONDS = 1.0
WEBLINKS_REQUEST_TIMEOUT_SECONDS = 30

# One-hop crawl: from the homepage follow at most a few same-host links whose
# href or anchor text mentions one of these. Keeps the crawl shallow and cheap.
ONE_HOP_KEYWORDS = [
    "research", "publication", "publications", "bio", "about",
    "project", "projects", "group", "lab", "labs", "people",
    "students", "news", "cv",
]
MAX_PAGES_PER_PROFESSOR = 5  # homepage + up to 4 one-hop pages

# Success guard: an extraction shorter than this is a failed/JS-only/error page
# and is NOT cached. Token bound: truncate combined text before the LLM call.
MIN_CLEAN_TEXT_CHARS = 200
MAX_CLEAN_TEXT_CHARS = 20_000

# Gemini extraction (Google AI Studio free tier). gemini-3.1-flash-lite: 500 RPD
# / 15 RPM / 250K TPM. Still a Gemini model, so response_schema works.
GEMINI_MODEL = "models/gemini-3.1-flash-lite"
GEMINI_RATE_LIMIT_SLEEP_SECONDS = 4.0  # 60s / 15 RPM = 4s min spacing
GEMINI_MAX_RETRIES = 6

# Bump to force re-extraction even when a page's text is unchanged (so prompt/
# schema changes propagate). Mixed into the stored page_hash.
SCHEMA_VERSION = "v1"

# Order sections are emitted in the stored record.
SECTION_TYPES: list[str] = [spec.key for spec in WEBLINKS_SECTIONS]

# Structured-output schema for Gemini (responseSchema).
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

# Structural guard: what Gemini is asked to return must be exactly what the
# source knows how to chunk. Previously a comment; now it fails at import.
_schema_keys = set(EXTRACTION_SCHEMA["properties"])
_section_keys = {spec.key for spec in WEBLINKS_SECTIONS}
if _schema_keys != _section_keys:
    raise ValueError(
        "EXTRACTION_SCHEMA keys must match WEBLINKS_SECTIONS; "
        f"schema-only={sorted(_schema_keys - _section_keys)} "
        f"sections-only={sorted(_section_keys - _schema_keys)}"
    )

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

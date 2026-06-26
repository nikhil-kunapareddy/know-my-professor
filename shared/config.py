"""Single source of truth for cross-cutting constants.

Anything that BOTH the serving side (``core``) and the data side
(``preprocessing``) must agree on lives here — most importantly the embedding
model/dimension and the Pinecone index. Keeping them in one module means the
query embedder and the document embedder can't silently drift apart (the old
layout duplicated these in ``api/app.py`` and ``ingest/config.py`` and relied on
a live test to catch divergence).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# --- Embeddings (shared by core.query and preprocessing.ingest) ------------

# Query and document vectors MUST be produced by the same model/dim or
# similarity search is meaningless.
EMBED_MODEL = "mistral-embed-2312"
EMBED_DIM = 1024

# Mistral's endpoint embeds a list of texts per request, so we batch instead of
# one-call-per-text — collapsing hundreds of requests into a handful.
EMBED_BATCH_SIZE = 64
EMBED_RATE_LIMIT_SLEEP_SECONDS = 1.0
EMBED_MAX_RETRIES = 6

# --- Pinecone --------------------------------------------------------------

# 1024-dim index for Mistral vectors. The old 3072-dim "know-my-professor"
# index (Gemini embeddings) is kept intact for rollback only.
PINECONE_DEFAULT_INDEX = "know-my-professor-m1024"
PINECONE_DEFAULT_CLOUD = "aws"
PINECONE_DEFAULT_REGION = "us-east-1"
UPSERT_BATCH_SIZE = 100
FETCH_BATCH_SIZE = 100

# --- Generation (core.llm) -------------------------------------------------

DEFAULT_CHAT_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"
DEFAULT_TOP_K = 8

# --- Section taxonomy (preprocessing.ingest chunking) ----------------------

# Profile accordion sections -> (field, human label). Vector IDs are
# {slug}#{field}; these field names must stay disjoint from ENRICHMENT_LABELS.
SECTION_FIELDS: list[tuple[str, str]] = [
    ("biography", "Biography"),
    ("research_interests", "Research interests"),
    ("education", "Education"),
    ("areas_of_interest", "Areas of interest"),
    ("labs_and_groups", "Labs and groups"),
    ("projects", "Projects"),
]

# Section types produced by the weblinks crawl/extract job. Vector IDs reuse the
# {slug}#{section_type} convention, so these keys must not collide with the
# SECTION_FIELDS field names above.
ENRICHMENT_LABELS: dict[str, str] = {
    "website_summary": "Website summary",
    "current_projects": "Current projects",
    "recent_publications": "Recent publications",
    "students_or_lab_members": "Students and lab members",
    "recent_news": "Recent news",
}

# --- Khoury directory scraper (preprocessing.sources.profiles) -------------

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
LOCAL_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

# --- Weblinks crawl/extract (preprocessing.sources.weblinks) ---------------

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

# Structured-output schema for Gemini (responseSchema). Keys MUST match
# ENRICHMENT_LABELS so ingest can label the produced sections.
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

# --- GCS layout ------------------------------------------------------------

PROFILE_PREFIX = "profiles/"
WEBLINKS_PREFIX = "weblinks/"


def gcs_bucket() -> str | None:
    """The configured GCS bucket name (env KMP_GCS_BUCKET), or None."""
    return os.environ.get("KMP_GCS_BUCKET")

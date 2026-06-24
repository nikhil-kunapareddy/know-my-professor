"""Shared constants for the Pinecone ingest job."""

from __future__ import annotations

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 3072
EMBED_TASK_TYPE = "retrieval_document"

# Free-tier Gemini embed quota is 100 req/min. Sleep ~0.7s between calls to stay safely under.
EMBED_RATE_LIMIT_SLEEP_SECONDS = 0.7
EMBED_MAX_RETRIES = 6

PINECONE_DEFAULT_INDEX = "know-my-professor"
PINECONE_DEFAULT_CLOUD = "aws"
PINECONE_DEFAULT_REGION = "us-east-1"

SECTION_FIELDS: list[tuple[str, str]] = [
    ("biography", "Biography"),
    ("research_interests", "Research interests"),
    ("education", "Education"),
    ("areas_of_interest", "Areas of interest"),
    ("labs_and_groups", "Labs and groups"),
    ("projects", "Projects"),
]

# Section types produced by the weblinks crawl/extract job, stored under
# gs://$KMP_GCS_BUCKET/weblinks/{slug}.json. Mapped to human labels for chunk headers.
# Vector IDs are {slug}#{section_type}, so these must not collide with SECTION_FIELDS.
ENRICHMENT_LABELS: dict[str, str] = {
    "website_summary": "Website summary",
    "current_projects": "Current projects",
    "recent_publications": "Recent publications",
    "students_or_lab_members": "Students and lab members",
    "recent_news": "Recent news",
}

UPSERT_BATCH_SIZE = 100
FETCH_BATCH_SIZE = 100

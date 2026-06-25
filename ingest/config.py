"""Shared constants for the Pinecone ingest job."""

from __future__ import annotations

EMBED_MODEL = "mistral-embed-2312"
EMBED_DIM = 1024

# Mistral embeddings are batched — many texts per request — so the per-day
# request cap stops being the bottleneck (the Gemini path embedded one text per
# request). Up to this many texts go in a single embeddings.create call.
EMBED_BATCH_SIZE = 64
# Brief pause between embedding requests to stay polite under the free-tier RPM.
EMBED_RATE_LIMIT_SLEEP_SECONDS = 1.0
EMBED_MAX_RETRIES = 6

# New 1024-dim index for Mistral vectors; the old 3072-dim "know-my-professor"
# index (Gemini embeddings) is left intact for rollback.
PINECONE_DEFAULT_INDEX = "know-my-professor-m1024"
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

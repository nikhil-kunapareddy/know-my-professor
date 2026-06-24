"""GCS I/O for the weblinks job: read profiles, read/write weblinks records."""

from __future__ import annotations

import json
from typing import Iterable

from google.cloud import storage

from config import PROFILE_PREFIX, WEBLINKS_PREFIX


def load_profiles_from_gcs(bucket_name: str) -> Iterable[dict]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in client.list_blobs(bucket, prefix=PROFILE_PREFIX):
        if blob.name.endswith(".json"):
            yield json.loads(blob.download_as_text())


def load_existing_page_hashes(bucket_name: str) -> dict[str, str]:
    """Map slug -> stored page_hash from prior weblinks/{slug}.json records.

    Lets the job skip the Gemini call when a site's clean text is unchanged.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    hashes: dict[str, str] = {}
    for blob in client.list_blobs(bucket, prefix=WEBLINKS_PREFIX):
        if not blob.name.endswith(".json"):
            continue
        try:
            record = json.loads(blob.download_as_text())
        except ValueError:
            continue
        slug, ph = record.get("slug"), record.get("page_hash")
        if slug and ph:
            hashes[slug] = ph
    return hashes


def write_weblinks_record(bucket_name: str, record: dict) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{WEBLINKS_PREFIX}{record['slug']}.json")
    blob.upload_from_string(
        json.dumps(record, ensure_ascii=False, indent=2),
        content_type="application/json",
    )

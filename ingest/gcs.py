"""Load scraped profile JSONs from GCS."""

from __future__ import annotations

import json
from typing import Iterable

from google.cloud import storage


def _load_json_blobs(bucket_name: str, prefix: str) -> Iterable[dict]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in client.list_blobs(bucket, prefix=prefix):
        if not blob.name.endswith(".json"):
            continue
        yield json.loads(blob.download_as_text())


def load_profiles_from_gcs(bucket_name: str, prefix: str = "profiles/") -> Iterable[dict]:
    yield from _load_json_blobs(bucket_name, prefix)


def load_enrichment_from_gcs(bucket_name: str, prefix: str = "weblinks/") -> Iterable[dict]:
    """Load weblinks/{slug}.json records written by the weblinks crawl/extract job."""
    yield from _load_json_blobs(bucket_name, prefix)

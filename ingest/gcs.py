"""Load scraped profile JSONs from GCS."""

from __future__ import annotations

import json
from typing import Iterable

from google.cloud import storage


def load_profiles_from_gcs(bucket_name: str, prefix: str = "profiles/") -> Iterable[dict]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in client.list_blobs(bucket, prefix=prefix):
        if not blob.name.endswith(".json"):
            continue
        yield json.loads(blob.download_as_text())

"""Storage backends shared across the preprocessing jobs.

``OutputStore`` abstracts local FS vs GCS so the scrape loop is identical for
both. ``GCSStore`` additionally exposes the JSON read/write helpers the ingest
and weblinks jobs need (iterate records under a prefix, read page hashes, write
a record), so all three jobs talk to GCS through one class.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class OutputStore(ABC):
    """Abstraction over local FS and GCS so the scrape loop is identical for both."""

    @abstractmethod
    def write_text(self, key: str, content: str) -> None: ...

    @abstractmethod
    def read_text(self, key: str) -> str | None: ...

    @abstractmethod
    def existing_profile_slugs(self) -> set[str]: ...

    @abstractmethod
    def describe(self) -> str: ...


class LocalStore(OutputStore):
    """Writes under a local directory root. Used for local scraper runs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        (self.root / "profiles").mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def write_text(self, key: str, content: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def read_text(self, key: str) -> str | None:
        path = self._path(key)
        return path.read_text() if path.exists() else None

    def existing_profile_slugs(self) -> set[str]:
        return {p.stem for p in (self.root / "profiles").glob("*.json")}

    def describe(self) -> str:
        return f"local:{self.root}"


class GCSStore(OutputStore):
    """Reads/writes a GCS bucket. Source of truth in production.

    Besides the ``OutputStore`` text interface used by the scraper, it offers
    JSON helpers (``iter_json``, ``write_json``, ``load_page_hashes``) consumed
    by the ingest and weblinks jobs.
    """

    def __init__(self, bucket_name: str, prefix: str = "") -> None:
        from google.cloud import storage

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix.rstrip("/")

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    # --- OutputStore (scraper) ---------------------------------------------

    def write_text(self, key: str, content: str) -> None:
        blob = self.bucket.blob(self._full_key(key))
        blob.upload_from_string(content, content_type="application/json")

    def read_text(self, key: str) -> str | None:
        blob = self.bucket.blob(self._full_key(key))
        if not blob.exists():
            return None
        return blob.download_as_text()

    def existing_profile_slugs(self) -> set[str]:
        prefix = self._full_key("profiles/")
        slugs: set[str] = set()
        for blob in self.client.list_blobs(self.bucket, prefix=prefix):
            name = blob.name[len(prefix):]
            if name.endswith(".json"):
                slugs.add(name[: -len(".json")])
        return slugs

    def describe(self) -> str:
        return f"gs://{self.bucket.name}/{self.prefix}".rstrip("/")

    # --- JSON helpers (ingest / weblinks) ----------------------------------

    def iter_json(self, prefix: str) -> Iterable[dict]:
        """Yield each JSON blob under ``prefix`` parsed into a dict."""
        for blob in self.client.list_blobs(self.bucket, prefix=self._full_key(prefix)):
            if blob.name.endswith(".json"):
                yield json.loads(blob.download_as_text())

    def write_json(self, key: str, obj: dict) -> None:
        """Write ``obj`` as pretty UTF-8 JSON at ``key``."""
        blob = self.bucket.blob(self._full_key(key))
        blob.upload_from_string(
            json.dumps(obj, ensure_ascii=False, indent=2),
            content_type="application/json",
        )

    def load_page_hashes(self, prefix: str) -> dict[str, str]:
        """Map slug -> stored page_hash from prior weblinks records under ``prefix``.

        Lets the weblinks job skip the Gemini call when a site's text is unchanged.
        """
        hashes: dict[str, str] = {}
        for record in self.iter_json(prefix):
            slug, ph = record.get("slug"), record.get("page_hash")
            if slug and ph:
                hashes[slug] = ph
        return hashes

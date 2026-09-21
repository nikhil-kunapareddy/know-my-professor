"""Storage backends shared across the preprocessing jobs.

``OutputStore`` abstracts local FS vs GCS so the scrape loop is identical for
both, including the JSON helpers (iterate records under a prefix, read stored
hashes). ``load_hashes`` lives on the base class rather than only on
``GCSStore`` because a scraper that skips unchanged records must behave the same
locally as in the cloud: while only GCS implemented it, every local run rewrote
everything and the skip logic could not be exercised without a bucket.

Every method takes the prefix it operates on. The store deliberately knows
nothing about which sources exist — that lives in
``preprocessing/sources/registry.py`` — so a new corpus needs no change here.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path


class OutputStore(ABC):
    """Abstraction over local FS and GCS so the scrape loop is identical for both."""

    @abstractmethod
    def write_text(self, key: str, content: str) -> None: ...

    @abstractmethod
    def read_text(self, key: str) -> str | None: ...

    @abstractmethod
    def existing_slugs(self, prefix: str) -> set[str]: ...

    @abstractmethod
    def describe(self) -> str: ...

    @abstractmethod
    def iter_json(self, prefix: str) -> Iterable[dict]: ...

    def load_hashes(self, prefix: str, field: str = "page_hash") -> dict[str, str]:
        """Map slug -> stored hash from prior records under ``prefix``.

        Lets a source skip its expensive step -- a Gemini extract call, or simply
        rewriting a catalog entry that has not changed -- when the upstream
        content is the same as last run. Defined once here because it is pure
        logic over ``iter_json``; only the iteration differs per backend.
        """
        hashes: dict[str, str] = {}
        for record in self.iter_json(prefix):
            slug, value = record.get("slug"), record.get(field)
            if slug and value:
                hashes[slug] = value
        return hashes


class LocalStore(OutputStore):
    """Writes under a local directory root. Used for local scraper runs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def write_text(self, key: str, content: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def read_text(self, key: str) -> str | None:
        path = self._path(key)
        return path.read_text() if path.exists() else None

    def existing_slugs(self, prefix: str) -> set[str]:
        return {p.stem for p in (self.root / prefix).glob("*.json")}

    def iter_json(self, prefix: str) -> Iterable[dict]:
        """Yield each JSON file under ``prefix`` parsed into a dict."""
        for path in sorted((self.root / prefix).glob("*.json")):
            yield json.loads(path.read_text())

    def describe(self) -> str:
        return f"local:{self.root}"


class GCSStore(OutputStore):
    """Reads/writes a GCS bucket. Source of truth in production.

    Besides the ``OutputStore`` text interface used by the scraper, it offers
    JSON helpers (``iter_json``, ``write_json``) consumed by the ingest and
    weblinks jobs; ``load_hashes`` is inherited.
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

    def existing_slugs(self, prefix: str) -> set[str]:
        full = self._full_key(prefix)
        slugs: set[str] = set()
        for blob in self.client.list_blobs(self.bucket, prefix=full):
            name = blob.name[len(full):]
            if name.endswith(".json"):
                slugs.add(name[: -len(".json")])
        return slugs

    def describe(self) -> str:
        return f"gs://{self.bucket.name}/{self.prefix}".rstrip("/")

    # --- JSON helpers (ingest / enrichment sources) -------------------------

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


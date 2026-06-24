"""Output stores: an abstraction over local FS and GCS.

Keeps the scrape loop identical whether output goes to ./data or gs://bucket/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


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
    def __init__(self, bucket_name: str, prefix: str = "") -> None:
        from google.cloud import storage

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix.rstrip("/")

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

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
                slugs.add(name[:-len(".json")])
        return slugs

    def describe(self) -> str:
        return f"gs://{self.bucket.name}/{self.prefix}".rstrip("/")

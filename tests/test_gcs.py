"""GCSStore behaviour that needs no bucket."""

from __future__ import annotations

from shared.gcs import GCSStore


class _Blob:
    def __init__(self, calls):
        self.calls = calls

    def upload_from_string(self, content, **kwargs):
        self.calls.append(kwargs)


class _Bucket:
    def __init__(self):
        self.calls = []

    def blob(self, key):
        return _Blob(self.calls)


def test_every_upload_is_retried():
    """Uploads are not retried by default without a generation precondition."""
    store = object.__new__(GCSStore)
    store.bucket, store.prefix, store._upload_retry = _Bucket(), "", object()

    store.write_text("grants_summaries.json", "{}")
    store.write_json("grants/jane-doe.json", {})

    assert [c["retry"] for c in store.bucket.calls] == [store._upload_retry] * 2

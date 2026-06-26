"""Unit tests for the ingest chunking + Pinecone hash-skip logic."""

from __future__ import annotations

import pytest

from preprocessing.ingest.chunking import Chunker
from preprocessing.ingest.pinecone_store import PineconeStore


@pytest.fixture
def chunker():
    return Chunker()


def test_content_hash_is_deterministic_and_text_sensitive(chunker):
    assert chunker.content_hash("hello") == chunker.content_hash("hello")
    assert chunker.content_hash("hello") != chunker.content_hash("hellp")
    assert chunker.content_hash("x").startswith("sha256:")


def test_is_substantive(chunker):
    assert chunker.is_substantive({"biography": "x"})
    assert chunker.is_substantive({"areas_of_interest": ["PL"]})
    assert not chunker.is_substantive({"slug": "x", "name": "n"})


def test_profile_chunks_have_ids_and_content_hash(chunker):
    profile = {
        "slug": "jane-doe",
        "name": "Jane Doe",
        "title": "Prof",
        "biography": "Works on PL.",
        "areas_of_interest": ["PL"],
    }
    chunks = chunker.profile_to_chunks(profile)
    ids = {c.vector_id for c in chunks}
    assert ids == {"jane-doe#biography", "jane-doe#areas_of_interest"}
    assert all("content_hash" in c.metadata for c in chunks)
    for c in chunks:
        assert c.metadata["content_hash"] == chunker.content_hash(c.text)


def test_profile_with_no_slug_yields_nothing(chunker):
    assert chunker.profile_to_chunks({"biography": "x"}) == []


def test_enrichment_chunks_use_source_url_and_disjoint_ids(chunker):
    profile = {"slug": "jane-doe", "name": "Jane Doe", "title": "Prof", "biography": "Bio."}
    enriched = {
        "slug": "jane-doe",
        "professor_name": "Jane Doe",
        "professor_title": "Prof",
        "sections": [
            {"section_type": "current_projects", "text": ["Proj A", "Proj B"],
             "source_url": "https://jane.example/"},
            {"section_type": "website_summary", "text": "Studies types.",
             "source_url": "https://jane.example/"},
        ],
    }
    ec = chunker.enrichment_to_chunks(enriched)
    by_id = {c.vector_id: c for c in ec}
    assert set(by_id) == {"jane-doe#current_projects", "jane-doe#website_summary"}
    assert all(c.metadata["url"] == "https://jane.example/" for c in ec)
    assert all("content_hash" in c.metadata for c in ec)
    profile_ids = {c.vector_id for c in chunker.profile_to_chunks(profile)}
    assert profile_ids.isdisjoint(set(by_id))


def test_enrichment_drops_empty_sections_and_bad_input(chunker):
    enriched = {
        "slug": "x", "professor_name": "X",
        "sections": [
            {"section_type": "recent_news", "text": [], "source_url": "u"},
            {"section_type": "website_summary", "text": "", "source_url": "u"},
            {"section_type": "current_projects", "text": ["P"], "source_url": "u"},
        ],
    }
    ec = chunker.enrichment_to_chunks(enriched)
    assert [c.vector_id for c in ec] == ["x#current_projects"]
    assert chunker.enrichment_to_chunks({"sections": []}) == []  # no slug


# --- PineconeStore.fetch_existing_hashes ----------------------------------


class _Vec:
    def __init__(self, content_hash):
        self.metadata = {"content_hash": content_hash}


class _Resp:
    def __init__(self, vectors):
        self.vectors = vectors


class _FakeIndex:
    """Minimal stand-in for a Pinecone index: returns stored hashes, records calls."""

    def __init__(self, store):
        self.store = store
        self.fetch_calls = []

    def fetch(self, ids):
        self.fetch_calls.append(list(ids))
        return _Resp({i: _Vec(self.store[i]) for i in ids if i in self.store})


def test_fetch_existing_hashes_maps_present_ids_only():
    index = _FakeIndex({"a#bio": "sha256:1", "b#bio": "sha256:2"})
    result = PineconeStore(index).fetch_existing_hashes(["a#bio", "b#bio", "c#bio"])
    assert result == {"a#bio": "sha256:1", "b#bio": "sha256:2"}  # c#bio absent


def test_fetch_existing_hashes_dedupes_and_batches():
    from shared.config import FETCH_BATCH_SIZE

    ids = [f"p{i}#bio" for i in range(150)] + ["p0#bio"]  # 150 unique + 1 dup
    store = {i: "sha256:h" for i in ids}
    index = _FakeIndex(store)
    result = PineconeStore(index).fetch_existing_hashes(ids)

    assert len(result) == 150
    assert [len(c) for c in index.fetch_calls] == [FETCH_BATCH_SIZE, 50]


def test_changed_hash_marks_chunk_pending(chunker):
    """The core 'no wasteful re-embed' rule: same hash skips, changed hash re-embeds."""
    chunks = chunker.profile_to_chunks({"slug": "s", "name": "S", "biography": "v1"})
    index = _FakeIndex({c.vector_id: c.metadata["content_hash"] for c in chunks})
    store = PineconeStore(index)

    existing = store.fetch_existing_hashes([c.vector_id for c in chunks])
    pending = [c for c in chunks if existing.get(c.vector_id) != c.metadata["content_hash"]]
    assert pending == []  # unchanged -> nothing to embed

    changed = chunker.profile_to_chunks({"slug": "s", "name": "S", "biography": "v2"})
    pending = [c for c in changed if existing.get(c.vector_id) != c.metadata["content_hash"]]
    assert {c.vector_id for c in pending} == {"s#biography"}  # only the changed section

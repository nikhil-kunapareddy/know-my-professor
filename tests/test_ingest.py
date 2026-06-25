"""Unit tests for the ingest chunking + Pinecone hash-skip logic."""

from __future__ import annotations

import pytest


@pytest.fixture
def chunking(load):
    return load("ingest", "chunking")


@pytest.fixture
def pinecone_store(load):
    return load("ingest", "pinecone_store")


def test_content_hash_is_deterministic_and_text_sensitive(chunking):
    assert chunking.content_hash("hello") == chunking.content_hash("hello")
    assert chunking.content_hash("hello") != chunking.content_hash("hellp")
    assert chunking.content_hash("x").startswith("sha256:")


def test_is_substantive(chunking):
    assert chunking.is_substantive({"biography": "x"})
    assert chunking.is_substantive({"areas_of_interest": ["PL"]})
    assert not chunking.is_substantive({"slug": "x", "name": "n"})


def test_profile_chunks_have_ids_and_content_hash(chunking):
    profile = {
        "slug": "jane-doe",
        "name": "Jane Doe",
        "title": "Prof",
        "biography": "Works on PL.",
        "areas_of_interest": ["PL"],
    }
    chunks = chunking.profile_to_chunks(profile)
    ids = {c.vector_id for c in chunks}
    assert ids == {"jane-doe#biography", "jane-doe#areas_of_interest"}
    assert all("content_hash" in c.metadata for c in chunks)
    # content_hash matches the rendered text
    for c in chunks:
        assert c.metadata["content_hash"] == chunking.content_hash(c.text)


def test_profile_with_no_slug_yields_nothing(chunking):
    assert chunking.profile_to_chunks({"biography": "x"}) == []


def test_enrichment_chunks_use_source_url_and_disjoint_ids(chunking):
    profile = {"slug": "jane-doe", "name": "Jane Doe", "title": "Prof",
               "biography": "Bio."}
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
    ec = chunking.enrichment_to_chunks(enriched)
    by_id = {c.vector_id: c for c in ec}
    assert set(by_id) == {"jane-doe#current_projects", "jane-doe#website_summary"}
    # citation url points at the real website, not the profile page
    assert all(c.metadata["url"] == "https://jane.example/" for c in ec)
    assert all("content_hash" in c.metadata for c in ec)
    # no collision with the base profile section ids
    profile_ids = {c.vector_id for c in chunking.profile_to_chunks(profile)}
    assert profile_ids.isdisjoint(set(by_id))


def test_enrichment_drops_empty_sections_and_bad_input(chunking):
    enriched = {
        "slug": "x", "professor_name": "X",
        "sections": [
            {"section_type": "recent_news", "text": [], "source_url": "u"},
            {"section_type": "website_summary", "text": "", "source_url": "u"},
            {"section_type": "current_projects", "text": ["P"], "source_url": "u"},
        ],
    }
    ec = chunking.enrichment_to_chunks(enriched)
    assert [c.vector_id for c in ec] == ["x#current_projects"]
    assert chunking.enrichment_to_chunks({"sections": []}) == []  # no slug


# --- pinecone_store.fetch_existing_hashes ---------------------------------


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


def test_fetch_existing_hashes_maps_present_ids_only(pinecone_store):
    index = _FakeIndex({"a#bio": "sha256:1", "b#bio": "sha256:2"})
    result = pinecone_store.fetch_existing_hashes(index, ["a#bio", "b#bio", "c#bio"])
    assert result == {"a#bio": "sha256:1", "b#bio": "sha256:2"}  # c#bio absent


def test_fetch_existing_hashes_dedupes_and_batches(pinecone_store):
    from config import FETCH_BATCH_SIZE  # ingest/config is on path during this load

    ids = [f"p{i}#bio" for i in range(150)] + ["p0#bio"]  # 150 unique + 1 dup
    store = {i: "sha256:h" for i in ids}
    index = _FakeIndex(store)
    result = pinecone_store.fetch_existing_hashes(index, ids)

    assert len(result) == 150
    # batched by FETCH_BATCH_SIZE, duplicates collapsed (150 unique -> 100 + 50)
    assert [len(c) for c in index.fetch_calls] == [FETCH_BATCH_SIZE, 50]


def test_changed_hash_marks_chunk_pending(chunking, pinecone_store):
    """The core 'no wasteful re-embed' rule: same hash skips, changed hash re-embeds."""
    chunks = chunking.profile_to_chunks(
        {"slug": "s", "name": "S", "biography": "v1"}
    )
    index = _FakeIndex({c.vector_id: c.metadata["content_hash"] for c in chunks})

    existing = pinecone_store.fetch_existing_hashes(index, [c.vector_id for c in chunks])
    pending = [c for c in chunks if existing.get(c.vector_id) != c.metadata["content_hash"]]
    assert pending == []  # unchanged -> nothing to embed

    changed = chunking.profile_to_chunks({"slug": "s", "name": "S", "biography": "v2"})
    pending = [c for c in changed if existing.get(c.vector_id) != c.metadata["content_hash"]]
    assert {c.vector_id for c in pending} == {"s#biography"}  # only the changed section

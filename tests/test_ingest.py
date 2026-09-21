"""Unit tests for source rendering, the source registry, and ingest orchestration."""

from __future__ import annotations

import pytest

from preprocessing.ingest.pinecone_store import PineconeStore
from preprocessing.ingest.runner import _collect_chunks
from preprocessing.sources.base import Source, content_hash
from preprocessing.sources.profiles.source import ProfileSource
from preprocessing.sources.weblinks.source import WeblinksSource


def _ids(store, limit=None):
    """Vector ids across every namespace _collect_chunks grouped them into."""
    return {c.vector_id for chunks in _collect_chunks(store, limit).values() for c in chunks}


@pytest.fixture
def profiles():
    return ProfileSource()


@pytest.fixture
def weblinks():
    return WeblinksSource()


# --- rendering helpers -----------------------------------------------------


def test_content_hash_is_deterministic_and_text_sensitive():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("hellp")
    assert content_hash("x").startswith("sha256:")


# --- profiles source -------------------------------------------------------


def test_is_ingestable(profiles):
    assert profiles.is_ingestable({"biography": "x"})
    assert profiles.is_ingestable({"areas_of_interest": ["PL"]})
    assert not profiles.is_ingestable({"slug": "x", "name": "n"})


def test_profile_chunks_have_ids_and_content_hash(profiles):
    profile = {
        "slug": "jane-doe",
        "name": "Jane Doe",
        "title": "Prof",
        "biography": "Works on PL.",
        "areas_of_interest": ["PL"],
    }
    chunks = profiles.to_chunks(profile)
    assert {c.vector_id for c in chunks} == {"jane-doe#biography", "jane-doe#areas_of_interest"}
    for c in chunks:
        assert c.metadata["content_hash"] == content_hash(c.text)


def test_profile_with_no_slug_yields_nothing(profiles):
    assert profiles.to_chunks({"biography": "x"}) == []


# --- weblinks source -------------------------------------------------------


def test_enrichment_chunks_use_source_url_and_disjoint_ids(profiles, weblinks):
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
    ec = weblinks.to_chunks(enriched)
    ids = {c.vector_id for c in ec}
    assert ids == {"jane-doe#current_projects", "jane-doe#website_summary"}
    assert all(c.metadata["url"] == "https://jane.example/" for c in ec)
    assert {c.vector_id for c in profiles.to_chunks(profile)}.isdisjoint(ids)


def test_enrichment_drops_empty_sections_and_bad_input(weblinks):
    enriched = {
        "slug": "x", "professor_name": "X",
        "sections": [
            {"section_type": "recent_news", "text": [], "source_url": "u"},
            {"section_type": "website_summary", "text": "", "source_url": "u"},
            {"section_type": "current_projects", "text": ["P"], "source_url": "u"},
        ],
    }
    assert [c.vector_id for c in weblinks.to_chunks(enriched)] == ["x#current_projects"]
    assert weblinks.to_chunks({"sections": []}) == []  # no slug


# --- registry invariants ---------------------------------------------------


def test_registry_rejects_colliding_section_keys():
    """Two sources sharing a section key would silently overwrite in Pinecone."""
    from preprocessing.sources.base import SectionSpec
    from preprocessing.sources.registry import _validate

    class _Dup(Source):
        name = "dup"
        prefix = "dup/"
        sections = (SectionSpec("biography", "Biography"),)

        def to_chunks(self, record):
            return []

    with pytest.raises(ValueError, match="duplicate section key"):
        _validate((ProfileSource(), _Dup()))


def test_registry_rejects_duplicate_prefixes():
    from preprocessing.sources.registry import _validate

    class _Shadow(Source):
        name = "shadow"
        prefix = "profiles/"

        def to_chunks(self, record):
            return []

    with pytest.raises(ValueError, match="duplicate GCS prefix"):
        _validate((ProfileSource(), _Shadow()))


def test_registry_requires_an_entity_defining_source():
    from preprocessing.sources.registry import _validate

    with pytest.raises(ValueError, match="must define entities"):
        _validate((WeblinksSource(),))


def test_live_registry_is_valid():
    from preprocessing.sources.registry import SOURCES, _validate, all_section_types

    _validate(SOURCES)
    assert len(set(all_section_types())) == len(all_section_types())


# --- ingest orchestration over the registry --------------------------------


class _FakeStore:
    """Serves canned records per prefix, like GCSStore.iter_json would."""

    def __init__(self, by_prefix):
        self.by_prefix = by_prefix

    def iter_json(self, prefix):
        return iter(self.by_prefix.get(prefix, []))


def _store_with(profiles_records, weblinks_records):
    return _FakeStore({
        ProfileSource.prefix: profiles_records,
        WeblinksSource.prefix: weblinks_records,
    })


def test_collect_chunks_skips_enrichment_for_unknown_entities():
    """A stale weblinks record whose profile is gone must not be ingested."""
    store = _store_with(
        [{"slug": "a", "name": "A", "biography": "bio a"}],
        [
            {"slug": "a", "professor_name": "A",
             "sections": [{"section_type": "website_summary", "text": "s", "source_url": "u"}]},
            {"slug": "ghost", "professor_name": "G",
             "sections": [{"section_type": "website_summary", "text": "s", "source_url": "u"}]},
        ],
    )
    ids = _ids(store)
    assert ids == {"a#biography", "a#website_summary"}


def test_collect_chunks_limit_scopes_enrichment_to_the_slice():
    """--limit slices entities; enrichment follows that slice, not its own."""
    store = _store_with(
        [
            {"slug": "a", "name": "A", "biography": "bio a"},
            {"slug": "b", "name": "B", "biography": "bio b"},
        ],
        [
            {"slug": "b", "professor_name": "B",
             "sections": [{"section_type": "website_summary", "text": "s", "source_url": "u"}]},
        ],
    )
    assert _ids(store, limit=1) == {"a#biography"}
    assert _ids(store, limit=2) == {
        "a#biography", "b#biography", "b#website_summary",
    }


def test_collect_chunks_keeps_enrichment_for_thin_profiles():
    """A profile too thin to ingest still anchors its enrichment."""
    store = _store_with(
        [{"slug": "thin", "name": "T"}],  # not ingestable: no bio/research/areas
        [{"slug": "thin", "professor_name": "T",
          "sections": [{"section_type": "website_summary", "text": "s", "source_url": "u"}]}],
    )
    assert _ids(store) == {"thin#website_summary"}


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


def test_changed_hash_marks_chunk_pending(profiles):
    """The core 'no wasteful re-embed' rule: same hash skips, changed hash re-embeds."""
    chunks = profiles.to_chunks({"slug": "s", "name": "S", "biography": "v1"})
    index = _FakeIndex({c.vector_id: c.metadata["content_hash"] for c in chunks})
    store = PineconeStore(index)

    existing = store.fetch_existing_hashes([c.vector_id for c in chunks])
    pending = [c for c in chunks if existing.get(c.vector_id) != c.metadata["content_hash"]]
    assert pending == []  # unchanged -> nothing to embed

    changed = profiles.to_chunks({"slug": "s", "name": "S", "biography": "v2"})
    pending = [c for c in changed if existing.get(c.vector_id) != c.metadata["content_hash"]]
    assert {c.vector_id for c in pending} == {"s#biography"}  # only the changed section

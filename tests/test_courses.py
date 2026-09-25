"""Course corpus: catalog parsing, Banner, entity ids, and namespace routing.

The namespace tests carry the most weight here. A namespace is a hard partition
— a Pinecone query reads exactly one — so getting it wrong does not raise, it
just silently returns nothing (on the read side) or writes into the wrong
partition (on the write side). Both look like "retrieval got worse".
"""

from __future__ import annotations

import pytest

from preprocessing.sources.courses.catalog import CatalogFetcher
from preprocessing.sources.courses.source import COURSES_NAMESPACE, CourseSource
from preprocessing.sources.schedule.banner import BannerClient
from preprocessing.sources.schedule.runner import course_slug
from preprocessing.sources.schedule.source import ScheduleSource
from shared.config import PEOPLE_NAMESPACE, RESEARCH_NAMESPACE

SUBJECT_HTML = """
<html><body>
  <div class="courseblock">
    <p class="courseblocktitle noindent">CS 3800.  Theory of Computation.  (4 Hours)</p>
    <p class="cb_desc">Covers automata theory, computability, and complexity.</p>
    <p class="courseblockextra noindent">Prerequisite(s): CS&#160;1800 with a minimum grade of C-</p>
    <p class="courseblockextra noindent">Attribute(s): NUpath Formal/Quant Reasoning</p>
  </div>
  <div class="courseblock">
    <p class="courseblocktitle noindent">CS 2001.  Lab for CS 2000.  (1 Hour)</p>
    <p class="cb_desc">Accompanies CS 2000.</p>
  </div>
  <div class="courseblock">
    <p class="cb_desc">A block with no title line at all.</p>
  </div>
</body></html>
"""


# --- catalog parsing -------------------------------------------------------


def test_parse_courses_reads_title_credits_description_and_requisites():
    records = CatalogFetcher.parse_courses(SUBJECT_HTML, "cs", "https://example.edu/cs/")
    assert [r["code"] for r in records] == ["CS 3800", "CS 2001"]

    first = records[0]
    assert first["slug"] == "cs3800"
    assert first["title"] == "Theory of Computation"
    assert first["credits"] == "4 Hours"
    assert first["description"].startswith("Covers automata theory")
    assert len(first["requisites"]) == 2


def test_parse_courses_normalises_non_breaking_spaces():
    """`CS\\xa01800` reaches content_hash, so an invisible char must not vary it."""
    requisite = CatalogFetcher.parse_courses(SUBJECT_HTML, "cs")[0]["requisites"][0]
    assert "\xa0" not in requisite
    assert "CS 1800" in requisite


def test_parse_courses_skips_a_block_with_no_readable_title():
    assert len(CatalogFetcher.parse_courses(SUBJECT_HTML, "cs")) == 2


def test_parse_courses_tolerates_a_missing_credits_hint():
    html = '<div class="courseblock"><p class="courseblocktitle">ENGW 1111.  First-Year Writing.</p>' \
           '<p class="cb_desc">Writing.</p></div>'
    record = CatalogFetcher.parse_courses(html, "engw")[0]
    assert record["code"] == "ENGW 1111"
    assert record["credits"] == ""


# --- entity ids ------------------------------------------------------------


def test_course_entity_ids_are_prefixed_so_they_cannot_collide_with_people():
    assert CourseSource().entity_id({"slug": "cs3800"}) == "course-cs3800"
    assert CourseSource().entity_id({}) is None


def test_schedule_mints_the_same_entity_id_as_courses():
    """Enrichment must land on the course it describes, as weblinks do for people."""
    record = {"slug": "cs3800"}
    assert ScheduleSource().entity_id(record) == CourseSource().entity_id(record)


def test_course_slug_matches_banner_and_catalog_spellings():
    """Banner writes `CS3800`, the catalog writes `CS 3800`; ids must agree."""
    assert course_slug("CS3800") == "cs3800"
    assert course_slug("CS 3800") == "cs3800"
    assert course_slug("ENGW1111") == "engw1111"
    assert course_slug("not a course") is None


# --- namespace -------------------------------------------------------------


def test_both_course_sources_share_one_namespace():
    """A course's description and its instructors must be retrievable together."""
    assert CourseSource().namespace == ScheduleSource().namespace == COURSES_NAMESPACE


def test_people_sources_share_the_people_namespace():
    """Weblinks enrich a profile, so both must land in the same partition.

    Split across namespaces, a professor's website chunks could never be
    retrieved alongside their biography — and nothing would error.
    """
    from preprocessing.sources.profiles.source import ProfileSource
    from preprocessing.sources.weblinks.source import WeblinksSource

    assert ProfileSource().namespace == WeblinksSource().namespace == PEOPLE_NAMESPACE


def test_every_registered_source_names_a_namespace():
    """``None`` is the unnamed default partition and means "someone forgot"."""
    from preprocessing.sources.registry import SOURCES

    assert [s.name for s in SOURCES if not s.namespace] == []


def test_the_api_searches_every_namespace_ingest_writes():
    """The one pairing that fails silently: empty results, never an error.

    Every registered source's namespace must be searchable, or that corpus is
    ingested, paid for, and unreachable — which is exactly what happened to the
    10,948 course vectors between landing them and wiring CHAT_NAMESPACES.
    """
    from preprocessing.sources.registry import SOURCES
    from shared.settings import ApiSettings

    assert {s.namespace for s in SOURCES} == set(ApiSettings().namespaces)


def test_registry_rejects_a_dependent_source_alone_in_its_namespace():
    """Its chunks could never match an entity — and would fail silently."""
    from preprocessing.sources.base import Source
    from preprocessing.sources.registry import _validate

    class Orphan(Source):
        name, prefix, sections = "orphan", "orphan/", ()
        depends_on_entities = True
        namespace = "nowhere"

        def to_chunks(self, record):  # pragma: no cover - never called
            return []

    class Anchor(Source):
        name, prefix, sections = "anchor", "anchor/", ()
        depends_on_entities = False
        namespace = None

        def to_chunks(self, record):  # pragma: no cover - never called
            return []

    with pytest.raises(ValueError, match="namespace with no entity-defining"):
        _validate((Anchor(), Orphan()))


def test_collect_chunks_groups_by_namespace():
    """Ids are unique only within a namespace, so the hash check must be scoped."""
    from preprocessing.ingest.runner import _collect_chunks

    class _Store:
        def iter_json(self, prefix):
            data = {
                "profiles/": [{"slug": "a", "name": "A", "biography": "Studies compilers."}],
                "courses/": [{"slug": "cs3800", "code": "CS 3800", "title": "Theory",
                              "description": "Covers automata theory and complexity."}],
            }
            return iter(data.get(prefix, []))

    grouped = _collect_chunks(_Store(), limit=None)
    assert set(grouped) == {PEOPLE_NAMESPACE, COURSES_NAMESPACE}
    assert [c.vector_id for c in grouped[PEOPLE_NAMESPACE]] == ["a#biography"]
    assert [c.vector_id for c in grouped[COURSES_NAMESPACE]] == ["course-cs3800#course_description"]


def test_research_sources_write_apart_but_enrich_people():
    """Publications and grants get their own slots, yet still join to profiles."""
    from preprocessing.sources.grants.source import GrantsSource
    from preprocessing.sources.publications.source import PublicationsSource

    for source in (PublicationsSource(), GrantsSource()):
        assert source.namespace == RESEARCH_NAMESPACE
        assert source.entity_scope() == PEOPLE_NAMESPACE


def test_registry_accepts_a_dependent_source_whose_entities_live_elsewhere():
    from preprocessing.sources.base import Source
    from preprocessing.sources.registry import _validate

    class Enrichment(Source):
        name, prefix, sections = "enrichment", "enrichment/", ()
        depends_on_entities = True
        namespace = "apart"
        entity_namespace = "home"

        def to_chunks(self, record):  # pragma: no cover - never called
            return []

    class Anchor(Source):
        name, prefix, sections = "anchor", "anchor/", ()
        depends_on_entities = False
        namespace = "home"

        def to_chunks(self, record):  # pragma: no cover - never called
            return []

    _validate((Anchor(), Enrichment()))


def test_collect_chunks_joins_research_to_people_ids_only():
    """A research record lands in research, and only if its entity is a person.

    The join is per namespace: an id that exists only among the courses must not
    admit it, because ids are unique only within one namespace.
    """
    from preprocessing.ingest.runner import _collect_chunks

    award = {"agency": "NSF", "award_id": "1", "title": "Compilers", "status": "active"}

    class _Store:
        def iter_json(self, prefix):
            data = {
                "profiles/": [{"slug": "a", "name": "A", "biography": "Studies compilers."}],
                "courses/": [{"slug": "cs3800", "code": "CS 3800", "title": "Theory",
                              "description": "Covers automata theory and complexity."}],
                "grants/": [
                    {"slug": "a", "professor_name": "A", "awards": [award]},
                    # Only a course has this id; it is no professor.
                    {"slug": "course-cs3800", "professor_name": "X", "awards": [award]},
                ],
            }
            return iter(data.get(prefix, []))

    grouped = _collect_chunks(_Store(), limit=None)
    assert [c.vector_id for c in grouped[RESEARCH_NAMESPACE]] == ["a#research_funding"]
    assert [c.vector_id for c in grouped[PEOPLE_NAMESPACE]] == ["a#biography"]


def test_pipeline_passes_the_namespace_to_the_retriever():
    from core.pipeline import RAGPipeline

    seen: dict = {}

    class _Retriever:
        def retrieve(self, query_embedding, top_k, filters=None, namespace=None):
            seen.setdefault("namespaces", []).append(namespace)
            return []

    class _Embedder:
        def embed_query(self, q):
            return [0.0]

    pipeline = RAGPipeline(
        embedder=_Embedder(), retriever=_Retriever(), generator=None, top_k=8, min_score=0.0,
        namespaces=(PEOPLE_NAMESPACE, COURSES_NAMESPACE),
    )

    # An explicit namespace restricts the search to exactly that one, which is
    # how the eval harness scores a case against the corpus that owns it.
    pipeline.answer("what does CS 3800 cover?", namespace=COURSES_NAMESPACE)
    assert seen["namespaces"] == [COURSES_NAMESPACE]

    # Omitting it searches every namespace the pipeline was built with, rather
    # than silently querying the unnamed default partition.
    seen.clear()
    pipeline.answer("who works on compilers?")
    assert seen["namespaces"] == [PEOPLE_NAMESPACE, COURSES_NAMESPACE]


# --- ingestable guards -----------------------------------------------------


def test_a_course_with_no_description_is_not_ingested():
    """Placeholder catalog entries would embed a vector matching only its title."""
    source = CourseSource()
    assert source.is_ingestable({"slug": "x", "description": "Covers things."})
    assert not source.is_ingestable({"slug": "x", "description": "   "})
    assert source.to_chunks({"slug": "x", "code": "X 1", "description": ""}) == []


def test_a_course_staffed_entirely_TBA_is_not_ingested():
    source = ScheduleSource()
    assert not source.is_ingestable({"slug": "x", "terms": [{"instructors": []}]})
    assert source.is_ingestable({"slug": "x", "terms": [{"instructors": ["Jane Doe"]}]})


# --- rendering -------------------------------------------------------------


def test_course_chunk_leads_with_the_code():
    """People refer to courses by code; leading with it retrieves better."""
    record = {"slug": "cs3800", "code": "CS 3800", "title": "Theory of Computation",
              "credits": "4 Hours", "description": "Covers automata theory.",
              "requisites": ["Prerequisite(s): CS 1800"]}
    chunk = CourseSource().to_chunks(record)[0]
    assert chunk.text.startswith("CS 3800 — Theory of Computation (4 Hours)")
    # requisites ride along in the description chunk rather than their own vector
    assert "Prerequisite(s): CS 1800" in chunk.text
    assert chunk.metadata["course_code"] == "CS 3800"


def test_instructor_chunk_names_the_term():
    """"Who teaches X" and "who taught X last term" differ only by the term."""
    record = {
        "slug": "cs3800", "code": "CS 3800", "title": "Theory of Computation",
        "terms": [
            {"code": "202710", "description": "Fall 2026 Semester", "instructors": ["Jane Doe"]},
            {"code": "202730", "description": "Spring 2027 Semester", "instructors": ["John Roe"]},
        ],
    }
    chunk = ScheduleSource().to_chunks(record)[0]
    assert "Fall 2026 Semester: Jane Doe" in chunk.text
    assert "Spring 2027 Semester: John Roe" in chunk.text
    assert chunk.metadata["instructors"] == ["Jane Doe", "John Roe"]


def test_instructor_chunk_drops_a_term_with_no_named_instructor():
    record = {
        "slug": "cs3800", "code": "CS 3800", "title": "Theory",
        "terms": [
            {"description": "Fall 2026 Semester", "instructors": ["Jane Doe"]},
            {"description": "Spring 2027 Semester", "instructors": []},
        ],
    }
    text = ScheduleSource().to_chunks(record)[0].text
    assert "Fall 2026" in text and "Spring 2027" not in text


# --- Banner ----------------------------------------------------------------


def test_active_terms_excludes_completed_ones():
    """`(View Only)` is Banner's own finished flag — the only one it exposes.

    Filtering on it means no hardcoded term list: a new term appears the day
    Banner publishes it, and drops out once it completes.
    """
    assert BannerClient.is_active({"description": "Fall 2026 Semester"})
    assert not BannerClient.is_active({"description": "Summer 2026 Semester (View Only)"})
    assert BannerClient.is_active({})


def test_display_name_flips_to_natural_order():
    """Banner writes surname-first; the professor corpus stores natural order."""
    assert BannerClient.display_to_name("Suciu, Alexandru") == "Alexandru Suciu"
    assert BannerClient.display_to_name("Barabasi, Albert-Laszlo") == "Albert-Laszlo Barabasi"
    # already-natural or malformed input must pass through, not be mangled
    assert BannerClient.display_to_name("Cher") == "Cher"
    assert BannerClient.display_to_name("") == ""


# --- incremental re-runs ---------------------------------------------------


class _MemStore:
    """Minimal OutputStore standing in for a bucket."""

    def __init__(self, records=None):
        self.written: dict[str, str] = {}
        self._records = records or []

    def write_text(self, key, content):
        self.written[key] = content

    def read_text(self, key):
        return self.written.get(key)

    def existing_slugs(self, prefix):
        return set()

    def describe(self):
        return "mem"

    def iter_json(self, prefix):
        return iter(self._records)


def test_catalog_record_hash_ignores_the_subject_page_url():
    """`url` names the subject page, not the course; a path change is not a change."""
    html_a = '<div class="courseblock"><p class="courseblocktitle">CS 1100.  Topics.  (4 Hours)</p>' \
             '<p class="cb_desc">Covers things at considerable length and in real detail.</p></div>'
    a = CatalogFetcher.parse_courses(html_a, "cs", "https://a.example/cs/")[0]
    b = CatalogFetcher.parse_courses(html_a, "cs", "https://b.example/other/")[0]
    assert a["record_hash"] == b["record_hash"]


def test_catalog_record_hash_changes_when_a_requisite_changes():
    base = '<div class="courseblock"><p class="courseblocktitle">CS 1100.  Topics.  (4 Hours)</p>' \
           '<p class="cb_desc">Covers things at considerable length and in real detail.</p>{}</div>'
    a = CatalogFetcher.parse_courses(base.format(""), "cs")[0]
    b = CatalogFetcher.parse_courses(
        base.format('<p class="courseblockextra">Prerequisite(s): CS 1200</p>'), "cs"
    )[0]
    assert a["record_hash"] != b["record_hash"]


def test_scrape_skips_a_course_whose_hash_is_unchanged():
    """Without this the catalog is rewritten every month for no change at all."""
    from preprocessing.sources.courses.runner import scrape

    class _Fetcher:
        def fetch_subject(self, subject):
            return CatalogFetcher.parse_courses(SUBJECT_HTML, subject)

    store = _MemStore()
    written, unchanged, skipped = scrape(_Fetcher(), store, ["cs"])
    # CS 2001's description is under MIN_DESCRIPTION_CHARS, so only CS 3800 stores
    assert (written, unchanged, skipped) == (1, 0, 1)

    prior = {r["slug"]: r["record_hash"] for r in CatalogFetcher.parse_courses(SUBJECT_HTML, "cs")}
    store2 = _MemStore()
    written, unchanged, skipped = scrape(_Fetcher(), store2, ["cs"], existing_hashes=prior)
    assert (written, unchanged) == (0, 1)
    assert store2.written == {}


def test_schedule_hash_tracks_instructors_not_section_counts():
    """A section count moving is not a teaching change and must not force a write."""
    from preprocessing.sources.schedule.runner import schedule_hash

    a = {"terms": [{"description": "Fall 2026", "instructors": ["B", "A"], "section_count": 3}]}
    b = {"terms": [{"description": "Fall 2026", "instructors": ["A", "B"], "section_count": 9}]}
    c = {"terms": [{"description": "Fall 2026", "instructors": ["A"], "section_count": 3}]}
    assert schedule_hash(a) == schedule_hash(b)   # order + counts are noise
    assert schedule_hash(a) != schedule_hash(c)   # a dropped instructor is not


def test_local_store_supports_load_hashes_like_gcs():
    """Local runs must exercise the same skip path the cloud job takes."""
    import json as _json

    from shared.gcs import LocalStore

    def _make(tmp):
        store = LocalStore(tmp)
        store.write_text("courses/cs1.json", _json.dumps({"slug": "cs1", "record_hash": "h1"}))
        store.write_text("courses/cs2.json", _json.dumps({"slug": "cs2", "record_hash": "h2"}))
        return store

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store = _make(Path(tmp))
        assert store.load_hashes("courses/", "record_hash") == {"cs1": "h1", "cs2": "h2"}

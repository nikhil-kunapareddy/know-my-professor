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
from preprocessing.sources.sections.banner import BannerClient
from preprocessing.sources.sections.runner import course_slug
from preprocessing.sources.sections.source import SectionSource

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


def test_sections_mint_the_same_entity_id_as_courses():
    """Enrichment must land on the course it describes, as weblinks do for people."""
    record = {"slug": "cs3800"}
    assert SectionSource().entity_id(record) == CourseSource().entity_id(record)


def test_course_slug_matches_banner_and_catalog_spellings():
    """Banner writes `CS3800`, the catalog writes `CS 3800`; ids must agree."""
    assert course_slug("CS3800") == "cs3800"
    assert course_slug("CS 3800") == "cs3800"
    assert course_slug("ENGW1111") == "engw1111"
    assert course_slug("not a course") is None


# --- namespace -------------------------------------------------------------


def test_both_course_sources_share_one_namespace():
    """A course's description and its instructors must be retrievable together."""
    assert CourseSource().namespace == SectionSource().namespace == COURSES_NAMESPACE


def test_people_stay_in_the_default_namespace():
    """Moving them would re-mint every existing vector."""
    from preprocessing.sources.profiles.source import ProfileSource
    from preprocessing.sources.weblinks.source import WeblinksSource

    assert ProfileSource().namespace is None
    assert WeblinksSource().namespace is None


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
    assert set(grouped) == {None, COURSES_NAMESPACE}
    assert [c.vector_id for c in grouped[None]] == ["a#biography"]
    assert [c.vector_id for c in grouped[COURSES_NAMESPACE]] == ["course-cs3800#course_description"]


def test_pipeline_passes_the_namespace_to_the_retriever():
    from core.pipeline import RAGPipeline

    seen = {}

    class _Retriever:
        def retrieve(self, query_embedding, top_k, filters=None, namespace=None):
            seen["namespace"] = namespace
            return []

    class _Embedder:
        def embed_query(self, q):
            return [0.0]

    pipeline = RAGPipeline(
        embedder=_Embedder(), retriever=_Retriever(), generator=None, top_k=8, min_score=0.0
    )
    pipeline.answer("what does CS 3800 cover?", namespace="courses")
    assert seen["namespace"] == "courses"


# --- ingestable guards -----------------------------------------------------


def test_a_course_with_no_description_is_not_ingested():
    """Placeholder catalog entries would embed a vector matching only its title."""
    source = CourseSource()
    assert source.is_ingestable({"slug": "x", "description": "Covers things."})
    assert not source.is_ingestable({"slug": "x", "description": "   "})
    assert source.to_chunks({"slug": "x", "code": "X 1", "description": ""}) == []


def test_a_course_staffed_entirely_TBA_is_not_ingested():
    source = SectionSource()
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
    chunk = SectionSource().to_chunks(record)[0]
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
    text = SectionSource().to_chunks(record)[0].text
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

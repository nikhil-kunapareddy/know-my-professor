"""Multi-college profile scraping: id namespacing, parser selection, pacing.

The invariant worth the most here is that adding colleges changed *nothing*
about Khoury. tests/test_chunk_golden.py proves the rendered text is unchanged;
these prove the vector ids and metadata are too.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from preprocessing.sources.profiles.config import (
    COLLEGES,
    COLLEGES_BY_KEY,
    DEFAULT_COLLEGE,
    EXTRACTION_SCHEMA,
    College,
)
from preprocessing.sources.profiles.runner import (
    ProfileScraper,
    _Pacer,
    _resolve_colleges,
    _urls_key,
    build_parser,
)
from preprocessing.sources.profiles.source import ProfileSource

KHOURY = COLLEGES_BY_KEY[DEFAULT_COLLEGE]
COS = COLLEGES_BY_KEY["cos"]


def _record(slug="jane-doe", college=None, **extra):
    record = {"slug": slug, "name": "Jane Doe", "biography": "Studies compilers.", **extra}
    if college:
        record["college"] = college
    return record


# --- college registry ------------------------------------------------------


def test_college_keys_are_unique():
    keys = [c.key for c in COLLEGES]
    assert len(keys) == len(set(keys))


def test_khoury_is_first_and_uses_the_accordion_parser():
    """Khoury's chunk text is pinned byte-for-byte; an LLM cannot reproduce it."""
    assert COLLEGES[0].key == DEFAULT_COLLEGE
    assert COLLEGES[0].parser == "accordion"


def test_listing_and_profile_url_pattern():
    assert COS.listing == "https://cos.northeastern.edu/people/"
    assert COS.profile_url_re.match("https://cos.northeastern.edu/people/jane-doe/")
    # the listing itself and its pagination links must not look like profiles
    assert not COS.profile_url_re.match("https://cos.northeastern.edu/people/")
    assert not COS.profile_url_re.match("https://cos.northeastern.edu/people/page/2/")
    # nor may one college's regex swallow another's URLs
    assert not COS.profile_url_re.match("https://www.khoury.northeastern.edu/people/jane-doe/")


def test_extraction_schema_covers_every_rendered_section():
    """A section the schema can't produce would silently never be populated."""
    rendered = {spec.key for spec in ProfileSource.sections}
    assert rendered <= set(EXTRACTION_SCHEMA["properties"])
    # strictness is what makes the structured output a guarantee, not a hope
    assert EXTRACTION_SCHEMA["additionalProperties"] is False


# --- entity ids ------------------------------------------------------------


def test_khoury_records_keep_bare_ids():
    source = ProfileSource()
    for record in (_record(), _record(college="khoury")):
        assert source.entity_id(record) == "jane-doe"
        assert source.to_chunks(record)[0].vector_id == "jane-doe#biography"


def test_other_colleges_are_namespaced():
    source = ProfileSource()
    record = _record(college="cos")
    assert source.entity_id(record) == "cos-jane-doe"
    assert source.to_chunks(record)[0].vector_id == "cos-jane-doe#biography"


def test_same_slug_in_two_colleges_does_not_collide():
    """The whole point: a shared name must not overwrite another college's vector."""
    source = ProfileSource()
    khoury_ids = {c.vector_id for c in source.to_chunks(_record())}
    cos_ids = {c.vector_id for c in source.to_chunks(_record(college="cos"))}
    assert khoury_ids and cos_ids
    assert not (khoury_ids & cos_ids)


def test_entity_id_is_none_without_a_slug():
    assert ProfileSource().entity_id({"college": "cos"}) is None
    assert ProfileSource().to_chunks({"college": "cos"}) == []


# --- metadata --------------------------------------------------------------


def test_college_metadata_is_absent_for_legacy_records():
    """Pre-multi-college records must render exactly as before, keys included."""
    chunk = ProfileSource().to_chunks(_record())[0]
    assert "college" not in chunk.metadata


def test_college_metadata_is_present_when_the_record_carries_it():
    chunk = ProfileSource().to_chunks(_record(college="cos"))[0]
    assert chunk.metadata["college"] == "cos"
    assert chunk.metadata["professor_slug"] == "jane-doe"  # slug stays un-namespaced


def test_adding_a_college_does_not_change_content_hash():
    """content_hash drives re-embedding; metadata must never move it."""
    plain = ProfileSource().to_chunks(_record())[0]
    tagged = ProfileSource().to_chunks(_record(college="khoury"))[0]
    assert plain.text == tagged.text
    assert plain.metadata["content_hash"] == tagged.metadata["content_hash"]


# --- runner wiring ---------------------------------------------------------


def test_record_key_namespaces_only_non_default_colleges():
    class _Store:
        pass

    khoury_scraper = ProfileScraper(None, None, _Store(), KHOURY)
    cos_scraper = ProfileScraper(None, None, _Store(), COS)
    assert khoury_scraper.record_key("jane-doe") == "jane-doe"
    assert cos_scraper.record_key("jane-doe") == "cos-jane-doe"


def test_urls_cache_key_keeps_khoury_on_the_original_filename():
    assert _urls_key(KHOURY) == "profile_urls.json"
    assert _urls_key(COS) == "profile_urls_cos.json"


def test_build_parser_picks_by_template():
    from preprocessing.sources.profiles.llm_parser import LlmProfileParser
    from preprocessing.sources.profiles.profile_parser import ProfileParser

    assert isinstance(build_parser(KHOURY), ProfileParser)
    assert isinstance(build_parser(COS), LlmProfileParser)


def test_resolve_colleges_defaults_to_all_and_rejects_unknown():
    assert _resolve_colleges(None) == list(COLLEGES)
    assert _resolve_colleges("cos,camd") == [COLLEGES_BY_KEY["cos"], COLLEGES_BY_KEY["camd"]]
    with pytest.raises(SystemExit):
        _resolve_colleges("school-of-hard-knocks")


def test_pacer_holds_the_delay_across_concurrent_callers():
    """Workers must not all fire at once and merely stagger the next round."""
    pacer = _Pacer(0.05)
    start = time.monotonic()
    for _ in range(4):
        pacer.wait()
    # first call is free, the next three each wait one delay
    assert time.monotonic() - start >= 0.15


def test_crawl_delay_is_honoured_per_college():
    assert COLLEGES_BY_KEY["camd"].crawl_delay == 10.0  # camd's robots.txt asks for it
    assert KHOURY.crawl_delay == 1.0


# --- LLM parser (no network) ----------------------------------------------


class _FakeResponse:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [type("Block", (), {"type": "text", "text": text})()]
        self.stop_reason = stop_reason


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.messages = type("M", (), {"create": lambda _self, **kw: response})()


def _parser(response):
    from preprocessing.sources.profiles.llm_parser import LlmProfileParser

    return LlmProfileParser(client=_FakeClient(response))


PAGE = "<html><body><article>" + ("Jane Doe is a professor of compilers. " * 12) + "</article></body></html>"


def test_llm_parser_maps_fields_onto_the_profile_model():
    payload = (
        '{"name": "Jane Doe", "title": "Professor", "biography": "Studies compilers.",'
        ' "research_interests": ["type systems"], "education": ["PhD, MIT"],'
        ' "areas_of_interest": ["PL"], "labs_and_groups": ["PLT"], "projects": ["a compiler"]}'
    )
    profile = _parser(_FakeResponse(payload)).parse(
        "https://cos.northeastern.edu/people/jane-doe/", PAGE
    )
    assert profile.slug == "jane-doe"
    assert profile.name == "Jane Doe"
    assert profile.research_interests == ["type systems"]
    assert profile.projects == ["a compiler"]
    # fields this template cannot supply stay empty rather than fabricated
    assert profile.pronouns is None and profile.websites == []


def test_llm_parser_rejects_a_thin_page_before_calling_the_model():
    from preprocessing.sources.profiles.llm_parser import ThinProfilePage

    def _explode(**kw):  # pragma: no cover - must never run
        raise AssertionError("model called for a thin page")

    parser = _parser(_FakeResponse("{}"))
    parser._client.messages.create = _explode
    with pytest.raises(ThinProfilePage):
        parser.parse("https://cos.northeastern.edu/people/x/", "<html><body>Hi</body></html>")


def test_llm_parser_treats_truncation_as_a_thin_page():
    """A max_tokens cut-off yields invalid JSON; half a profile must not be stored."""
    from preprocessing.sources.profiles.llm_parser import ThinProfilePage

    with pytest.raises(ThinProfilePage):
        _parser(_FakeResponse('{"name": "Ja', stop_reason="max_tokens")).parse(
            "https://cos.northeastern.edu/people/jane-doe/", PAGE
        )


def test_llm_parser_treats_unparseable_output_as_a_thin_page():
    from preprocessing.sources.profiles.llm_parser import ThinProfilePage

    with pytest.raises(ThinProfilePage):
        _parser(_FakeResponse("not json at all")).parse(
            "https://cos.northeastern.edu/people/jane-doe/", PAGE
        )


def test_retry_predicate_only_retries_transient_failures():
    anthropic = pytest.importorskip("anthropic")
    from preprocessing.sources.profiles.llm_parser import _is_retryable

    assert not _is_retryable(ValueError("nope"))
    assert not _is_retryable(anthropic.BadRequestError.__new__(anthropic.BadRequestError))

    rate_limited = anthropic.RateLimitError.__new__(anthropic.RateLimitError)
    assert _is_retryable(rate_limited)


def test_college_is_a_frozen_value_object():
    college = College("x", "https://x.example.com")
    with pytest.raises(dataclasses.FrozenInstanceError):
        college.key = "y"

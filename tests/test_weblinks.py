"""Unit tests for the weblinks crawl + extract + record-building logic."""

from __future__ import annotations

import pytest


@pytest.fixture
def crawl(load):
    return load("weblinks", "crawl")


@pytest.fixture
def extract(load):
    return load("weblinks", "extract")


@pytest.fixture
def weblinks(load):
    return load("weblinks", "weblinks")


# --- crawl: one-hop link selection ----------------------------------------


def test_select_one_hop_links_same_host_keyword_dedup(crawl):
    html = """
      <a href="/research/">My Research</a>
      <a href="publications.html">Papers</a>
      <a href="https://other-site.com/lab">External lab</a>
      <a href="/contact">Contact</a>
      <a href="#top">anchor</a>
      <a href="/research/">dup research</a>
      <a href="mailto:x@y.com">email</a>
      <a href="/about-me">About</a>
    """
    links = crawl._select_one_hop_links(html, "https://jane.example")
    assert links == [
        "https://jane.example/research",
        "https://jane.example/publications.html",
        "https://jane.example/about-me",
    ]


def test_select_one_hop_links_excludes_cross_host(crawl):
    html = '<a href="https://elsewhere.org/research">x</a>'
    assert crawl._select_one_hop_links(html, "https://jane.example") == []


class _FakeResp:
    """Mimics the requests encoding behavior: .text depends on .encoding."""

    def __init__(self, content_type):
        self.status_code = 200
        self.headers = {"Content-Type": content_type}
        self.apparent_encoding = "utf-8"
        self.encoding = "ISO-8859-1"  # what requests defaults to absent a charset
        self._by_encoding = {"ISO-8859-1": "mojibake", "utf-8": "correct"}

    @property
    def text(self):
        return self._by_encoding.get(self.encoding, "?")


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp

    def get(self, url, timeout=None):
        return self._resp


def test_fetch_falls_back_to_detected_encoding_when_no_charset(crawl):
    html, reason = crawl._fetch(_FakeSession(_FakeResp("text/html")), "https://x/")
    assert reason is None
    assert html == "correct"  # apparent_encoding (utf-8) used, not ISO-8859-1


def test_fetch_respects_declared_charset(crawl):
    html, reason = crawl._fetch(
        _FakeSession(_FakeResp("text/html; charset=ISO-8859-1")), "https://x/"
    )
    assert reason is None
    assert html == "mojibake"  # server declared it, so we don't override


def test_fetch_reports_http_and_non_html(crawl):
    bad = _FakeResp("text/html")
    bad.status_code = 404
    assert crawl._fetch(_FakeSession(bad), "https://x/") == (None, "http_404")

    nonhtml = _FakeResp("application/pdf")
    assert crawl._fetch(_FakeSession(nonhtml), "https://x/") == (None, "non_html")


# --- extract: guard, hashing, cleaning ------------------------------------


def test_extraction_succeeded_threshold(extract):
    from config import MIN_CLEAN_TEXT_CHARS

    assert not extract.extraction_succeeded(" " * (MIN_CLEAN_TEXT_CHARS - 1))
    assert extract.extraction_succeeded("x" * MIN_CLEAN_TEXT_CHARS)


def test_page_hash_deterministic_and_text_sensitive(extract):
    assert extract.page_hash("hello world") == extract.page_hash("hello world")
    assert extract.page_hash("hello world") != extract.page_hash("hello worlx")
    assert extract.page_hash("x").startswith("sha256:")


def test_page_hash_changes_with_schema_version(extract, monkeypatch):
    baseline = extract.page_hash("same text")
    monkeypatch.setattr(extract, "SCHEMA_VERSION", "v999")
    assert extract.page_hash("same text") != baseline  # schema bump forces refresh


def test_clean_pages_sorts_by_url_and_labels(extract, monkeypatch):
    # echo the html so the test is independent of real trafilatura behavior
    monkeypatch.setattr(extract.trafilatura, "extract", lambda html, **kw: html)
    pages = [("https://b.example/", "B-content"), ("https://a.example/", "A-content")]
    combined = extract.clean_pages(pages)
    assert combined.index("[https://a.example/]") < combined.index("[https://b.example/]")
    assert "A-content" in combined and "B-content" in combined


def test_clean_pages_truncates(extract, monkeypatch):
    from config import MAX_CLEAN_TEXT_CHARS

    monkeypatch.setattr(extract.trafilatura, "extract", lambda html, **kw: html)
    pages = [("https://a.example/", "x" * (MAX_CLEAN_TEXT_CHARS + 5000))]
    assert len(extract.clean_pages(pages)) == MAX_CLEAN_TEXT_CHARS


# --- weblinks orchestration helpers ---------------------------------------


def test_website_url_picks_first_http_link(weblinks):
    assert weblinks.website_url({"websites": [{"href": "https://a/"}]}) == "https://a/"
    assert weblinks.website_url(
        {"websites": [{"href": "mailto:x@y"}, {"href": "https://b/"}]}
    ) == "https://b/"
    assert weblinks.website_url({"websites": []}) is None
    assert weblinks.website_url({}) is None


def test_build_record_drops_empties_and_stamps_metadata(weblinks):
    profile = {"slug": "s", "name": "N", "title": "T"}
    extracted = {
        "website_summary": "Studies types.",
        "current_projects": [],
        "recent_publications": ["Paper 1"],
        "students_or_lab_members": [],
        "recent_news": [],
    }
    rec = weblinks.build_record(profile, "https://src/", "sha256:abc", extracted)

    assert rec["slug"] == "s"
    assert rec["schema_version"] == weblinks.SCHEMA_VERSION
    assert rec["page_hash"] == "sha256:abc"
    assert rec["source_url"] == "https://src/"
    # only non-empty fields become sections, in SECTION_TYPES order
    assert [s["section_type"] for s in rec["sections"]] == [
        "website_summary", "recent_publications",
    ]
    assert all(s["source_url"] == "https://src/" for s in rec["sections"])

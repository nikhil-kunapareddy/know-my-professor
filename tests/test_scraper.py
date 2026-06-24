"""Unit tests for the scraper HTML parsing (profile_parser + fetcher discovery)."""

from __future__ import annotations

import pytest

PROFILE_HTML = """
<html><body>
  <h1 class="single-people__header-title">Jane Doe</h1>
  <p class="single-people__header-subtitle">she/her</p>
  <p class="single-people__header-description">Associate Professor</p>
  <figure class="single-people__header-figure"><img src="https://x/photo.jpg"></figure>
  <p class="single-people__aside-roles">Faculty, Researcher</p>

  <div class="single-people__aside-block"><h3>Campus</h3>
    <ul><li class="single-people__aside-list-item">Boston</li></ul></div>
  <div class="single-people__aside-block"><h3>Website</h3>
    <ul><li class="single-people__aside-list-item"><a href="https://jane.example/">jane.example</a></li></ul></div>
  <div class="single-people__aside-block"><h3>Google Scholar</h3>
    <ul><li class="single-people__aside-list-item"><a href="https://scholar.google.com/x">Scholar</a></li></ul></div>
  <div class="single-people__aside-block"><h3>Area of interest</h3>
    <ul><li class="single-people__aside-list-item">Programming Languages</li></ul></div>

  <div class="accordion-item"><button class="accordion-header">Biography</button>
    <div class="accordion-content__inner"><p>Jane works on PL.</p></div></div>
  <div class="accordion-item"><button class="accordion-header">Research interests</button>
    <div class="accordion-content__inner">
      <ul class="wp-block-list"><li>Types</li><li>Compilers</li></ul></div></div>
</body></html>
"""


@pytest.fixture
def profile_parser(load):
    return load("scraper", "profile_parser")


@pytest.fixture
def fetcher(load):
    return load("scraper", "fetcher")


def test_parse_profile_header_fields(profile_parser):
    p = profile_parser.parse_profile(
        "https://www.khoury.northeastern.edu/people/jane-doe/", PROFILE_HTML
    )
    assert p.slug == "jane-doe"
    assert p.name == "Jane Doe"
    assert p.pronouns == "she/her"
    assert p.title == "Associate Professor"
    assert p.photo_url == "https://x/photo.jpg"
    assert p.roles == ["Faculty", "Researcher"]


def test_parse_profile_aside_fields(profile_parser):
    p = profile_parser.parse_profile(
        "https://www.khoury.northeastern.edu/people/jane-doe/", PROFILE_HTML
    )
    assert p.campuses == ["Boston"]
    assert p.websites == [{"text": "jane.example", "href": "https://jane.example/"}]
    assert p.google_scholar == "https://scholar.google.com/x"
    assert p.areas_of_interest == ["Programming Languages"]


def test_parse_profile_accordion_sections(profile_parser):
    p = profile_parser.parse_profile(
        "https://www.khoury.northeastern.edu/people/jane-doe/", PROFILE_HTML
    )
    assert p.biography == "Jane works on PL."
    assert p.research_interests == ["Types", "Compilers"]
    assert p.education == []  # absent section -> empty list


def test_parse_profile_handles_empty_html(profile_parser):
    p = profile_parser.parse_profile(
        "https://www.khoury.northeastern.edu/people/empty/", "<html></html>"
    )
    assert p.slug == "empty"
    assert p.name is None
    assert p.websites == []
    assert p.biography is None


def test_extract_total_pages(fetcher):
    html = """
      <a class="page-numbers" href="https://www.khoury.northeastern.edu/people/page/2/">2</a>
      <a class="page-numbers" href="https://www.khoury.northeastern.edu/people/page/5/">5</a>
      <a class="page-numbers" href="https://www.khoury.northeastern.edu/people/page/3/">3</a>
    """
    assert fetcher.extract_total_pages(html) == 5
    assert fetcher.extract_total_pages("<html></html>") == 1  # no pagination -> 1


def test_extract_profile_urls_matches_only_profile_slugs(fetcher):
    html = """
      <a href="https://www.khoury.northeastern.edu/people/jane-doe/">a</a>
      <a href="https://www.khoury.northeastern.edu/people/john-smith/">b</a>
      <a href="https://www.khoury.northeastern.edu/people/page/2/">paginate</a>
      <a href="https://other.com/people/x/">external</a>
    """
    urls = fetcher.extract_profile_urls(html)
    assert urls == {
        "https://www.khoury.northeastern.edu/people/jane-doe/",
        "https://www.khoury.northeastern.edu/people/john-smith/",
    }

"""The programs source: CourseLeaf program pages -> requirement chunks.

The HTML below reproduces the catalog's real markup, row for row, as served on
the BSCS page (2026-09-29): the hidden screen-reader header, ``areaheader``
rows, ``orclass`` alternatives, comment rows carrying hours, ``div.blockindent``
options, and the ``span.blockindent`` "and" of a paired course -- which is NOT
an indent, and which a looser rule gets wrong.
"""

from __future__ import annotations

import json

import pytest

from preprocessing.documents.ir import BlockKind, Document
from preprocessing.sources.programs import courseleaf
from preprocessing.sources.programs.courseleaf import parse_program, sitemap_program_urls, slug_for
from preprocessing.sources.programs.runner import scrape
from preprocessing.sources.programs.source import ProgramSource

URL = "https://catalog.northeastern.edu/undergraduate/computer-information-science/computer-science/bscs/"


def _course(code, title, hours="", indent=False, pair=None, cls="even"):
    link = f'<a class="bubblelink code" title="{code}">{code.replace(" ", "&#160;")}</a>'
    if pair:
        link += f'<br/><span class="blockindent" style="margin-left:20px;">and <a class="bubblelink code">{pair}</a></span>'
    if indent:
        link = f'<div class="blockindent" style="margin-left:20px;">{link}</div>'
    return f'<tr class="{cls}"><td class="codecol">{link}</td><td>{title}</td><td class="hourscol">{hours}</td></tr>'


def _or(code, title):
    return (f'<tr class="orclass odd"><td class="codecol"><div style="margin-left:20px;" class="blockindent">'
            f'or <a class="bubblelink code">{code}</a></div></td><td>{title}</td><td class="hourscol"></td></tr>')


def _area(text):
    return (f'<tr class="even areaheader"><td colspan="2"><span class="courselistcomment areaheader">{text}</span>'
            f'</td><td class="hourscol"></td></tr>')


def _comment(text, hours="", indent=False):
    body = f'<span class="courselistcomment">{text}</span>'
    if indent:
        body = f'<div class="blockindent" style="margin-left:20px;">{body}</div>'
    return f'<tr class="odd"><td colspan="2">{body}</td><td class="hourscol">{hours}</td></tr>'


def _courselist(*rows):
    head = ('<tr class="hidden noscript"><th scope="col">Code</th><th scope="col">Title</th>'
            '<th scope="col" class="hourscol">Hours</th></tr>')
    return f'<table class="sc_courselist"><colgroup></colgroup><tbody>{head}{"".join(rows)}</tbody></table>'


ELECTIVES = [_course(f"CS {4100 + i}", f"Upper-Division Elective Number {i} in Computing", indent=True) for i in range(40)]

PAGE = f"""<html><body>
<h1 class="page-title">Computer Science, BSCS (Boston)</h1>
<nav id="tabs"><ul>
  <li><a href="#textcontainer">Overview</a></li>
  <li><a href="#programrequirementstextcontainer">Program Requirements</a></li>
</ul></nav>
<div id="textcontainer" class="page_content tab_content">
  <p>The BSCS prepares students for careers in software and research.</p>
</div>
<div id="programrequirementstextcontainer" class="page_content tab_content">
  <a name="requirementstext"></a>
  <p>Complete all courses listed below unless otherwise indicated.</p>
  <h2>Computer Science Requirements</h2>
  {_courselist(
      _area("Computer Science Overview"),
      _course("CS 1200", "First Year Seminar", "1"),
      _or("INPR 1000", "First-Year Interdisciplinary Seminar"),
      _area("Computer Science Fundamental Courses"),
      _course("CS 1800", "Discrete Structures", "4"),
      _or("MATH 1365", "Introduction to Mathematical Reasoning"),
      _or("MATH 1465", "Intensive Mathematical Reasoning"),
      _course("CS 2000", "Introduction to Program Design and Implementation and Lab for CS 2000", "5", pair="CS 2001"),
      _area("Security Required Course"),
      _comment("Complete one of the following:", "4"),
      _course("CY 2550", "Foundations of Cybersecurity", indent=True),
      _course("CY 3740", "Systems Security", indent=True),
  )}
  <h2>Khoury Approved Electives</h2>
  <p>Complete 12 semester hours from the following options.</p>
  {_courselist(_comment("CS 2300 or higher, except CS 5010", indent=True), *ELECTIVES)}
  <h2>Concentrations</h2>
  <ul><li>Artificial Intelligence</li><li>Systems</li></ul>
</div>
</body></html>"""


@pytest.fixture(scope="module")
def record():
    return parse_program(URL, PAGE)


def _blocks(record, key="program_requirements"):
    return Document.from_dict(record[key]).blocks


# --- parsing ----------------------------------------------------------------------


def test_identity_fields(record):
    assert record["slug"] == "undergraduate-computer-information-science-computer-science-bscs"
    assert record["name"] == "Computer Science, BSCS (Boston)"
    assert (record["level"], record["catalog_college"]) == ("undergraduate", "computer-information-science")
    assert record["record_hash"].startswith("sha256:")


def test_area_headers_become_headings_under_their_h2(record):
    headings = [(b.text, b.level) for b in _blocks(record) if b.kind is BlockKind.HEADING]
    assert headings == [
        ("Computer Science Requirements", 2),
        ("Computer Science Overview", 3),
        ("Computer Science Fundamental Courses", 3),
        ("Security Required Course", 3),
        ("Khoury Approved Electives", 2),
        ("Concentrations", 2),
    ]


def test_a_second_table_under_one_h2_does_not_nest_inside_the_first():
    page = PAGE.replace(
        "<h2>Khoury Approved Electives</h2>",
        _courselist(_area("Writing Requirement"), _course("ENGW 3302", "Advanced Writing", "4"))
        + "<h2>Khoury Approved Electives</h2>",
    )
    headings = [(b.text, b.level) for b in _blocks(parse_program(URL, page)) if b.kind is BlockKind.HEADING]
    assert ("Security Required Course", 3) in headings and ("Writing Requirement", 3) in headings


def test_rows_keep_their_meaning(record):
    tables = [b for b in _blocks(record) if b.kind is BlockKind.TABLE]
    overview, fundamentals, security = tables[:3]
    assert overview.rows == (
        ("Code", "Title", "Hours"),
        ("CS 1200 or INPR 1000", "First Year Seminar or First-Year Interdisciplinary Seminar", "1"),
    )
    assert fundamentals.rows[1] == (
        "CS 1800 or MATH 1365 or MATH 1465",
        "Discrete Structures or Introduction to Mathematical Reasoning or Intensive Mathematical Reasoning",
        "4",
    )
    # the paired course's "and" span is not an indent
    assert fundamentals.rows[2][0] == "CS 2000 and CS 2001"
    assert security.rows[1:] == (
        ("Complete one of the following:", "", "4"),
        ("- CY 2550", "Foundations of Cybersecurity", ""),
        ("- CY 3740", "Systems Security", ""),
    )
    assert all(t.header_rows == 1 and t.rows[0] == ("Code", "Title", "Hours") for t in tables)


def test_overview_and_lists(record):
    [overview] = _blocks(record, "program_overview")
    assert overview.text == "The BSCS prepares students for careers in software and research."
    [concentrations] = [b for b in _blocks(record) if b.kind is BlockKind.LIST]
    assert concentrations.items == ("Artificial Intelligence", "Systems")


def test_a_page_without_a_requirements_tab_is_not_a_program():
    department = PAGE.replace("programrequirementstextcontainer", "programstextcontainer")
    assert parse_program(URL, department) is None


def test_several_requirement_tabs_are_told_apart_by_their_labels():
    page = PAGE.replace(
        '<li><a href="#programrequirementstextcontainer">Program Requirements</a></li>',
        '<li><a href="#programrequirementstextcontainer">Program Requirements</a></li>'
        '<li><a href="#advancedentryphdprogramrequirementstextcontainer">Advanced Entry</a></li>',
    ).replace("</body>", '<div id="advancedentryphdprogramrequirementstextcontainer"><p>Fewer courses.</p></div></body>')
    blocks = _blocks(parse_program(URL, page))
    level_one = [b.text for b in blocks if b.kind is BlockKind.HEADING and b.level == 1]
    assert level_one == ["Program Requirements", "Advanced Entry"]


def test_record_hash_moves_only_with_content():
    assert parse_program(URL, PAGE)["record_hash"] == parse_program(URL, PAGE)["record_hash"]
    changed = parse_program(URL, PAGE.replace("Systems Security", "Systems Security 2"))
    assert changed["record_hash"] != parse_program(URL, PAGE)["record_hash"]


def test_text_is_normalised():
    assert courseleaf._norm("CS\xa02000  and CS\xa02001 .") == "CS 2000 and CS 2001."


def test_slugs_are_the_whole_path():
    assert slug_for(URL) == "undergraduate-computer-information-science-computer-science-bscs"
    assert slug_for(URL.replace("bscs", "minor")) != slug_for(URL.replace("computer-science/bscs", "minor"))


def test_sitemap_filter_keeps_program_candidates_only():
    locs = [
        "https://catalog.northeastern.edu/course-descriptions/cs/",
        "https://catalog.northeastern.edu/undergraduate/",
        "https://catalog.northeastern.edu/undergraduate/computer-information-science/",
        "https://catalog.northeastern.edu/undergraduate/computer-information-science/computer-science/bscs/",
        "https://catalog.northeastern.edu/graduate/science/academic-policies-procedures/attendance/",
        "https://catalog.northeastern.edu/graduate/computer-information-science/artificial-intelligence-ms/",
        "https://catalog.northeastern.edu/handbook/some-policy/deep/page/",
    ]
    xml = "<urlset>" + "".join(f"<url><loc>{u}</loc></url>" for u in locs) + "</urlset>"
    assert sitemap_program_urls(xml, ("undergraduate", "graduate", "professional-studies"), 3,
                                {"academic-policies-procedures"}) == [locs[5], locs[3]]


# --- chunks --------------------------------------------------------------------------


def test_chunks_follow_the_page_structure(record):
    chunks = ProgramSource().to_chunks(record)
    by_id = {c.vector_id: c for c in chunks}
    base = "program-undergraduate-computer-information-science-computer-science-bscs"

    assert f"{base}#program_overview" in by_id
    security = next(c for c in chunks if "Security Required Course" in c.text)
    assert security.text.startswith(
        "Computer Science, BSCS (Boston) — Undergraduate program\n"
        "Program requirements > Computer Science Requirements > Security Required Course:\n"
        "| Code | Title | Hours |"
    )
    assert security.metadata["heading_path"] == "Computer Science Requirements > Security Required Course"
    assert security.metadata["program_name"] == "Computer Science, BSCS (Boston)"
    assert security.metadata["section_type"] == "program_requirements"
    assert all(c.metadata["text"] == c.text for c in chunks)


def test_an_oversized_area_splits_between_rows_with_its_header(record):
    electives = [c for c in ProgramSource().to_chunks(record) if "Khoury Approved Electives" in c.text]
    assert len(electives) > 1
    # parts are numbered across the whole requirements section, so the area's
    # pieces are consecutive parts rather than starting again at 2
    parts = [c.metadata["part"] for c in electives]
    assert parts == list(range(parts[0], parts[0] + len(electives)))
    assert all(c.vector_id.endswith(f"@{n}") for c, n in zip(electives, parts, strict=True))
    for c in electives[1:]:
        assert "Program requirements > Khoury Approved Electives (continued):\n| Code | Title | Hours |" in c.text
    rows = [line for c in electives for line in c.text.splitlines() if line.startswith("| - CS 41")]
    assert len(rows) == 40  # every elective survives the split, once


def test_is_ingestable():
    assert not ProgramSource().is_ingestable({"slug": "x", "program_requirements": {"ir_version": 1, "blocks": []}})
    assert ProgramSource().to_chunks({"program_requirements": {}}) == []


def test_programs_share_the_courses_namespace_without_sharing_ids():
    from preprocessing.sources.courses.source import CourseSource

    assert ProgramSource.namespace == CourseSource.namespace
    assert ProgramSource().entity_id({"slug": "cs1800"}) != CourseSource().entity_id({"slug": "cs1800"})


# --- the job ----------------------------------------------------------------------------


class _Fetcher:
    def __init__(self, pages):
        self.pages = pages

    def fetch(self, url):
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        return page


class _Store:
    def __init__(self):
        self.written = {}

    def write_text(self, key, text):
        self.written[key] = json.loads(text)


def test_scrape_writes_programs_and_skips_the_rest():
    department = "https://catalog.northeastern.edu/undergraduate/computer-information-science/computer-science/"
    broken = "https://catalog.northeastern.edu/undergraduate/x/y/z/"
    fetcher = _Fetcher({URL: PAGE, department: "<html><h1>Department</h1></html>", broken: TimeoutError("slow")})
    store = _Store()

    counts = scrape(fetcher, store, [URL, department, broken], sleep=lambda s: None)
    assert counts == {"written": 1, "unchanged": 0, "not_program": 1, "failed": 1}
    [(key, stored)] = store.written.items()
    assert key == "programs/undergraduate-computer-information-science-computer-science-bscs.json"
    assert stored == parse_program(URL, PAGE)

    again = scrape(fetcher, _Store(), [URL], existing_hashes={stored["slug"]: stored["record_hash"]}, sleep=lambda s: None)
    assert again["unchanged"] == 1 and again["written"] == 0


def test_a_dropped_connection_is_retried_and_a_404_is_not():
    import requests

    class Flaky(_Fetcher):
        calls = 0

        def fetch(self, url):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise requests.ConnectionError("Remote end closed connection without response")
            return super().fetch(url)

    counts = scrape(Flaky({URL: PAGE}), _Store(), [URL], sleep=lambda s: None)
    assert counts["written"] == 1 and Flaky.calls == 2

    missing = requests.HTTPError(response=type("R", (), {"status_code": 404})())
    counts = scrape(_Fetcher({URL: missing}), _Store(), [URL], sleep=lambda s: None)
    assert counts["failed"] == 1


def test_scrape_paces_between_requests_and_dry_runs_write_nothing():
    slept, store = [], _Store()
    scrape(_Fetcher({URL: PAGE, URL + "x/": PAGE}), store, [URL, URL + "x/"], delay=1.5, dry_run=True, sleep=slept.append)
    assert slept == [1.5] and store.written == {}

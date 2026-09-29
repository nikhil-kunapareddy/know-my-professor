"""CourseLeaf program pages -> records whose tabs are IR ``Document``s.

This is a site-specific tier 0 in front of the generic HTML tier
(``preprocessing/documents/tiers/html.py``), and it exists because requirement
tables carry their meaning in row CLASSES that a generic reader discards:

- ``tr.areaheader`` is a sub-heading ("Computer Science Required Courses"). It
  becomes a heading one level below the ``<h2>`` it sits under, so every area
  is its own chunk section, titled.
- ``tr.orclass`` ("or MATH 1365") is an alternative to the row above it. It is
  merged into that row -- ``CS 1800 or MATH 1365`` -- because as a row of its
  own it reads as one more required course.
- A row holding only ``span.courselistcomment`` ("Complete one of the
  following:", with the hours in the hours column) introduces the options
  under it, which CourseLeaf indents with ``div.blockindent``. Indented rows
  get a ``- `` prefix on their code, so the option group survives as text.
- The first row is a hidden ``<th>`` header for screen readers; every table
  gets the same ``Code | Title | Hours`` header instead.

Everything else in a tab -- headings, paragraphs, lists -- maps straight onto
blocks. A page is a program iff it has a requirements tab (see config).
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

from ...documents.ir import Block, BlockKind, Document
from ...documents.probe import HTML
from .config import OVERVIEW_TAB, REQUIREMENTS_TAB_SUFFIX

TABLE_HEADER = ("Code", "Title", "Hours")
_SKIP_TAGS = {"script", "style", "noscript", "a", "hr", "br", "img", "button", "form"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)])")


def _norm(text: str | None) -> str:
    """Whitespace collapsed, and no space before punctuation.

    ``str.split`` also takes CourseLeaf's ``\\xa0`` in ``CS\\xa01800``. Joining a
    tag's text with spaces -- needed so ``<a>CS 2000</a>and`` does not fuse --
    leaves "place out of CS 2001 ." where a link meets a full stop.
    """
    return _SPACE_BEFORE_PUNCT.sub(r"\1", " ".join((text or "").split()))


def _text(el) -> str:
    return _norm(el.get_text(" ")) if el is not None else ""


def path_segments(url: str) -> list[str]:
    return [s for s in urlparse(url).path.split("/") if s]


def slug_for(url: str) -> str:
    """The whole catalog path, joined: unique by construction.

    ``undergraduate/computer-information-science/computer-science/bscs/`` ->
    ``undergraduate-computer-information-science-computer-science-bscs``. The last
    segment alone is not unique ("minor" appears hundreds of times), and neither
    is program-within-level, since two colleges can each run a program of one name.
    """
    return "-".join(path_segments(url))


# --- tabs -----------------------------------------------------------------------


def _tabs(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """(container id, tab label) for each tab, in page order."""
    tabs = []
    for a in soup.select("#tabs a[href^='#']"):
        tab_id = a["href"][1:]
        if soup.find(id=tab_id) is not None:
            tabs.append((tab_id, _text(a)))
    return tabs


class _Walker:
    """Folds one tab's DOM into blocks, tracking the heading level it is under."""

    def __init__(self) -> None:
        self.blocks: list[Block] = []
        #: Level of the last real ``<h*>`` tag. Area headers sit one below it --
        #: not below the last area header, or the second table under one <h2>
        #: would nest its areas inside the first table's last area.
        self.dom_level = 1

    def heading(self, text: str, level: int, *, from_dom: bool = True) -> None:
        if text:
            level = max(1, min(level, 6))
            if from_dom:
                self.dom_level = level
            self.blocks.append(Block(BlockKind.HEADING, text=text, level=level))

    def walk(self, el: Tag) -> list[Block]:
        for child in el.children:
            if isinstance(child, NavigableString):
                if text := _norm(str(child)):
                    self.blocks.append(Block(BlockKind.PARAGRAPH, text=text))
                continue
            if not isinstance(child, Tag) or child.name in _SKIP_TAGS or "hidden" in (child.get("class") or []):
                continue
            if child.name in _HEADINGS:
                self.heading(_text(child), int(child.name[1]))
            elif child.name == "table":
                if "sc_courselist" in (child.get("class") or []):
                    self._courselist(child)
                else:
                    self._table(child)
            elif child.name in ("ul", "ol"):
                items = tuple(t for li in child.find_all("li", recursive=False) if (t := _text(li)))
                if items:
                    self.blocks.append(Block(BlockKind.LIST, items=items))
            elif child.name == "p":
                if text := _text(child):
                    self.blocks.append(Block(BlockKind.PARAGRAPH, text=text))
            elif child.name in ("div", "section", "article", "span", "blockquote"):
                self.walk(child)
            elif text := _text(child):
                self.blocks.append(Block(BlockKind.PARAGRAPH, text=text))
        return self.blocks

    def _table(self, table: Tag) -> None:
        rows = tuple(
            tuple(_text(c) for c in tr.find_all(["th", "td"]))
            for tr in table.find_all("tr")
            if tr.find_all(["th", "td"])
        )
        rows = tuple(r for r in rows if any(r))
        if rows:
            width = max(len(r) for r in rows)
            first_is_head = bool(table.find("tr") and table.find("tr").find("th"))
            self.blocks.append(
                Block(
                    BlockKind.TABLE,
                    rows=tuple(r + ("",) * (width - len(r)) for r in rows),
                    header_rows=1 if first_is_head and len(rows) > 1 else 0,
                )
            )

    def _courselist(self, table: Tag) -> None:
        """One ``sc_courselist`` table: area headings, then Code | Title | Hours rows."""
        under = self.dom_level
        rows: list[tuple[str, str, str]] = []

        def flush() -> None:
            if rows:
                self.blocks.append(Block(BlockKind.TABLE, rows=(TABLE_HEADER, *rows), header_rows=1))
                rows.clear()

        for tr in table.find_all("tr"):
            classes = tr.get("class") or []
            if "hidden" in classes or tr.find("th") is not None:
                continue
            if "areaheader" in classes:
                flush()
                self.heading(_text(tr), under + 1, from_dom=False)
                continue

            hours = _text(tr.find("td", class_="hourscol"))
            code_cell = tr.find("td", class_="codecol")
            if code_cell is not None:
                cells = tr.find_all("td")
                code = _text(code_cell)
                title = _text(cells[1]) if len(cells) > 1 and cells[1] is not code_cell else ""
            else:
                # A comment row: "Complete one of the following:", "Total Hours".
                first = tr.find("td")
                code, title = _text(first), ""
            if not (code or title):
                continue

            if "orclass" in classes and rows:
                prev_code, prev_title, prev_hours = rows[-1]
                alternative = code if code.lower().startswith("or ") else f"or {code}"
                rows[-1] = (
                    f"{prev_code} {alternative}",
                    f"{prev_title} or {title}" if prev_title and title else prev_title or title,
                    prev_hours or hours,
                )
                continue
            if _indented(tr):
                code = f"- {code}"
            rows.append((code, title, hours))
        flush()


def _indented(tr: Tag) -> bool:
    """True for an option row: its first cell LEADS with ``div.blockindent``.

    Not any ``.blockindent`` in the row: the "and CS 2001" of a paired course is
    a ``span.blockindent`` inside an ordinary, unindented row.
    """
    first = tr.find("td")
    lead = next((c for c in first.children if isinstance(c, Tag)), None) if first else None
    return lead is not None and lead.name == "div" and "blockindent" in (lead.get("class") or [])


def tab_blocks(container: Tag) -> list[Block]:
    return _Walker().walk(container)


# --- pages ----------------------------------------------------------------------


def parse_program(url: str, html: str) -> dict | None:
    """The program record for one catalog page, or ``None`` if it is not a program."""
    soup = BeautifulSoup(html, "html.parser")
    tabs = _tabs(soup)
    requirement_tabs = [(tab_id, label) for tab_id, label in tabs if tab_id.endswith(REQUIREMENTS_TAB_SUFFIX)]
    if not requirement_tabs:
        return None

    segments = path_segments(url)
    overview = soup.find(id=OVERVIEW_TAB)

    requirements: list[Block] = []
    for tab_id, label in requirement_tabs:
        if len(requirement_tabs) > 1:
            # Two requirement tabs (a PhD's standard and advanced-entry tracks)
            # need their labels to tell the tracks apart; one tab does not.
            requirements.append(Block(BlockKind.HEADING, text=label, level=1))
        requirements.extend(tab_blocks(soup.find(id=tab_id)))

    record = {
        "slug": slug_for(url),
        "url": url,
        "name": _text(soup.find("h1")),
        "level": segments[0] if segments else "",
        "catalog_college": segments[1] if len(segments) > 1 else "",
        "program_overview": Document(url, HTML, tab_blocks(overview) if overview else []).to_dict(),
        "program_requirements": Document(url, HTML, requirements).to_dict(),
    }
    record["record_hash"] = record_hash(record)
    return record


def record_hash(record: dict) -> str:
    """Fingerprint of everything ingest reads, so an unchanged program is not rewritten."""
    payload = {k: v for k, v in record.items() if k != "record_hash"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return f"sha256:{digest[:32]}"


def sitemap_program_urls(sitemap_xml: str, roots, min_segments: int, skip_segments) -> list[str]:
    """Candidate program URLs from the catalog sitemap, sorted.

    Candidates, not programs: department pages pass this filter too, and are
    only told apart by fetching them (no requirements tab).
    """
    urls = set()
    for el in ET.fromstring(sitemap_xml).iter():
        # Tags arrive namespaced ("{http://www.sitemaps.org/...}loc").
        if not el.tag.endswith("loc"):
            continue
        url = _norm(el.text)
        segments = path_segments(url)
        if (
            len(segments) >= min_segments
            and segments[0] in roots
            and not any(s in skip_segments for s in segments)
        ):
            urls.add(url)
    return sorted(urls)

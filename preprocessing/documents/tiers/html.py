"""Tier 0 for HTML: main content, with its structure kept.

Every HTML path in this repo already runs trafilatura, but for plain text
(``extract(...)``), which flattens a table into a run of cell values and a
heading into just another line. Its XML output keeps what the text output
throws away -- ``<head rend="h2">``, ``<list>``/``<item>``,
``<table>``/``<row>``/``<cell role="head">``, ``<graphic alt=...>`` -- with the
same boilerplate removal, so this reads that instead of writing a DOM walker.

HTML images become figures only through their alt text. Captioning them would
mean fetching every ``src``, which a caller that wants it can do and pass
through the image path of ``DocumentParser``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import trafilatura
from bs4 import BeautifulSoup

from ..ir import Block, BlockKind


def _norm(text: str) -> str:
    return " ".join(text.split())


def _own_text(el: ET.Element, skip: str) -> str:
    """Text of ``el`` without the text of any ``skip`` descendant (e.g. a nested list)."""
    parts = [el.text or ""]
    for child in el:
        if child.tag != skip:
            parts.append("".join(child.itertext()))
        parts.append(child.tail or "")
    return _norm("".join(parts))


def _heading_level(el: ET.Element) -> int:
    rend = el.get("rend", "")
    if len(rend) == 2 and rend[0] == "h" and rend[1] in "123456":
        return int(rend[1])
    return 2


def _list_items(el: ET.Element, depth: int = 0) -> list[str]:
    """Items of a list, nested lists flattened in order and indented by depth."""
    items: list[str] = []
    for item in el.findall("item"):
        text = _own_text(item, skip="list")
        if text:
            items.append("  " * depth + text)
        for nested in item.findall("list"):
            items.extend(_list_items(nested, depth + 1))
    return items


def _table(el: ET.Element) -> Block | None:
    rows: list[tuple[str, ...]] = []
    header_rows = 0
    for row in el.iter("row"):
        cells = row.findall("cell")
        if not cells:
            continue
        rows.append(tuple(_norm("".join(c.itertext())) for c in cells))
        # Only a LEADING run of all-header rows is the header; a header cell
        # further down is a row label, not a column title.
        if len(rows) == header_rows + 1 and all(c.get("role") == "head" for c in cells):
            header_rows += 1
    if not rows:
        return None
    width = max(len(r) for r in rows)
    padded = tuple(r + ("",) * (width - len(r)) for r in rows)
    return Block(BlockKind.TABLE, rows=padded, header_rows=header_rows)


def _page_heading(html: str) -> str:
    """The page's ``<h1>``, which trafilatura drops as boilerplate on some templates."""
    h1 = BeautifulSoup(html, "html.parser").find("h1")
    return _norm(h1.get_text(" ")) if h1 else ""


def parse_html(html: str, source: str = "") -> list[Block]:
    """Blocks of the page's main content, in document order. Empty if it has none."""
    xml = trafilatura.extract(
        html,
        url=source or None,
        output_format="xml",
        include_tables=True,
        include_images=True,
        include_comments=False,
        include_formatting=False,
        favor_recall=True,
    )
    if not xml:
        return []

    main = ET.fromstring(xml).find("main")
    if main is None:
        return []

    blocks: list[Block] = []
    for el in main:
        if el.tag == "head":
            if text := _norm("".join(el.itertext())):
                blocks.append(Block(BlockKind.HEADING, text=text, level=_heading_level(el)))
        elif el.tag == "list":
            if items := _list_items(el):
                blocks.append(Block(BlockKind.LIST, items=tuple(items)))
        elif el.tag == "table":
            if (table := _table(el)) is not None and not table.is_empty():
                blocks.append(table)
        elif el.tag == "graphic":
            if alt := _norm(el.get("alt") or el.get("title") or ""):
                blocks.append(Block(BlockKind.FIGURE, text=alt))
        elif text := _norm("".join(el.itertext())):
            # p, quote, code, and anything trafilatura adds later.
            blocks.append(Block(BlockKind.PARAGRAPH, text=text))

    title = _page_heading(html)
    if title and not any(b.kind is BlockKind.HEADING and b.text == title for b in blocks):
        blocks.insert(0, Block(BlockKind.HEADING, text=title, level=1))
    return blocks

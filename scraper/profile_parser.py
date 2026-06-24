"""Parse a Khoury profile HTML page into a Profile dataclass."""

from __future__ import annotations

from bs4 import BeautifulSoup, NavigableString, ProcessingInstruction, Tag

from models import Profile


def _text(tag: Tag | None) -> str | None:
    if tag is None:
        return None
    text = tag.get_text(strip=True)
    return text or None


def _clean_link_text(a: Tag) -> str:
    """Extract anchor text, ignoring SVG icons and XML processing instructions."""
    parts: list[str] = []
    for child in a.children:
        if isinstance(child, ProcessingInstruction):
            continue
        if isinstance(child, Tag):
            if child.name == "svg":
                continue
            text = child.get_text(" ", strip=True)
            if text:
                parts.append(text)
        elif isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                parts.append(text)
    return " ".join(parts).strip()


def _link_dict(a: Tag) -> dict[str, str]:
    return {"text": _clean_link_text(a), "href": a["href"]}


def _list_items(block: Tag) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for li in block.select("li.single-people__aside-list-item"):
        link = li.find("a", href=True)
        if link:
            items.append(_link_dict(link))
        else:
            text = li.get_text(strip=True)
            if text:
                items.append({"text": text})
    return items


def _parse_aside(soup: BeautifulSoup) -> dict[str, list]:
    aside: dict[str, list] = {}
    for block in soup.select("div.single-people__aside-block"):
        h3 = block.find("h3")
        if not h3:
            continue
        key = h3.get_text(strip=True).lower().replace(" ", "_")

        items = _list_items(block)
        if items:
            aside[key] = items
            continue

        para_links = [
            _link_dict(a)
            for p in block.find_all("p")
            for a in p.find_all("a", href=True)
        ]
        if para_links:
            aside[key] = para_links
            continue

        paragraphs = [
            p.get_text(strip=True)
            for p in block.find_all("p")
            if p.get_text(strip=True)
        ]
        if paragraphs:
            aside[key] = paragraphs
    return aside


def _parse_accordion_sections(soup: BeautifulSoup) -> dict[str, str | list[str]]:
    sections: dict[str, str | list[str]] = {}
    for item in soup.select("div.accordion-item"):
        header = item.find("button", class_="accordion-header")
        body = item.select_one("div.accordion-content__inner")
        if not header or not body:
            continue
        key = header.get_text(strip=True).lower().replace(" ", "_")

        list_items = [
            li.get_text(" ", strip=True)
            for li in body.select("ul.wp-block-list > li")
            if li.get_text(strip=True)
        ]
        if list_items:
            sections[key] = list_items
            continue

        paragraphs = [
            p.get_text(" ", strip=True)
            for p in body.find_all("p")
            if p.get_text(strip=True)
        ]
        if paragraphs:
            sections[key] = "\n\n".join(paragraphs)
    return sections


def parse_profile(url: str, html: str) -> Profile:
    soup = BeautifulSoup(html, "html.parser")
    slug = url.rstrip("/").rsplit("/", 1)[-1]

    name = _text(soup.select_one("h1.single-people__header-title"))
    pronouns = _text(soup.select_one("p.single-people__header-subtitle"))
    title = _text(soup.select_one("p.single-people__header-description"))

    photo_tag = soup.select_one("figure.single-people__header-figure img")
    photo_url = photo_tag.get("src") if photo_tag else None

    roles_text = _text(soup.select_one("p.single-people__aside-roles"))
    roles = [r.strip() for r in roles_text.split(",")] if roles_text else []

    aside = _parse_aside(soup)
    campuses = [item["text"] for item in aside.get("campus", []) if isinstance(item, dict)]
    areas = [item["text"] for item in aside.get("area_of_interest", []) if isinstance(item, dict)]
    websites = [item for item in aside.get("website", []) if isinstance(item, dict)]
    scholar_items = [item for item in aside.get("google_scholar", []) if isinstance(item, dict)]
    google_scholar = scholar_items[0]["href"] if scholar_items else None

    contact_raw = aside.get("contact", [])
    contact: list[str] = []
    for item in contact_raw:
        if isinstance(item, dict):
            contact.append(item["text"])
        else:
            contact.append(item)

    sections = _parse_accordion_sections(soup)

    def _as_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    biography = sections.get("biography")
    if isinstance(biography, list):
        biography = "\n\n".join(biography)

    return Profile(
        slug=slug,
        url=url,
        name=name,
        pronouns=pronouns,
        title=title,
        photo_url=photo_url,
        roles=roles,
        campuses=campuses,
        websites=websites,
        google_scholar=google_scholar,
        areas_of_interest=areas,
        contact=contact,
        research_interests=_as_list(sections.get("research_interests")),
        education=_as_list(sections.get("education")),
        biography=biography,
        labs_and_groups=_as_list(sections.get("labs_and_groups")),
        projects=_as_list(sections.get("projects")),
        raw_aside=aside,
    )

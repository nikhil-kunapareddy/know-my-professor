"""Parse a Khoury profile HTML page into a Profile dataclass."""

from __future__ import annotations

from bs4 import BeautifulSoup, NavigableString, ProcessingInstruction, Tag

from .models import Profile


class ProfileParser:
    """Turns a single faculty profile page's HTML into a ``Profile``.

    Stateless — all helpers are static — so one instance can parse many pages.
    """

    def parse(self, url: str, html: str) -> Profile:
        """Parse one profile page into a Profile dataclass."""
        soup = BeautifulSoup(html, "html.parser")
        slug = url.rstrip("/").rsplit("/", 1)[-1]

        name = self._text(soup.select_one("h1.single-people__header-title"))
        pronouns = self._text(soup.select_one("p.single-people__header-subtitle"))
        title = self._text(soup.select_one("p.single-people__header-description"))

        photo_tag = soup.select_one("figure.single-people__header-figure img")
        photo_url = photo_tag.get("src") if photo_tag else None

        roles_text = self._text(soup.select_one("p.single-people__aside-roles"))
        roles = [r.strip() for r in roles_text.split(",")] if roles_text else []

        aside = self._parse_aside(soup)
        campuses = [i["text"] for i in aside.get("campus", []) if isinstance(i, dict)]
        areas = [i["text"] for i in aside.get("area_of_interest", []) if isinstance(i, dict)]
        websites = [i for i in aside.get("website", []) if isinstance(i, dict)]
        scholar_items = [i for i in aside.get("google_scholar", []) if isinstance(i, dict)]
        google_scholar = scholar_items[0]["href"] if scholar_items else None

        contact: list[str] = []
        for item in aside.get("contact", []):
            contact.append(item["text"] if isinstance(item, dict) else item)

        sections = self._parse_accordion_sections(soup)

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
            research_interests=self._as_list(sections.get("research_interests")),
            education=self._as_list(sections.get("education")),
            biography=biography,
            labs_and_groups=self._as_list(sections.get("labs_and_groups")),
            projects=self._as_list(sections.get("projects")),
            raw_aside=aside,
        )

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _text(tag: Tag | None) -> str | None:
        if tag is None:
            return None
        return tag.get_text(strip=True) or None

    @staticmethod
    def _as_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _clean_link_text(a: Tag) -> str:
        """Anchor text, ignoring SVG icons and XML processing instructions."""
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

    @classmethod
    def _link_dict(cls, a: Tag) -> dict[str, str]:
        return {"text": cls._clean_link_text(a), "href": a["href"]}

    @classmethod
    def _list_items(cls, block: Tag) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for li in block.select("li.single-people__aside-list-item"):
            link = li.find("a", href=True)
            if link:
                items.append(cls._link_dict(link))
            else:
                text = li.get_text(strip=True)
                if text:
                    items.append({"text": text})
        return items

    @classmethod
    def _parse_aside(cls, soup: BeautifulSoup) -> dict[str, list]:
        aside: dict[str, list] = {}
        for block in soup.select("div.single-people__aside-block"):
            h3 = block.find("h3")
            if not h3:
                continue
            key = h3.get_text(strip=True).lower().replace(" ", "_")

            items = cls._list_items(block)
            if items:
                aside[key] = items
                continue

            para_links = [
                cls._link_dict(a)
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

    @staticmethod
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

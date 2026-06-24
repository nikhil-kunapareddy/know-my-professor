"""Data model for a scraped faculty profile."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Profile:
    slug: str
    url: str
    name: str | None
    pronouns: str | None
    title: str | None
    photo_url: str | None
    roles: list[str]
    campuses: list[str]
    websites: list[dict[str, str]]
    google_scholar: str | None
    areas_of_interest: list[str]
    contact: list[str]
    research_interests: list[str]
    education: list[str]
    biography: str | None
    labs_and_groups: list[str]
    projects: list[str]
    raw_aside: dict[str, list]

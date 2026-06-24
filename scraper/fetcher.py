"""HTTP session, page fetching, and profile-URL discovery."""

from __future__ import annotations

import re
import time

import requests
from bs4 import BeautifulSoup

from config import (
    LISTING,
    PROFILE_URL_RE,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def extract_total_pages(listing_html: str) -> int:
    soup = BeautifulSoup(listing_html, "html.parser")
    page_numbers: list[int] = []
    for a in soup.select("a.page-numbers"):
        href = a.get("href", "")
        m = re.search(r"/page/(\d+)/?", href)
        if m:
            page_numbers.append(int(m.group(1)))
    return max(page_numbers) if page_numbers else 1


def extract_profile_urls(listing_html: str) -> set[str]:
    soup = BeautifulSoup(listing_html, "html.parser")
    urls: set[str] = set()
    for a in soup.find_all("a", href=True):
        if PROFILE_URL_RE.match(a["href"]):
            urls.add(a["href"])
    return urls


def discover_all_profile_urls(session: requests.Session) -> list[str]:
    print("Discovering profile URLs...")
    first_page_html = fetch(session, LISTING)
    total_pages = extract_total_pages(first_page_html)
    print(f"  Found {total_pages} listing pages")

    all_urls: set[str] = set()
    all_urls |= extract_profile_urls(first_page_html)

    for page in range(2, total_pages + 1):
        time.sleep(REQUEST_DELAY_SECONDS)
        html = fetch(session, f"{LISTING}page/{page}/")
        new = extract_profile_urls(html)
        all_urls |= new
        print(f"  page {page}/{total_pages} -> {len(new)} on this page, {len(all_urls)} total")

    return sorted(all_urls)

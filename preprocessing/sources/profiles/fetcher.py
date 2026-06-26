"""HTTP session, page fetching, and profile-URL discovery."""

from __future__ import annotations

import re
import time

import requests
from bs4 import BeautifulSoup

from shared.config import (
    KHOURY_LISTING,
    PROFILE_URL_RE,
    SCRAPER_REQUEST_DELAY_SECONDS,
    SCRAPER_REQUEST_TIMEOUT_SECONDS,
    SCRAPER_USER_AGENT,
)


class DirectoryFetcher:
    """Fetches Khoury directory pages and discovers profile URLs.

    Holds the HTTP session; the listing-parsing helpers are stateless and
    exposed as static methods so they can be unit-tested without a network.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": SCRAPER_USER_AGENT})

    def fetch(self, url: str) -> str:
        """GET ``url`` and return its decoded body, raising on HTTP errors."""
        response = self.session.get(url, timeout=SCRAPER_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.text

    @staticmethod
    def extract_total_pages(listing_html: str) -> int:
        """Highest /page/N/ number in the listing pagination (1 if none)."""
        soup = BeautifulSoup(listing_html, "html.parser")
        page_numbers: list[int] = []
        for a in soup.select("a.page-numbers"):
            m = re.search(r"/page/(\d+)/?", a.get("href", ""))
            if m:
                page_numbers.append(int(m.group(1)))
        return max(page_numbers) if page_numbers else 1

    @staticmethod
    def extract_profile_urls(listing_html: str) -> set[str]:
        """Profile-page URLs on a listing page (matches the strict slug pattern)."""
        soup = BeautifulSoup(listing_html, "html.parser")
        return {
            a["href"]
            for a in soup.find_all("a", href=True)
            if PROFILE_URL_RE.match(a["href"])
        }

    def discover_all_profile_urls(self) -> list[str]:
        """Walk every listing page and return the sorted set of profile URLs."""
        print("Discovering profile URLs...")
        first_page_html = self.fetch(KHOURY_LISTING)
        total_pages = self.extract_total_pages(first_page_html)
        print(f"  Found {total_pages} listing pages")

        all_urls: set[str] = set(self.extract_profile_urls(first_page_html))
        for page in range(2, total_pages + 1):
            time.sleep(SCRAPER_REQUEST_DELAY_SECONDS)
            html = self.fetch(f"{KHOURY_LISTING}page/{page}/")
            new = self.extract_profile_urls(html)
            all_urls |= new
            print(f"  page {page}/{total_pages} -> {len(new)} on this page, {len(all_urls)} total")

        return sorted(all_urls)

"""Shallow, robots-respecting crawl of a single faculty website.

Fetches the homepage plus a few same-host one-hop links (research, publications,
bio, ...). Uses plain ``requests`` — no headless browser — so JS-only pages
return little; the extract step's success guard handles that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from shared.config import (
    MAX_PAGES_PER_PROFESSOR,
    ONE_HOP_KEYWORDS,
    WEBLINKS_REQUEST_DELAY_SECONDS,
    WEBLINKS_REQUEST_TIMEOUT_SECONDS,
    WEBLINKS_USER_AGENT,
)


@dataclass
class CrawlResult:
    """Outcome of crawling one site.

    ``pages`` holds (url, html) on success; ``reason`` is a short, stable failure
    code when the homepage couldn't be crawled — "robots_disallowed",
    "http_404", "non_html", "timeout", "conn_error:<Type>".
    """

    pages: list[tuple[str, str]] = field(default_factory=list)
    reason: str | None = None


class SiteCrawler:
    """Crawls a faculty homepage plus a few same-host one-hop pages."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": WEBLINKS_USER_AGENT})

    def crawl(self, start_url: str) -> CrawlResult:
        """Crawl the homepage and a few one-hop pages.

        Returns a CrawlResult: ``.pages`` populated on success, or ``.reason``
        set when the homepage can't be crawled. One-hop pages that fail are
        silently skipped — only the homepage is fatal.
        """
        start_url = start_url.rstrip("/") or start_url
        rp = self._robots_for(self.session, start_url)
        if not rp.can_fetch(WEBLINKS_USER_AGENT, start_url):
            return CrawlResult(reason="robots_disallowed")

        homepage, reason = self._fetch(self.session, start_url)
        if homepage is None:
            return CrawlResult(reason=reason)
        pages: list[tuple[str, str]] = [(start_url, homepage)]

        for link in self._select_one_hop_links(homepage, start_url):
            if len(pages) >= MAX_PAGES_PER_PROFESSOR:
                break
            if link == start_url or not rp.can_fetch(WEBLINKS_USER_AGENT, link):
                continue
            time.sleep(WEBLINKS_REQUEST_DELAY_SECONDS)
            html, _ = self._fetch(self.session, link)
            if html is not None:
                pages.append((link, html))

        return CrawlResult(pages=pages)

    # --- helpers (stateless; take the session explicitly for testability) ---

    @staticmethod
    def _robots_for(session: requests.Session, base_url: str) -> RobotFileParser:
        parsed = urlparse(base_url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        try:
            resp = session.get(robots_url, timeout=WEBLINKS_REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                rp.allow_all = True
        except requests.RequestException:
            rp.allow_all = True
        return rp

    @staticmethod
    def _fetch(session: requests.Session, url: str) -> tuple[str | None, str | None]:
        """Return (html, None) on success, or (None, reason) describing the failure."""
        try:
            resp = session.get(url, timeout=WEBLINKS_REQUEST_TIMEOUT_SECONDS)
        except requests.Timeout:
            return None, "timeout"
        except requests.RequestException as e:
            return None, f"conn_error:{type(e).__name__}"
        content_type = resp.headers.get("Content-Type", "").lower()
        if resp.status_code >= 400:
            return None, f"http_{resp.status_code}"
        if "html" not in content_type:
            return None, "non_html"
        # When the server omits a charset, requests defaults to ISO-8859-1 and
        # mangles UTF-8 text. Fall back to detection.
        if "charset" not in content_type:
            resp.encoding = resp.apparent_encoding
        return resp.text, None

    @staticmethod
    def _select_one_hop_links(homepage_html: str, base_url: str) -> list[str]:
        """Same-host links whose href or anchor text matches a keyword."""
        soup = BeautifulSoup(homepage_html, "html.parser")
        base_host = urlparse(base_url).netloc
        seen: set[str] = set()
        out: list[str] = []
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"]).split("#", 1)[0].rstrip("/")
            if not href or href in seen:
                continue
            parsed = urlparse(href)
            if parsed.scheme not in ("http", "https") or parsed.netloc != base_host:
                continue
            haystack = (href + " " + a.get_text(" ", strip=True)).lower()
            if any(kw in haystack for kw in ONE_HOP_KEYWORDS):
                seen.add(href)
                out.append(href)
        return out

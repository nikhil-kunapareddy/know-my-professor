"""Shallow, robots-respecting crawl of a single faculty website.

Fetches the homepage plus a few same-host one-hop links (research, publications,
bio, ...). Uses plain `requests` — no headless browser — so JS-only pages return
little; the extract step's success guard handles that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from config import (
    MAX_PAGES_PER_PROFESSOR,
    ONE_HOP_KEYWORDS,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)


@dataclass
class CrawlResult:
    pages: list[tuple[str, str]] = field(default_factory=list)
    # Short, stable reason the homepage couldn't be crawled (None on success):
    # "robots_disallowed", "http_404", "non_html", "timeout", "conn_error:<Type>".
    reason: str | None = None


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _robots_for(session: requests.Session, base_url: str) -> RobotFileParser:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        resp = session.get(robots_url, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            rp.allow_all = True
    except requests.RequestException:
        rp.allow_all = True
    return rp


def _fetch(session: requests.Session, url: str) -> tuple[str | None, str | None]:
    """Return (html, None) on success, or (None, reason) describing the failure."""
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
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
    # mangles UTF-8 text (e.g. curly apostrophes). Fall back to detection.
    if "charset" not in content_type:
        resp.encoding = resp.apparent_encoding
    return resp.text, None


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


def crawl_site(session: requests.Session, start_url: str) -> CrawlResult:
    """Crawl the homepage and a few one-hop pages.

    Returns a CrawlResult: `.pages` populated on success, or `.reason` set when
    the homepage can't be crawled (robots block, HTTP error, non-HTML, network).
    One-hop pages that fail are silently skipped — only the homepage is fatal.
    """
    start_url = start_url.rstrip("/") or start_url
    rp = _robots_for(session, start_url)
    if not rp.can_fetch(USER_AGENT, start_url):
        return CrawlResult(reason="robots_disallowed")

    homepage, reason = _fetch(session, start_url)
    if homepage is None:
        return CrawlResult(reason=reason)
    pages: list[tuple[str, str]] = [(start_url, homepage)]

    for link in _select_one_hop_links(homepage, start_url):
        if len(pages) >= MAX_PAGES_PER_PROFESSOR:
            break
        if link == start_url or not rp.can_fetch(USER_AGENT, link):
            continue
        time.sleep(REQUEST_DELAY_SECONDS)
        html, _ = _fetch(session, link)
        if html is not None:
            pages.append((link, html))

    return CrawlResult(pages=pages)

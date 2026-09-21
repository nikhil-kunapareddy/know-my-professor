"""
Faculty directory scraper (entrypoint).

Walks one or more college directories and writes structured JSON per professor
to ./data/profiles/<entity-id>.json (local) or gs://<bucket>/profiles/<entity-id>.json
(GCS), plus a per-college URL list.

Khoury profiles keep bare ``<slug>`` filenames and vector ids; every other
college is namespaced ``<college>-<slug>`` (see ``ProfileSource.entity_id``).

Usage:
    python -m preprocessing.sources.profiles.runner                        # every college
    python -m preprocessing.sources.profiles.runner --college cos          # just one
    python -m preprocessing.sources.profiles.runner --college cos --limit 5
    python -m preprocessing.sources.profiles.runner --urls-only            # discovery only
    python -m preprocessing.sources.profiles.runner --gcs-bucket BUCKET    # write to GCS
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from shared.config import gcs_bucket
from shared.gcs import GCSStore, LocalStore, OutputStore

from ..entities import DEFAULT_COLLEGE
from .config import COLLEGES, COLLEGES_BY_KEY, LOCAL_OUTPUT_DIR, College
from .fetcher import DirectoryFetcher
from .llm_parser import LlmProfileParser, ThinProfilePage
from .profile_parser import ProfileParser
from .source import ProfileSource

#: Concurrent extraction workers. Fetching stays serialized behind the college's
#: crawl delay regardless -- this only overlaps the LLM calls, which are the
#: slow half for llm-parsed colleges. Anthropic allows thousands of requests per
#: minute, so the ceiling here is politeness to the directory, not the API.
DEFAULT_WORKERS = 8


class _Pacer:
    """Serializes fetches to one host, holding ``delay`` between them.

    A plain ``sleep`` inside each worker would let N workers hit the directory
    at once and merely stagger the next round. Holding the lock across the sleep
    is what actually caps the rate at one request per ``delay``, no matter how
    many workers are running.
    """

    def __init__(self, delay: float):
        self.delay = delay
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                time.sleep(self._next_at - now)
            self._next_at = time.monotonic() + self.delay


class ProfileScraper:
    """Discovers profile URLs and scrapes each into a JSON record in the store."""

    #: Where records land; the source owns the layout, not the scraper.
    prefix = ProfileSource.prefix

    def __init__(
        self,
        fetcher: DirectoryFetcher,
        parser: ProfileParser | LlmProfileParser,
        store: OutputStore,
        college: College | None = None,
        workers: int = DEFAULT_WORKERS,
    ):
        self.fetcher = fetcher
        self.parser = parser
        self.store = store
        self.college = college or fetcher.college
        self.workers = max(1, workers)

    def record_key(self, slug: str) -> str:
        """Filename stem for a slug: namespaced unless this is the base college."""
        if self.college.key == DEFAULT_COLLEGE:
            return slug
        return f"{self.college.key}-{slug}"

    def scrape(self, urls: list[str], limit: int | None = None) -> tuple[int, int]:
        """Scrape each URL, skipping records already present. Returns (written, failed)."""
        if limit is not None:
            urls = urls[:limit]

        existing = self.store.existing_slugs(self.prefix)
        print(f"  ({len(existing)} profiles already present in {self.store.describe()})")

        pending = [u for u in urls if self.record_key(u.rstrip("/").rsplit("/", 1)[-1]) not in existing]
        print(f"  {len(urls) - len(pending)} already stored, {len(pending)} to fetch")
        if not pending:
            return 0, 0

        pacer = _Pacer(self.college.crawl_delay)
        written = failed = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._scrape_one, url, pacer): url for url in pending}
            for i, future in enumerate(as_completed(futures), 1):
                url = futures[future]
                try:
                    name = future.result()
                except ThinProfilePage as e:
                    failed += 1
                    print(f"  [{i}/{len(pending)}] skip (thin page) {url}: {e}")
                except Exception as e:
                    failed += 1
                    print(f"  [{i}/{len(pending)}] FAILED {url}: {e}")
                else:
                    written += 1
                    print(f"  [{i}/{len(pending)}] {name}")
        return written, failed

    def _scrape_one(self, url: str, pacer: _Pacer) -> str:
        """Fetch (paced) and parse (concurrent) one profile, then store it."""
        pacer.wait()
        html = self.fetcher.fetch(url)

        profile = self.parser.parse(url, html)
        record = asdict(profile)
        # Stamped on every record so ``ProfileSource.entity_id`` can namespace
        # the vector id without re-deriving it from the URL at ingest time.
        record["college"] = self.college.key

        key = self.record_key(profile.slug)
        self.store.write_text(
            f"{self.prefix}{key}.json", json.dumps(record, indent=2, ensure_ascii=False)
        )
        return profile.name or key


def build_parser(college: College) -> ProfileParser | LlmProfileParser:
    """The parser a college's profile template needs."""
    return ProfileParser() if college.parser == "accordion" else LlmProfileParser()


def _build_store(bucket: str | None) -> OutputStore:
    if bucket:
        return GCSStore(bucket)
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return LocalStore(LOCAL_OUTPUT_DIR)


def _urls_key(college: College) -> str:
    """Cache key for a college's discovered URL list.

    Khoury keeps the original un-suffixed filename so the list already in the
    bucket is still found instead of silently re-discovered.
    """
    if college.key == DEFAULT_COLLEGE:
        return "profile_urls.json"
    return f"profile_urls_{college.key}.json"


def _resolve_colleges(requested: str | None) -> list[College]:
    if not requested:
        return list(COLLEGES)
    keys = [k.strip() for k in requested.split(",") if k.strip()]
    unknown = [k for k in keys if k not in COLLEGES_BY_KEY]
    if unknown:
        sys.exit(f"error: unknown college(s) {unknown}; known: {sorted(COLLEGES_BY_KEY)}")
    return [COLLEGES_BY_KEY[k] for k in keys]


def run_college(college: College, store: OutputStore, args) -> tuple[int, int]:
    """Discover and scrape one college. Returns (written, failed)."""
    print(f"\n=== {college.key} ({college.base}) — {college.parser} parser ===")
    fetcher = DirectoryFetcher(college)

    cached = store.read_text(_urls_key(college))
    if cached and not args.refresh_urls:
        urls = json.loads(cached)
        print(f"Using cached URL list ({len(urls)} entries) — pass --refresh-urls to rediscover.")
    else:
        urls = fetcher.discover_all_profile_urls()
        store.write_text(_urls_key(college), json.dumps(urls, indent=2))
        print(f"Saved {len(urls)} profile URLs to {store.describe()}/{_urls_key(college)}")

    if args.urls_only:
        return 0, 0

    scraper = ProfileScraper(fetcher, build_parser(college), store, college, args.workers)
    return scraper.scrape(urls, limit=args.limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--college", default=None,
        help=f"Comma-separated college keys (default: all). Known: {', '.join(c.key for c in COLLEGES)}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only scrape first N profiles per college")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Concurrent extraction workers (fetching stays paced per college)",
    )
    parser.add_argument(
        "--urls-only", action="store_true",
        help="Discover and save profile URLs but skip profile fetching",
    )
    parser.add_argument(
        "--refresh-urls", action="store_true",
        help="Re-discover profile URLs even if a cached list exists",
    )
    parser.add_argument(
        "--gcs-bucket", default=gcs_bucket(),
        help="If set (or env KMP_GCS_BUCKET), write outputs to gs://BUCKET/ instead of ./data/",
    )
    args = parser.parse_args()

    colleges = _resolve_colleges(args.college)
    store = _build_store(args.gcs_bucket)
    print(f"Output store: {store.describe()}")
    print(f"Colleges: {', '.join(c.key for c in colleges)}")

    totals = [0, 0]
    for college in colleges:
        written, failed = run_college(college, store, args)
        totals[0] += written
        totals[1] += failed

    print(f"\nDone. {totals[0]} profile(s) written, {totals[1]} failed/skipped.")


if __name__ == "__main__":
    main()

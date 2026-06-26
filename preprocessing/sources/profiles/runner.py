"""
Khoury faculty directory scraper (entrypoint).

Writes structured JSON per professor either to ./data/profiles/<slug>.json
(local) or to gs://<bucket>/profiles/<slug>.json (GCS), plus a master URL list.

Usage:
    python -m preprocessing.sources.profiles.runner                       # local, everyone
    python -m preprocessing.sources.profiles.runner --limit 5             # first 5 profiles
    python -m preprocessing.sources.profiles.runner --urls-only           # only discover URLs
    python -m preprocessing.sources.profiles.runner --gcs-bucket BUCKET   # write to GCS
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from shared.config import LOCAL_OUTPUT_DIR, SCRAPER_REQUEST_DELAY_SECONDS, gcs_bucket
from shared.gcs import GCSStore, LocalStore, OutputStore

from .fetcher import DirectoryFetcher
from .profile_parser import ProfileParser


class ProfileScraper:
    """Discovers profile URLs and scrapes each into a JSON record in the store."""

    def __init__(self, fetcher: DirectoryFetcher, parser: ProfileParser, store: OutputStore):
        self.fetcher = fetcher
        self.parser = parser
        self.store = store

    def scrape(self, urls: list[str], limit: int | None = None) -> None:
        """Scrape each URL, skipping slugs already present in the store."""
        if limit is not None:
            urls = urls[:limit]

        existing = self.store.existing_profile_slugs()
        print(f"  ({len(existing)} profiles already present in {self.store.describe()})")

        for i, url in enumerate(urls, 1):
            slug = url.rstrip("/").rsplit("/", 1)[-1]
            if slug in existing:
                print(f"  [{i}/{len(urls)}] skip (exists): {slug}")
                continue

            time.sleep(SCRAPER_REQUEST_DELAY_SECONDS)
            try:
                html = self.fetcher.fetch(url)
                profile = self.parser.parse(url, html)
                payload = json.dumps(asdict(profile), indent=2, ensure_ascii=False)
                self.store.write_text(f"profiles/{slug}.json", payload)
                print(f"  [{i}/{len(urls)}] {profile.name or slug}")
            except Exception as e:
                print(f"  [{i}/{len(urls)}] FAILED {slug}: {e}")


def _build_store(bucket: str | None) -> OutputStore:
    if bucket:
        return GCSStore(bucket)
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return LocalStore(LOCAL_OUTPUT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only scrape first N profiles")
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

    fetcher = DirectoryFetcher()
    store = _build_store(args.gcs_bucket)
    print(f"Output store: {store.describe()}")

    cached_urls_json = store.read_text("profile_urls.json")
    if cached_urls_json and not args.refresh_urls:
        urls = json.loads(cached_urls_json)
        print(
            f"Using cached profile URL list ({len(urls)} entries) "
            "— pass --refresh-urls to rediscover."
        )
    else:
        urls = fetcher.discover_all_profile_urls()
        store.write_text("profile_urls.json", json.dumps(urls, indent=2))
        print(f"Saved {len(urls)} profile URLs to {store.describe()}/profile_urls.json")

    if args.urls_only:
        return

    print(f"\nScraping profiles (limit={args.limit})...")
    ProfileScraper(fetcher, ProfileParser(), store).scrape(urls, limit=args.limit)
    print("Done.")


if __name__ == "__main__":
    main()

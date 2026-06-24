"""
Khoury faculty directory scraper.

Writes structured JSON per professor either to ./data/profiles/<slug>.json
(local) or to gs://<bucket>/profiles/<slug>.json (GCS), plus a master URL list.

Usage:
    python scraper.py                                # local output, scrape everyone
    python scraper.py --limit 5                      # scrape first 5 profiles
    python scraper.py --urls-only                    # only discover URLs
    python scraper.py --gcs-bucket know-my-professor-raw   # write to GCS

Module layout:
    config.py          shared constants
    models.py          Profile dataclass
    fetcher.py         HTTP session, fetching, URL discovery
    profile_parser.py  HTML -> Profile
    stores.py          LocalStore / GCSStore
    scraper.py         orchestration (this file)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

from config import OUTPUT_DIR, REQUEST_DELAY_SECONDS
from fetcher import discover_all_profile_urls, fetch, make_session
from profile_parser import parse_profile
from stores import GCSStore, LocalStore, OutputStore


def scrape_profiles(
    session,
    urls: list[str],
    store: OutputStore,
    limit: int | None,
) -> None:
    if limit is not None:
        urls = urls[:limit]

    existing = store.existing_profile_slugs()
    print(f"  ({len(existing)} profiles already present in {store.describe()})")

    for i, url in enumerate(urls, 1):
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        if slug in existing:
            print(f"  [{i}/{len(urls)}] skip (exists): {slug}")
            continue

        time.sleep(REQUEST_DELAY_SECONDS)
        try:
            html = fetch(session, url)
            profile = parse_profile(url, html)
            payload = json.dumps(asdict(profile), indent=2, ensure_ascii=False)
            store.write_text(f"profiles/{slug}.json", payload)
            print(f"  [{i}/{len(urls)}] {profile.name or slug}")
        except Exception as e:
            print(f"  [{i}/{len(urls)}] FAILED {slug}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only scrape first N profiles")
    parser.add_argument(
        "--urls-only",
        action="store_true",
        help="Discover and save profile URLs but skip profile fetching",
    )
    parser.add_argument(
        "--refresh-urls",
        action="store_true",
        help="Re-discover profile URLs even if cached list exists",
    )
    parser.add_argument(
        "--gcs-bucket",
        default=os.environ.get("KMP_GCS_BUCKET"),
        help="If set (or env KMP_GCS_BUCKET), write outputs to gs://BUCKET/ instead of ./data/",
    )
    args = parser.parse_args()

    session = make_session()
    if args.gcs_bucket:
        store: OutputStore = GCSStore(args.gcs_bucket)
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        store = LocalStore(OUTPUT_DIR)
    print(f"Output store: {store.describe()}")

    cached_urls_json = store.read_text("profile_urls.json")
    if cached_urls_json and not args.refresh_urls:
        urls = json.loads(cached_urls_json)
        print(
            f"Using cached profile URL list ({len(urls)} entries) "
            "— pass --refresh-urls to rediscover."
        )
    else:
        urls = discover_all_profile_urls(session)
        store.write_text("profile_urls.json", json.dumps(urls, indent=2))
        print(f"Saved {len(urls)} profile URLs to {store.describe()}/profile_urls.json")

    if args.urls_only:
        return

    print(f"\nScraping profiles (limit={args.limit})...")
    scrape_profiles(session, urls, store, limit=args.limit)
    print("Done.")


if __name__ == "__main__":
    main()

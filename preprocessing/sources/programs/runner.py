"""
Northeastern program requirements scraper (entrypoint).

Reads the catalog sitemap, fetches every candidate program page (~1,600, 1s
apart, so ~35 minutes), keeps the ones with a requirements tab, and writes one
record per program to gs://<bucket>/programs/<slug>.json. A program whose
``record_hash`` matches what is stored is not rewritten -- the catalog changes
about once a year, and restating ~1,000 unchanged records monthly would spend
GCS's free write quota on nothing.

Usage:
    python -m preprocessing.sources.programs.runner --dry-run --limit 20
    python -m preprocessing.sources.programs.runner --url https://catalog.northeastern.edu/undergraduate/computer-information-science/computer-science/minor/
    python -m preprocessing.sources.programs.runner --gcs-bucket BUCKET
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import requests

from shared.config import gcs_bucket
from shared.gcs import GCSStore, LocalStore, OutputStore
from shared.retry import with_backoff

from ..courses.catalog import CatalogFetcher
from ..courses.config import REQUEST_DELAY_SECONDS
from ..profiles.config import LOCAL_OUTPUT_DIR
from .config import MIN_PATH_SEGMENTS, PROGRAM_ROOTS, SITEMAP_URL, SKIP_SEGMENTS
from .courseleaf import parse_program, sitemap_program_urls
from .source import ProgramSource


def _build_store(bucket: str | None) -> OutputStore:
    if bucket:
        return GCSStore(bucket)
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return LocalStore(LOCAL_OUTPUT_DIR)


#: A full crawl is ~1,400 requests; the first one (2026-09-29) lost 3 to
#: "Remote end closed connection without response", each fine on a re-request.
FETCH_ATTEMPTS = 3


def is_transient(exc: BaseException) -> bool:
    """A dropped connection, a timeout, a 429 or a 5xx -- worth one more try."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    response = getattr(exc, "response", None)
    return isinstance(exc, requests.HTTPError) and response is not None and (
        response.status_code == 429 or response.status_code >= 500
    )


def discover(fetcher: CatalogFetcher) -> list[str]:
    return sitemap_program_urls(fetcher.fetch(SITEMAP_URL), PROGRAM_ROOTS, MIN_PATH_SEGMENTS, SKIP_SEGMENTS)


def scrape(
    fetcher: CatalogFetcher,
    store: OutputStore,
    urls: list[str],
    *,
    delay: float = REQUEST_DELAY_SECONDS,
    dry_run: bool = False,
    existing_hashes: dict[str, str] | None = None,
    sleep=time.sleep,
) -> dict[str, int]:
    """Fetch each URL and store the programs among them. Returns counts by outcome."""
    existing_hashes = existing_hashes or {}
    counts = {"written": 0, "unchanged": 0, "not_program": 0, "failed": 0}
    for i, url in enumerate(urls, 1):
        if i > 1:
            sleep(delay)
        try:
            html = with_backoff(
                lambda url=url: fetcher.fetch(url),
                is_retryable=is_transient,
                max_attempts=FETCH_ATTEMPTS,
                initial_delay=5.0,
                label="catalog fetch",
                sleep=sleep,
            )
            record = parse_program(url, html)
        except Exception as e:  # noqa: BLE001 - one bad page must not end the run
            counts["failed"] += 1
            print(f"  [{i}/{len(urls)}] FAILED {url}: {type(e).__name__}: {e}")
            continue
        if record is None:
            counts["not_program"] += 1
            continue
        if existing_hashes.get(record["slug"]) == record["record_hash"]:
            counts["unchanged"] += 1
            continue
        if not dry_run:
            store.write_text(
                f"{ProgramSource.prefix}{record['slug']}.json",
                json.dumps(record, indent=2, ensure_ascii=False),
            )
        counts["written"] += 1
        print(f"  [{i}/{len(urls)}] {record['name'] or record['slug']}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=None, help="Comma-separated page URLs (default: the sitemap)")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N candidates")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report; write nothing")
    parser.add_argument("--force", action="store_true", help="Rewrite every program, even if unchanged")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS, help="Seconds between requests")
    parser.add_argument(
        "--gcs-bucket", default=gcs_bucket(),
        help="If set (or env KMP_GCS_BUCKET), write to gs://BUCKET/ instead of ./data/",
    )
    args = parser.parse_args()

    fetcher = CatalogFetcher(delay=args.delay)
    store = _build_store(args.gcs_bucket)
    print(f"Output store: {store.describe()}")

    urls = [u.strip() for u in args.url.split(",") if u.strip()] if args.url else discover(fetcher)
    print(f"{len(urls)} candidate page(s)")
    if args.limit is not None:
        urls = urls[: args.limit]
    if not urls:
        sys.exit("error: no candidate pages")

    existing = {} if args.force else store.load_hashes(ProgramSource.prefix, "record_hash")
    if existing:
        print(f"{len(existing)} program(s) already stored — unchanged ones will be skipped")

    counts = scrape(fetcher, store, urls, delay=args.delay, dry_run=args.dry_run, existing_hashes=existing)
    verb = "would be written (dry run)" if args.dry_run else "written"
    print(
        f"\nDone. {counts['written']} program(s) {verb}, {counts['unchanged']} unchanged, "
        f"{counts['not_program']} page(s) not a program, {counts['failed']} failed."
    )
    if counts["failed"] and not (counts["written"] or counts["unchanged"]):
        sys.exit("error: every page failed")


if __name__ == "__main__":
    main()

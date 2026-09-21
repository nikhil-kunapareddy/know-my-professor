"""
Northeastern course catalog scraper (entrypoint).

Walks every subject the catalog publishes (~227) and writes one JSON record per
course to gs://<bucket>/courses/<slug>.json.

Unlike the faculty directories this needs no LLM: CourseLeaf renders every
course in the same ``div.courseblock`` structure, so ~6,500 courses parse
deterministically and for free.

Usage:
    python -m preprocessing.sources.courses.runner --dry-run --limit 3
    python -m preprocessing.sources.courses.runner --subject cs
    python -m preprocessing.sources.courses.runner --gcs-bucket BUCKET
"""

from __future__ import annotations

import argparse
import json
import sys

from shared.config import gcs_bucket
from shared.gcs import GCSStore, LocalStore, OutputStore

from ..profiles.config import LOCAL_OUTPUT_DIR
from .catalog import CatalogFetcher
from .config import MIN_DESCRIPTION_CHARS, REQUEST_DELAY_SECONDS
from .source import CourseSource


def _build_store(bucket: str | None) -> OutputStore:
    if bucket:
        return GCSStore(bucket)
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return LocalStore(LOCAL_OUTPUT_DIR)


def scrape(
    fetcher: CatalogFetcher,
    store: OutputStore,
    subjects: list[str],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Fetch each subject and store its courses. Returns (written, skipped)."""
    prefix = CourseSource.prefix
    written = skipped = 0

    for i, subject in enumerate(subjects, 1):
        try:
            records = fetcher.fetch_subject(subject)
        except Exception as e:
            print(f"  [{i}/{len(subjects)}] FAILED {subject}: {e}")
            continue

        kept = 0
        for record in records:
            # Placeholder entries carry a title and nothing else. Storing them
            # would only give ingest something to filter out later, and they
            # would inflate every count in between.
            if len(record.get("description") or "") < MIN_DESCRIPTION_CHARS:
                skipped += 1
                continue
            if not dry_run:
                store.write_text(
                    f"{prefix}{record['slug']}.json",
                    json.dumps(record, indent=2, ensure_ascii=False),
                )
            kept += 1
        written += kept
        print(f"  [{i}/{len(subjects)}] {subject}: {len(records)} parsed, {kept} stored")

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default=None, help="Comma-separated subject codes (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N subjects")
    parser.add_argument("--dry-run", action="store_true", help="Parse and report; write nothing")
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY_SECONDS, help="Seconds between catalog requests"
    )
    parser.add_argument(
        "--gcs-bucket", default=gcs_bucket(),
        help="If set (or env KMP_GCS_BUCKET), write to gs://BUCKET/ instead of ./data/",
    )
    args = parser.parse_args()

    fetcher = CatalogFetcher(delay=args.delay)
    store = _build_store(args.gcs_bucket)
    print(f"Output store: {store.describe()}")

    if args.subject:
        subjects = [s.strip() for s in args.subject.split(",") if s.strip()]
    else:
        subjects = fetcher.discover_subjects()
        print(f"Discovered {len(subjects)} subjects")
    if args.limit is not None:
        subjects = subjects[: args.limit]

    if not subjects:
        sys.exit("error: no subjects to scrape")

    print(f"\nScraping {len(subjects)} subject(s){' (dry run)' if args.dry_run else ''}...")
    written, skipped = scrape(fetcher, store, subjects, dry_run=args.dry_run)
    print(f"\nDone. {written} course(s) stored, {skipped} skipped (no usable description).")


if __name__ == "__main__":
    main()

"""
Crawl each Khoury faculty member's personal/lab website and extract structured
sections, stored under gs://$KMP_GCS_BUCKET/weblinks/{slug}.json for ingest.

Pipeline per profile that has a `website` link:
  - crawl homepage + a few same-host one-hop pages (requests, robots-respecting)
  - trafilatura -> clean text; SUCCESS GUARD: skip (don't cache) if too short
  - hash the clean text (mixing in SCHEMA_VERSION); if unchanged since last run,
    SKIP the Gemini call (resumable + cheap)
  - else Gemini extracts {website_summary, current_projects, recent_publications,
    students_or_lab_members, recent_news} and we write weblinks/{slug}.json

Re-extraction is forced when a site's text changes OR when SCHEMA_VERSION bumps.

Required env:
  KMP_GCS_BUCKET     — source/destination bucket (e.g. know-my-professor-raw)
  GEMINI_API_KEY     — Google AI Studio API key (not needed for --dry-run)
Usage:
  python weblinks.py --dry-run --limit 5   # crawl + report, no Gemini, no writes
  python weblinks.py --limit 5             # real run on 5 profiles
  python weblinks.py                       # full run
  python weblinks.py --force               # ignore hash skip, re-extract all
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import google.generativeai as genai

from config import SCHEMA_VERSION, SECTION_TYPES
from crawl import crawl_site, make_session
from extract import (
    clean_pages,
    extract_structured,
    extraction_succeeded,
    page_hash,
)
from gcs import (
    load_existing_page_hashes,
    load_profiles_from_gcs,
    write_weblinks_record,
)


def website_url(profile: dict) -> str | None:
    """First http(s) website link on the profile, if any."""
    for site in profile.get("websites") or []:
        href = (site or {}).get("href", "")
        if href.startswith("http"):
            return href
    return None


def build_record(profile: dict, source_url: str, ph: str, extracted: dict) -> dict:
    sections = []
    for section_type in SECTION_TYPES:
        value = extracted.get(section_type)
        if value:  # drop empty strings / empty lists
            sections.append(
                {"section_type": section_type, "text": value, "source_url": source_url}
            )
    return {
        "slug": profile.get("slug"),
        "professor_name": profile.get("name") or "",
        "professor_title": profile.get("title") or "",
        "source_url": source_url,
        "schema_version": SCHEMA_VERSION,
        "page_hash": ph,
        "sections": sections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N profiles")
    parser.add_argument("--dry-run", action="store_true", help="Crawl + report; no Gemini, no writes")
    parser.add_argument("--force", action="store_true", help="Ignore hash skip; re-extract all")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("KMP_GCS_BUCKET"),
        help="GCS bucket (default: $KMP_GCS_BUCKET)",
    )
    args = parser.parse_args()

    if not args.bucket:
        sys.exit("error: --bucket or KMP_GCS_BUCKET env required")

    profiles: list[dict] = []
    for i, profile in enumerate(load_profiles_from_gcs(args.bucket)):
        if args.limit is not None and i >= args.limit:
            break
        profiles.append(profile)

    with_sites = [(p, website_url(p)) for p in profiles]
    with_sites = [(p, u) for p, u in with_sites if u]
    print(f"Loaded {len(profiles)} profile(s); {len(with_sites)} have a website link.")
    if not with_sites:
        print("Nothing to crawl.")
        return

    existing_hashes = {} if args.force else load_existing_page_hashes(args.bucket)

    if not args.dry_run:
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            sys.exit("error: GEMINI_API_KEY env required for real run")
        genai.configure(api_key=gemini_key)

    session = make_session()
    n_skipped_unchanged = n_failed = n_written = 0
    failures: Counter[str] = Counter()

    for profile, url in with_sites:
        slug = profile.get("slug")
        result = crawl_site(session, url)
        if not result.pages:
            print(f"  [{slug}] crawl failed ({result.reason}): {url}")
            failures[result.reason] += 1
            n_failed += 1
            continue

        text = clean_pages(result.pages)
        if not extraction_succeeded(text):
            print(f"  [{slug}] too_short ({len(text.strip())} chars) — not caching: {url}")
            failures["too_short"] += 1
            n_failed += 1
            continue

        ph = page_hash(text)
        if existing_hashes.get(slug) == ph:
            print(f"  [{slug}] unchanged — skipping Gemini extraction")
            n_skipped_unchanged += 1
            continue

        if args.dry_run:
            print(f"  [{slug}] would extract: {len(result.pages)} page(s), {len(text)} chars, hash {ph[:14]}…")
            n_written += 1
            continue

        extracted = extract_structured(text)
        record = build_record(profile, url, ph, extracted)
        write_weblinks_record(args.bucket, record)
        print(f"  [{slug}] wrote {len(record['sections'])} section(s) from {len(result.pages)} page(s)")
        n_written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(
        f"\nDone. {verb}: {n_written}; skipped unchanged: {n_skipped_unchanged}; "
        f"failed: {n_failed}."
    )
    if failures:
        print("Failure breakdown:")
        for reason, count in failures.most_common():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()

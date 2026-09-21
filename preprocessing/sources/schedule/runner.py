"""
Banner class-schedule scraper (entrypoint).

For every term Banner has not yet marked ``(View Only)``, collects each
section's instructors and writes one JSON record per course to
gs://<bucket>/schedule/<slug>.json.

Cost note: Banner returns an empty ``faculty`` list in bulk search results, so
instructors need one request per SECTION (~9,700 for a full term). That is the
whole run. Listing the sections themselves takes about twenty requests.

Usage:
    python -m preprocessing.sources.schedule.runner --dry-run --subject CS
    python -m preprocessing.sources.schedule.runner --subject CS,MATH
    python -m preprocessing.sources.schedule.runner --gcs-bucket BUCKET
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from shared.config import gcs_bucket
from shared.gcs import GCSStore, LocalStore, OutputStore

from ..pacing import Pacer
from ..profiles.config import LOCAL_OUTPUT_DIR
from .banner import BannerClient
from .config import DEFAULT_WORKERS, REQUEST_DELAY_SECONDS
from .source import ScheduleSource

#: "CS3800" / "CS 3800" -> subject, number. Banner writes it unspaced.
SUBJECT_COURSE_RE = re.compile(r"^([A-Z]{2,5})\s*(\d{3,4}[A-Z]?)$")


def course_slug(subject_course: str) -> str | None:
    """``"CS3800"`` -> ``"cs3800"``, matching the catalog source's slug."""
    m = SUBJECT_COURSE_RE.match((subject_course or "").strip())
    return f"{m.group(1)}{m.group(2)}".lower() if m else None


def _build_store(bucket: str | None) -> OutputStore:
    if bucket:
        return GCSStore(bucket)
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return LocalStore(LOCAL_OUTPUT_DIR)


def collect_term(
    client: BannerClient,
    term: dict,
    subjects: list[str] | None,
    workers: int,
    pacer: Pacer,
) -> dict[str, dict]:
    """Map course slug -> {code, title, subject, instructors} for one term."""
    listings: list[dict] = []
    for subject in subjects or [None]:
        listings.extend(client.sections(term["code"], subject=subject))
    print(f"  {term['description']}: {len(listings)} section(s) listed")

    by_course: dict[str, dict] = {}
    for row in listings:
        slug = course_slug(row.get("subjectCourse") or "")
        if not slug:
            continue
        entry = by_course.setdefault(
            slug,
            {
                "code": f"{row.get('subject','')} {row.get('courseNumber','')}".strip(),
                "title": row.get("courseTitle") or "",
                "subject": row.get("subject") or "",
                "instructors": set(),
                "section_count": 0,
                "crns": [],
            },
        )
        entry["section_count"] += 1
        entry["crns"].append(row.get("courseReferenceNumber"))

    crns = [(slug, crn) for slug, e in by_course.items() for crn in e["crns"]]
    print(f"  resolving instructors for {len(crns)} section(s) ({workers} workers)...")

    def fetch(item):
        slug, crn = item
        pacer.wait()
        return slug, client.instructors(term["code"], crn)

    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, item) for item in crns]
        for future in as_completed(futures):
            try:
                slug, faculty = future.result()
            except Exception:
                failed += 1
                continue
            for f in faculty:
                name = BannerClient.display_to_name(f.get("displayName") or "")
                if name:
                    by_course[slug]["instructors"].add(name)
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{len(crns)} sections resolved")

    if failed:
        print(f"  warning: {failed} section lookup(s) failed")
    for entry in by_course.values():
        entry["instructors"] = sorted(entry["instructors"])
        entry.pop("crns", None)
    return by_course


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default=None, help="Comma-separated Banner subjects (default: all)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS)
    parser.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
    parser.add_argument(
        "--gcs-bucket", default=gcs_bucket(),
        help="If set (or env KMP_GCS_BUCKET), write to gs://BUCKET/ instead of ./data/",
    )
    args = parser.parse_args()

    client = BannerClient()
    client.start()
    terms = client.active_terms()
    if not terms:
        sys.exit("error: Banner lists no active term (every term is marked View Only)")
    print(f"Active term(s): {', '.join(t['description'] for t in terms)}")

    subjects = [s.strip().upper() for s in args.subject.split(",")] if args.subject else None
    pacer = Pacer(args.delay)

    # slug -> record, merged across terms so a course taught in two upcoming
    # terms produces one chunk listing both rather than two competing chunks.
    merged: dict[str, dict] = defaultdict(lambda: {"terms": []})
    for term in terms:
        for slug, entry in collect_term(client, term, subjects, args.workers, pacer).items():
            record = merged[slug]
            record.update(
                slug=slug, code=entry["code"], title=entry["title"], subject=entry["subject"]
            )
            record["terms"].append(
                {
                    "code": term["code"],
                    "description": term["description"],
                    "instructors": entry["instructors"],
                    "section_count": entry["section_count"],
                }
            )

    store = _build_store(args.gcs_bucket)
    print(f"\nOutput store: {store.describe()}")
    source = ScheduleSource()
    written = skipped = 0
    for slug, record in sorted(merged.items()):
        if not source.is_ingestable(record):
            skipped += 1
            continue
        if not args.dry_run:
            store.write_text(
                f"{source.prefix}{slug}.json", json.dumps(record, indent=2, ensure_ascii=False)
            )
        written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(f"Done. {verb} {written} course record(s); {skipped} skipped (no named instructor).")


if __name__ == "__main__":
    main()

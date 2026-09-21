"""One-shot: stamp ``college`` onto the profile vectors that predate the field.

Why this exists
---------------
``ProfileSource`` only writes ``college`` into a chunk's metadata when the stored
record carries it, so the ~2,221 Khoury vectors ingested before multi-college
support have no ``college`` key. A normal re-ingest will NOT fix that: metadata
is not part of ``content_hash``, so the runner sees an unchanged hash and skips
the chunk entirely. The field has to be set directly on the index.

Until this runs, "absent means khoury" is the rule a metadata filter must follow.
After it runs, ``filter={"college": {"$eq": "khoury"}}`` works and the rule can
be dropped.

Cost: Pinecone charges an update like an upsert -- roughly 1 WU per KB of the
existing record, about 5 WU per vector here, so ~12k of the 2M monthly budget.

Usage:
    python -m preprocessing.ingest.backfill_college_metadata --dry-run
    python -m preprocessing.ingest.backfill_college_metadata
    python -m preprocessing.ingest.backfill_college_metadata --college khoury --index other-index
"""

from __future__ import annotations

import argparse
import os
import sys

from pinecone import Pinecone

from shared.config import PINECONE_DEFAULT_INDEX

from ..sources.entities import DEFAULT_COLLEGE
from ..sources.profiles.config import COLLEGES_BY_KEY

#: Vector ids are ``{entity}#{section}``; a namespaced entity carries its college
#: already, so only un-prefixed ids need the backfill.
PREFIXES = tuple(f"{key}-" for key in COLLEGES_BY_KEY if key != DEFAULT_COLLEGE)


def needs_backfill(vector_id: str) -> bool:
    """True for a bare ``{slug}#{section}`` id -- i.e. a pre-namespacing record."""
    return not vector_id.startswith(PREFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--college", default=DEFAULT_COLLEGE, help="Value to stamp")
    parser.add_argument("--index", default=os.environ.get("PINECONE_INDEX_NAME") or PINECONE_DEFAULT_INDEX)
    parser.add_argument("--dry-run", action="store_true", help="Count what would change; write nothing")
    args = parser.parse_args()

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        sys.exit("error: PINECONE_API_KEY env required")

    index = Pinecone(api_key=api_key).Index(args.index)
    print(f"Index: {args.index}  ->  stamping college={args.college!r}")

    scanned = updated = 0
    for page in index.list():
        targets = [vid for vid in page if needs_backfill(vid)]
        scanned += len(page)
        if not targets:
            continue
        if args.dry_run:
            updated += len(targets)
            continue
        for vector_id in targets:
            index.update(id=vector_id, set_metadata={"college": args.college})
            updated += 1
        print(f"  {updated} updated ({scanned} scanned)")

    verb = "would update" if args.dry_run else "updated"
    print(f"Done. {verb} {updated} of {scanned} vector(s).")


if __name__ == "__main__":
    main()

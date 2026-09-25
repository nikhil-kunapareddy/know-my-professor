"""
OpenAlex publications for every professor (entrypoint).

Per run:
  1. list every Northeastern author in OpenAlex once (~159 credits)
  2. for each ingestable profile, find same-name author records locally
  3. skip the professor if neither the profile nor its candidates changed since
     the last run -- the candidates' works_count is the change signal, and it
     comes free with step 1, so an unchanged professor costs nothing
  4. otherwise list each candidate's recent works (1 credit each), and have
     Claude pick which candidates are this person and describe their work
  5. write gs://$KMP_GCS_BUCKET/publications/{entity}.json

Stops cleanly when the OpenAlex daily budget runs out, and the next run resumes:
step 3 skips everything already done.

Required env:  KMP_GCS_BUCKET, ANTHROPIC_API_KEY (not for --dry-run)
Optional env:  OPENALEX_API_KEY -- $1/day of OpenAlex credit instead of $0.10

Usage:
  python -m preprocessing.sources.publications.runner --dry-run
  python -m preprocessing.sources.publications.runner --preview --college cos --limit 5
  python -m preprocessing.sources.publications.runner --limit 50
  python -m preprocessing.sources.publications.runner            # full run
  python -m preprocessing.sources.publications.runner --force    # ignore the skip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

from shared.config import gcs_bucket
from shared.gcs import GCSStore, OutputStore

from ..base import content_hash
from ..entities import college_of, entity_key
from ..llm import Claude
from ..registry import get_source
from .config import (
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    LOOKBACK_YEARS,
    MATCH_PROMPT,
    MATCH_SCHEMA,
    MAX_LISTED_WORKS,
    SCHEMA_VERSION,
)
from .openalex import AuthorIndex, BudgetExhausted, OpenAlexClient
from .source import PublicationsSource

#: Read profiles through the registry, as weblinks does, rather than importing
#: that package: this source only needs to know where they are stored.
PROFILE_SOURCE = get_source("profiles")
PREFIX = PublicationsSource.prefix

#: Overlaps the Claude calls. OpenAlex stays behind one shared Pacer.
DEFAULT_WORKERS = 8

_PROFILE_CONTEXT_CHARS = 600


def input_hash(profile: dict, candidates: list[dict]) -> str:
    """Everything the model's answer depends on, except the works themselves.

    ``works_count`` stands in for the works: it moves when a paper is added. It
    misses a correction that leaves the count alone, which a monthly refresh
    can live with -- and ``--force`` exists.
    """
    payload = {
        "schema": SCHEMA_VERSION,
        "name": profile.get("name") or "",
        "title": profile.get("title") or "",
        "college": college_of(profile),
        "candidates": [[c["id"], c["works_count"]] for c in candidates],
    }
    return content_hash(json.dumps(payload, sort_keys=True))


def build_prompt(profile: dict, candidates: list[tuple[dict, list[dict]]]) -> str:
    """The profile, then each candidate with its topics and recent works."""
    interests = "; ".join(
        [*(profile.get("research_interests") or []), *(profile.get("areas_of_interest") or [])]
    )
    lines = [
        MATCH_PROMPT + "FACULTY PROFILE",
        f"Name: {profile.get('name') or ''}",
        f"Title: {profile.get('title') or 'unknown'}",
        f"College: {college_of(profile)}",
        f"Research interests: {interests[:_PROFILE_CONTEXT_CHARS] or 'not listed'}",
        f"Biography: {(profile.get('biography') or '')[:_PROFILE_CONTEXT_CHARS] or 'not given'}",
        "",
        "CANDIDATE AUTHOR RECORDS",
    ]
    for author, works in candidates:
        topics = ", ".join(author["topics"]) or "unknown"
        lines.append(f"\n[{author['id']}] {author['name']} -- {author['works_count']} works; topics: {topics}")
        if not works:
            lines.append(f"- (no works in the last {LOOKBACK_YEARS} years)")
        for work in works:
            venue = f" ({work['venue']})" if work["venue"] else ""
            lines.append(f"- {work['year']}: {work['title']}{venue}")
            if work["abstract"]:
                lines.append(f"  {work['abstract']}")
    return "\n".join(lines)


def merge_works(work_lists: list[list[dict]]) -> list[dict]:
    """Newest first, one entry per paper.

    Two author records for the same person often both carry a paper, and a
    preprint and its published version share a title; either would otherwise
    list the paper twice.
    """
    everything = [w for works in work_lists for w in works if w.get("title")]
    everything.sort(key=lambda w: (w.get("date") or "", w["id"]), reverse=True)
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    merged: list[dict] = []
    for work in everything:
        title_key = work["title"].casefold()
        if work["id"] in seen_ids or title_key in seen_titles:
            continue
        seen_ids.add(work["id"])
        seen_titles.add(title_key)
        merged.append(work)
    return merged[:MAX_LISTED_WORKS]


def build_record(
    profile: dict,
    candidates: list[dict],
    works_by_author: dict[str, list[dict]],
    decision: dict,
    ihash: str,
    model: str,
) -> dict:
    """The stored record. Ids the model returns that were never offered are dropped."""
    offered = {c["id"] for c in candidates}
    picked = set(decision.get("matching_author_ids") or []) & offered
    chosen = [c["id"] for c in candidates if c["id"] in picked]  # candidate order, not the model's
    works = merge_works([works_by_author.get(i, []) for i in chosen])
    return {
        "slug": profile.get("slug"),
        "college": college_of(profile),
        "professor_name": profile.get("name") or "",
        "professor_title": profile.get("title") or "",
        "openalex_ids": chosen,
        "url": f"https://openalex.org/{chosen[0]}" if chosen else "",
        # Themes describe the accepted works; with none there is nothing to describe.
        "themes": (decision.get("themes") or "").strip() if works else "",
        "works": [{k: w[k] for k in ("id", "title", "year", "venue", "doi", "type")} for w in works],
        "candidates_considered": sorted(offered),
        "input_hash": ihash,
        "schema_version": SCHEMA_VERSION,
        "model": model,
    }


class PublicationsJob:
    """Matches, describes, and stores each professor's OpenAlex works."""

    def __init__(self, client: OpenAlexClient, claude: Claude, store: OutputStore | None,
                 since: str, preview: bool = False):
        self.client = client
        self.claude = claude
        self.store = store
        self.since = since
        self.preview = preview
        self.stop = threading.Event()

    def process(self, profile: dict, candidates: list[dict], ihash: str) -> str:
        """One professor. Returns an outcome label for the run summary."""
        if self.stop.is_set():
            return "not_reached"
        try:
            works = {c["id"]: self.client.recent_works(c["id"], self.since) for c in candidates}
            prompt = build_prompt(profile, [(c, works[c["id"]]) for c in candidates])
            decision = self.claude.json(prompt, MATCH_SCHEMA, CLAUDE_MAX_TOKENS)
        except BudgetExhausted:
            self.stop.set()
            return "budget_exhausted"
        if decision is None:
            return "model_failed"  # nothing written, so the next run retries it

        record = build_record(profile, candidates, works, decision, ihash, self.claude.model)
        key = entity_key(profile)
        if self.preview:
            print(json.dumps(record, indent=2, ensure_ascii=False))
        else:
            self.store.write_text(f"{PREFIX}{key}.json", json.dumps(record, indent=2, ensure_ascii=False))
        return "matched" if record["works"] else "no_match"

    def run(self, todo: list[tuple[dict, list[dict], str]], workers: int) -> Counter:
        outcomes: Counter[str] = Counter()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(self.process, p, c, h): p for p, c, h in todo}
            for i, future in enumerate(as_completed(futures), 1):
                profile = futures[future]
                try:
                    outcome = future.result()
                except Exception as e:  # one bad professor must not end the run
                    outcome = "failed"
                    print(f"  [{i}/{len(todo)}] FAILED {entity_key(profile)}: {e}")
                else:
                    if outcome not in ("not_reached",):
                        print(f"  [{i}/{len(todo)}] {entity_key(profile)}: {outcome}")
                outcomes[outcome] += 1
        return outcomes


def select_profiles(store: OutputStore, colleges: set[str] | None) -> list[dict]:
    """Ingestable profiles only: a staff entry with no content gets no papers either."""
    profiles = []
    for profile in store.iter_json(PROFILE_SOURCE.prefix):
        if not entity_key(profile) or not PROFILE_SOURCE.is_ingestable(profile):
            continue
        if colleges and college_of(profile) not in colleges:
            continue
        profiles.append(profile)
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N professors with candidates")
    parser.add_argument("--college", default=None, help="Comma-separated college keys (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="List authors and count candidates; no works, no Claude, no writes")
    parser.add_argument("--preview", action="store_true", help="Full pipeline, but print records instead of writing them")
    parser.add_argument("--force", action="store_true", help="Re-run professors whose inputs are unchanged")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--bucket", default=gcs_bucket(), help="GCS bucket (default: $KMP_GCS_BUCKET)")
    args = parser.parse_args()

    if not args.bucket:
        sys.exit("error: --bucket or KMP_GCS_BUCKET env required")
    store = GCSStore(args.bucket)
    colleges = {c.strip() for c in args.college.split(",")} if args.college else None

    profiles = select_profiles(store, colleges)
    existing = {} if args.force else {
        entity_key(r): r.get("input_hash") for r in store.iter_json(PREFIX) if entity_key(r)
    }
    print(f"Loaded {len(profiles)} ingestable profile(s); {len(existing)} existing publication record(s).")

    api_key = os.environ.get("OPENALEX_API_KEY") or None
    client = OpenAlexClient(api_key=api_key)
    print(f"Listing Northeastern authors in OpenAlex ({'with' if api_key else 'WITHOUT'} an API key)...")
    try:
        index = AuthorIndex(client.northeastern_authors())
    except BudgetExhausted:
        sys.exit("OpenAlex budget spent before the author list finished; nothing to do until it resets.")
    print(f"  {index.size} author record(s) indexed, {client.credits_used} credit(s) used")

    todo, skipped, without = [], 0, 0
    for profile in profiles:
        candidates = index.candidates(profile.get("name"))
        if not candidates:
            without += 1
            continue
        ihash = input_hash(profile, candidates)
        if existing.get(entity_key(profile)) == ihash:
            skipped += 1
            continue
        todo.append((profile, candidates, ihash))
    if args.limit is not None:
        todo = todo[: args.limit]
    n_candidates = sum(len(c) for _, c, _ in todo)
    print(f"{len(todo)} professor(s) to process ({n_candidates} candidate record(s), ~{n_candidates} credit(s)); "
          f"{skipped} unchanged; {without} with no same-name author.")

    if args.dry_run or not todo:
        return

    since = (date.today() - timedelta(days=365 * LOOKBACK_YEARS)).isoformat()
    job = PublicationsJob(client, Claude(model=CLAUDE_MODEL, label="claude publications"),
                          store, since, preview=args.preview)
    outcomes = job.run(todo, args.workers)

    print("\nDone. " + ", ".join(f"{k}: {v}" for k, v in sorted(outcomes.items())))
    print(f"openalex: {client.requests} request(s), {client.credits_used} credit(s)")
    print(job.claude.usage_line())
    if job.stop.is_set():
        print("Stopped early: OpenAlex budget spent. The next run resumes where this one stopped.")


if __name__ == "__main__":
    main()

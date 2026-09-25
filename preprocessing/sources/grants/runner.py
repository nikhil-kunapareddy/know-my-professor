"""
NSF and NIH research funding for every professor (entrypoint).

Per run:
  1. list Northeastern's NSF awards and NIH projects that are active or ended
     within LOOKBACK_YEARS (~40 requests between the two APIs)
  2. join each award's investigators to profiles by name
  3. summarise each matched award once with Claude; summaries are cached by
     award and only regenerated when the title or abstract changes
  4. write gs://$KMP_GCS_BUCKET/grants/{entity}.json, only where it changed

A professor whose awards have all aged out gets a record with no awards rather
than keeping a stale one: ingest stops embedding it (``is_ingestable``).

Required env:  KMP_GCS_BUCKET, ANTHROPIC_API_KEY (not for --dry-run)

Usage:
  python -m preprocessing.sources.grants.runner --dry-run
  python -m preprocessing.sources.grants.runner --preview --limit 5
  python -m preprocessing.sources.grants.runner
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

from shared.config import gcs_bucket
from shared.gcs import GCSStore, OutputStore

from ..base import content_hash
from ..entities import college_of, entity_key
from ..llm import Claude
from ..names import ProfileIndex
from ..registry import get_source
from .apis import NihClient, NsfClient
from .config import (
    ABSTRACT_CHARS,
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    LOOKBACK_YEARS,
    SCHEMA_VERSION,
    SUMMARY_CACHE_KEY,
    SUMMARY_PROMPT,
)
from .source import GrantsSource

PROFILE_SOURCE = get_source("profiles")
PREFIX = GrantsSource.prefix
DEFAULT_WORKERS = 8

#: Stored per award in the record; the investigators list and abstract are not
#: -- the chunk names the professor, and the summary stands in for the abstract.
_RECORD_FIELDS = ("agency", "award_id", "title", "program", "start", "end", "amount", "amount_basis", "url")


def award_key(award: dict) -> str:
    return f"{award['agency']}-{award['award_id']}"


def summary_hash(award: dict) -> str:
    return content_hash(f"{SCHEMA_VERSION}\n{award['title']}\n{award['abstract']}")


def match_awards(awards: list[dict], index: ProfileIndex) -> dict[str, tuple[dict, list[tuple[dict, str]]]]:
    """entity -> (profile, [(award, role)]).

    An award is attached once per professor even if their name appears twice
    on it; the first role listed wins, and NSF and NIH both list the lead first.
    """
    matched: dict[str, tuple[dict, list[tuple[dict, str]]]] = {}
    for award in awards:
        for investigator in award["investigators"]:
            for profile in index.match(investigator["name"]):
                key = entity_key(profile)
                _, items = matched.setdefault(key, (profile, []))
                if all(award_key(a) != award_key(award) for a, _ in items):
                    items.append((award, investigator["role"]))
    return matched


def build_record(profile: dict, items: list[tuple[dict, str]], summaries: dict[str, str], today: str) -> dict:
    """The stored record: newest-ending award first, status fixed as of ``today``."""
    awards = []
    for award, role in items:
        entry = {k: award[k] for k in _RECORD_FIELDS}
        entry["role"] = role
        entry["status"] = "active" if (award["end"] or "9999") >= today else "ended"
        entry["summary"] = summaries.get(award_key(award), "")
        awards.append(entry)
    awards.sort(key=lambda a: (a["end"] or "", a["agency"], a["award_id"]), reverse=True)
    return {
        "slug": profile.get("slug"),
        "college": college_of(profile),
        "professor_name": profile.get("name") or "",
        "professor_title": profile.get("title") or "",
        "awards": awards,
        "schema_version": SCHEMA_VERSION,
    }


def empty_record(existing: dict) -> dict:
    """What a professor with no awards left in the window is rewritten to."""
    return {**existing, "awards": [], "schema_version": SCHEMA_VERSION}


def summarise(awards: list[dict], cache: dict[str, dict], claude: Claude, workers: int) -> Counter:
    """Fill ``cache`` for every award whose text changed. Mutates ``cache``."""
    todo = [a for a in awards if (cache.get(award_key(a)) or {}).get("hash") != summary_hash(a)]
    outcomes: Counter[str] = Counter(cached=len(awards) - len(todo))

    def one(award: dict) -> tuple[dict, str]:
        if not award["abstract"]:
            return award, ""
        prompt = f"{SUMMARY_PROMPT}Title: {award['title']}\n\nAbstract:\n{award['abstract'][:ABSTRACT_CHARS]}"
        return award, claude.text(prompt, CLAUDE_MAX_TOKENS)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for future in as_completed([pool.submit(one, a) for a in todo]):
            try:
                award, summary = future.result()
            except Exception as e:  # a failed summary is retried next run, never fatal
                outcomes["failed"] += 1
                print(f"  summary FAILED: {e}")
                continue
            cache[award_key(award)] = {"hash": summary_hash(award), "summary": summary}
            outcomes["summarised"] += 1
    return outcomes


def select_profiles(store: OutputStore) -> list[dict]:
    return [
        p for p in store.iter_json(PROFILE_SOURCE.prefix)
        if entity_key(p) and PROFILE_SOURCE.is_ingestable(p)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="Only write the first N professors")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and match only; no Claude, no writes")
    parser.add_argument("--preview", action="store_true", help="Full pipeline, but print records instead of writing")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--bucket", default=gcs_bucket(), help="GCS bucket (default: $KMP_GCS_BUCKET)")
    args = parser.parse_args()

    if not args.bucket:
        sys.exit("error: --bucket or KMP_GCS_BUCKET env required")
    store = GCSStore(args.bucket)

    index = ProfileIndex(select_profiles(store))
    print(f"Indexed {len(index)} ingestable profile(s) by name.")

    today = date.today()
    since = (today - timedelta(days=365 * LOOKBACK_YEARS)).isoformat()
    nsf, nih = NsfClient(), NihClient()
    awards = nsf.awards(since) + nih.projects(since)
    print(f"Fetched {len(awards)} award(s) ending on or after {since} "
          f"({nsf.requests} NSF + {nih.requests} NIH request(s)).")

    matched = match_awards(awards, index)
    matched_awards = {award_key(a): a for _, items in matched.values() for a, _ in items}
    print(f"{len(matched_awards)} award(s) matched to {len(matched)} professor(s).")
    if args.dry_run:
        return

    cache = json.loads(store.read_text(SUMMARY_CACHE_KEY) or "{}")
    claude = Claude(model=CLAUDE_MODEL, label="claude grants")
    outcomes = summarise(list(matched_awards.values()), cache, claude, args.workers)
    if not args.preview:
        store.write_text(SUMMARY_CACHE_KEY, json.dumps(cache, indent=1, sort_keys=True, ensure_ascii=False))
    summaries = {k: v["summary"] for k, v in cache.items()}
    print("Summaries: " + ", ".join(f"{k}: {v}" for k, v in sorted(outcomes.items())))

    existing = {entity_key(r): r for r in store.iter_json(PREFIX) if entity_key(r)}
    records = {key: build_record(profile, items, summaries, today.isoformat())
               for key, (profile, items) in matched.items()}
    # Aged out since last run: rewrite empty rather than leave the old awards live.
    for key, old in existing.items():
        if key not in records and old.get("awards"):
            records[key] = empty_record(old)

    written = unchanged = 0
    for i, (key, record) in enumerate(sorted(records.items())):
        if args.limit is not None and i >= args.limit:
            break
        text = json.dumps(record, indent=2, ensure_ascii=False)
        if args.preview:
            print(text)
            continue
        if existing.get(key) == record:
            unchanged += 1
            continue
        store.write_text(f"{PREFIX}{key}.json", text)
        written += 1

    print(f"\nDone. {written} record(s) written, {unchanged} unchanged.")
    print(claude.usage_line())


if __name__ == "__main__":
    main()

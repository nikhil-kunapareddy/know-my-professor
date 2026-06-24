"""
Ingest scraped profile JSONs from GCS into a Pinecone vector index.

For each profile in gs://$KMP_GCS_BUCKET/profiles/*.json:
  - skip if it has no biography / research_interests / areas_of_interest
  - emit one chunk per populated section (biography, research_interests, education,
    areas_of_interest, labs_and_groups, projects)
  - embed with Gemini gemini-embedding-001 (3072 dim)
  - upsert into Pinecone with deterministic id `{slug}#{section}`

Required env:
  KMP_GCS_BUCKET     — source bucket (e.g. know-my-professor-raw)
  GEMINI_API_KEY     — Google AI Studio API key
  PINECONE_API_KEY   — Pinecone API key
Optional env:
  PINECONE_INDEX_NAME (default: know-my-professor)
  PINECONE_REGION     (default: us-east-1)
  PINECONE_CLOUD      (default: aws)

Usage:
  python ingest.py --dry-run --limit 5   # print chunk samples, no embed/upsert
  python ingest.py --limit 5             # real run on 5 profiles
  python ingest.py                       # full ingest

Module layout:
  config.py          shared constants
  chunking.py        Chunk + profile -> chunks
  embedding.py       Gemini embedding (rate-limited)
  gcs.py             load profiles from GCS
  pinecone_store.py  index management + upsert
  ingest.py          orchestration (this file)
"""

from __future__ import annotations

import argparse
import os
import sys

import google.generativeai as genai
from pinecone import Pinecone

from chunking import Chunk, enrichment_to_chunks, is_substantive, profile_to_chunks
from config import (
    EMBED_DIM,
    PINECONE_DEFAULT_CLOUD,
    PINECONE_DEFAULT_INDEX,
    PINECONE_DEFAULT_REGION,
)
from embedding import embed_texts
from gcs import load_enrichment_from_gcs, load_profiles_from_gcs
from pinecone_store import (
    fetch_existing_hashes,
    get_or_create_index,
    upsert_in_batches,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N profiles")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chunking summary, do not call Gemini or Pinecone",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("KMP_GCS_BUCKET"),
        help="GCS bucket (default: $KMP_GCS_BUCKET)",
    )
    parser.add_argument(
        "--index",
        default=os.environ.get("PINECONE_INDEX_NAME", PINECONE_DEFAULT_INDEX),
        help="Pinecone index name",
    )
    args = parser.parse_args()

    if not args.bucket:
        sys.exit("error: --bucket or KMP_GCS_BUCKET env required")

    profiles: list[dict] = []
    for i, profile in enumerate(load_profiles_from_gcs(args.bucket)):
        if args.limit is not None and i >= args.limit:
            break
        profiles.append(profile)
    print(f"Loaded {len(profiles)} profile(s) from gs://{args.bucket}")

    substantive = [p for p in profiles if is_substantive(p)]
    print(f"Substantive (bio/research/areas non-empty): {len(substantive)}/{len(profiles)}")

    all_chunks: list[Chunk] = []
    for p in substantive:
        all_chunks.extend(profile_to_chunks(p))
    profile_chunk_count = len(all_chunks)

    # Fold in weblinks enrichment for the profiles we loaded (respects --limit).
    loaded_slugs = {p.get("slug") for p in profiles}
    enrichment_chunks: list[Chunk] = []
    for enriched in load_enrichment_from_gcs(args.bucket):
        if enriched.get("slug") in loaded_slugs:
            enrichment_chunks.extend(enrichment_to_chunks(enriched))
    all_chunks.extend(enrichment_chunks)
    print(
        f"Total chunks: {len(all_chunks)} "
        f"({profile_chunk_count} profile + {len(enrichment_chunks)} weblinks)"
    )
    if not all_chunks:
        print("Nothing to ingest.")
        return

    if args.dry_run:
        print("\n--- DRY RUN: sample chunks ---")
        for c in all_chunks[:3]:
            print(f"\n[{c.vector_id}]")
            print(c.text[:300])
            if len(c.text) > 300:
                print("...")
        return

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        sys.exit("error: GEMINI_API_KEY env required for real run")
    genai.configure(api_key=gemini_key)

    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        sys.exit("error: PINECONE_API_KEY env required for real run")
    pc = Pinecone(api_key=pinecone_key)

    index = get_or_create_index(
        pc,
        args.index,
        EMBED_DIM,
        os.environ.get("PINECONE_CLOUD", PINECONE_DEFAULT_CLOUD),
        os.environ.get("PINECONE_REGION", PINECONE_DEFAULT_REGION),
    )

    # Hash-aware skip: embed a chunk only if its id is new OR its text changed
    # since last ingest. Unchanged chunks (same content_hash) are skipped, so a
    # re-run on identical data embeds nothing — no wasteful re-embedding.
    existing_hashes = fetch_existing_hashes(index, [c.vector_id for c in all_chunks])
    pending = [
        c for c in all_chunks
        if existing_hashes.get(c.vector_id) != c.metadata["content_hash"]
    ]
    changed = sum(1 for c in pending if c.vector_id in existing_hashes)
    print(
        f"\n{len(existing_hashes)} of {len(all_chunks)} chunks already in Pinecone; "
        f"{len(pending)} need embedding ({len(pending) - changed} new, {changed} changed)."
    )
    if not pending:
        print("Nothing to embed. Done.")
        return

    print("Embedding + upserting (paced for 100 RPM)...")
    SLICE = 25
    for i in range(0, len(pending), SLICE):
        batch = pending[i : i + SLICE]
        texts = [c.text for c in batch]
        vectors = embed_texts(texts)
        upsert_in_batches(index, batch, vectors)
        print(f"  {i + len(batch)}/{len(pending)} chunks ingested")

    print("Done.")


if __name__ == "__main__":
    main()

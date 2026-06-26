"""
Ingest scraped profiles + weblinks enrichment from GCS into a Pinecone index.

For each profile in gs://$KMP_GCS_BUCKET/profiles/*.json:
  - skip if it has no biography / research_interests / areas_of_interest
  - emit one chunk per populated section, plus weblinks enrichment sections
  - embed with Mistral (1024 dim), batched
  - upsert into Pinecone with deterministic id `{slug}#{section}`

Resumable: a chunk is embedded only if its id is new OR its text changed since
last ingest (content_hash compare), so a re-run on identical data embeds nothing.

Required env:
  KMP_GCS_BUCKET, MISTRAL_API_KEY, PINECONE_API_KEY
Optional env:
  PINECONE_INDEX_NAME, PINECONE_REGION, PINECONE_CLOUD

Usage:
  python -m preprocessing.ingest.runner --dry-run --limit 5
  python -m preprocessing.ingest.runner --limit 5
  python -m preprocessing.ingest.runner            # full ingest
"""

from __future__ import annotations

import argparse
import os
import sys

from pinecone import Pinecone

from shared.config import (
    EMBED_DIM,
    PINECONE_DEFAULT_CLOUD,
    PINECONE_DEFAULT_INDEX,
    PINECONE_DEFAULT_REGION,
    PROFILE_PREFIX,
    WEBLINKS_PREFIX,
    gcs_bucket,
)
from shared.gcs import GCSStore

from .chunking import Chunk, Chunker
from .embedding import DocumentEmbedder
from .pinecone_store import PineconeStore

EMBED_SLICE = 25  # chunks embedded + upserted per progress step


def _collect_chunks(store: GCSStore, chunker: Chunker, limit: int | None) -> list[Chunk]:
    """Load profiles (+ matching weblinks) and render them into chunks."""
    profiles: list[dict] = []
    for i, profile in enumerate(store.iter_json(PROFILE_PREFIX)):
        if limit is not None and i >= limit:
            break
        profiles.append(profile)
    print(f"Loaded {len(profiles)} profile(s)")

    substantive = [p for p in profiles if chunker.is_substantive(p)]
    print(f"Substantive (bio/research/areas non-empty): {len(substantive)}/{len(profiles)}")

    chunks: list[Chunk] = []
    for p in substantive:
        chunks.extend(chunker.profile_to_chunks(p))
    profile_chunk_count = len(chunks)

    # Fold in weblinks enrichment for the profiles we loaded (respects --limit).
    loaded_slugs = {p.get("slug") for p in profiles}
    enrichment_chunks: list[Chunk] = []
    for enriched in store.iter_json(WEBLINKS_PREFIX):
        if enriched.get("slug") in loaded_slugs:
            enrichment_chunks.extend(chunker.enrichment_to_chunks(enriched))
    chunks.extend(enrichment_chunks)
    print(
        f"Total chunks: {len(chunks)} "
        f"({profile_chunk_count} profile + {len(enrichment_chunks)} weblinks)"
    )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N profiles")
    parser.add_argument("--dry-run", action="store_true", help="Print chunking summary; no embed/upsert")
    parser.add_argument("--bucket", default=gcs_bucket(), help="GCS bucket (default: $KMP_GCS_BUCKET)")
    parser.add_argument(
        "--index",
        default=os.environ.get("PINECONE_INDEX_NAME", PINECONE_DEFAULT_INDEX),
        help="Pinecone index name",
    )
    args = parser.parse_args()

    if not args.bucket:
        sys.exit("error: --bucket or KMP_GCS_BUCKET env required")

    store = GCSStore(args.bucket)
    chunker = Chunker()
    all_chunks = _collect_chunks(store, chunker, args.limit)
    if not all_chunks:
        print("Nothing to ingest.")
        return

    if args.dry_run:
        print("\n--- DRY RUN: sample chunks ---")
        for c in all_chunks[:3]:
            print(f"\n[{c.vector_id}]")
            print(c.text[:300] + ("..." if len(c.text) > 300 else ""))
        return

    if not os.environ.get("MISTRAL_API_KEY"):
        sys.exit("error: MISTRAL_API_KEY env required for real run")
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        sys.exit("error: PINECONE_API_KEY env required for real run")

    pc = Pinecone(api_key=pinecone_key)
    pinecone_store = PineconeStore.get_or_create(
        pc,
        args.index,
        EMBED_DIM,
        os.environ.get("PINECONE_CLOUD", PINECONE_DEFAULT_CLOUD),
        os.environ.get("PINECONE_REGION", PINECONE_DEFAULT_REGION),
    )

    existing_hashes = pinecone_store.fetch_existing_hashes([c.vector_id for c in all_chunks])
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

    print("Embedding + upserting (Mistral, batched)...")
    embedder = DocumentEmbedder()
    for i in range(0, len(pending), EMBED_SLICE):
        batch = pending[i : i + EMBED_SLICE]
        vectors = embedder.embed_texts([c.text for c in batch])
        pinecone_store.upsert_in_batches(batch, vectors)
        print(f"  {i + len(batch)}/{len(pending)} chunks ingested")

    print("Done.")


if __name__ == "__main__":
    main()

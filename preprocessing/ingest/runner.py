"""
Ingest every registered source's records from GCS into a Pinecone index.

For each source in ``preprocessing.sources.registry``:
  - read gs://$KMP_GCS_BUCKET/{source.prefix}*.json
  - render each record into per-section chunks
  - embed the chunks whose text changed, batched
  - upsert into Pinecone with deterministic id `{slug}#{section}`

This module names no source. Adding a corpus means adding a ``Source`` to the
registry; nothing here changes.

Resumable: a chunk is embedded only if its id is new OR its text changed since
last ingest (content_hash compare), so a re-run on identical data embeds nothing.

Required env:
  KMP_GCS_BUCKET, PINECONE_API_KEY, plus the embedding provider's key
  (MISTRAL_API_KEY by default)
Optional env:
  PINECONE_INDEX_NAME, PINECONE_REGION, PINECONE_CLOUD

Usage:
  python -m preprocessing.ingest.runner --dry-run --limit 5
  python -m preprocessing.ingest.runner --limit 5
  python -m preprocessing.ingest.runner            # full ingest
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator

from pinecone import Pinecone

from shared.config import EMBED_RATE_LIMIT_SLEEP_SECONDS, gcs_bucket
from shared.embeddings import build_embedder
from shared.gcs import GCSStore
from shared.settings import IngestSettings, MissingSettingError, require_env

from ..sources.base import Chunk
from ..sources.registry import SOURCES, dependent_sources, entity_sources
from .pinecone_store import PineconeStore

EMBED_SLICE = 25  # chunks embedded + upserted per progress step


def _iter_limited(store: GCSStore, prefix: str, limit: int | None) -> Iterator[dict]:
    """Yield records under ``prefix``, stopping after ``limit`` of them."""
    for i, record in enumerate(store.iter_json(prefix)):
        if limit is not None and i >= limit:
            return
        yield record


def _collect_chunks(store: GCSStore, limit: int | None) -> list[Chunk]:
    """Render every registered source's records into chunks.

    Entity-defining sources run first and fix the set of known entity ids;
    dependent sources are then restricted to those ids. That ordering is what
    makes ``--limit`` coherent — you get enrichment for the professors in the
    slice, not enrichment for an unrelated arbitrary slice.
    """
    chunks: list[Chunk] = []
    entity_ids: set[str] = set()

    for source in entity_sources():
        loaded = ingestable = produced = 0
        for record in _iter_limited(store, source.prefix, limit):
            loaded += 1
            entity_id = source.entity_id(record)
            if entity_id:
                # Registered even when not ingestable, so enrichment for a
                # thin profile is still scoped to a professor we know about.
                entity_ids.add(entity_id)
            if not source.is_ingestable(record):
                continue
            ingestable += 1
            rendered = source.to_chunks(record)
            chunks.extend(rendered)
            produced += len(rendered)
        print(f"  {source.name}: {loaded} record(s), {ingestable} ingestable -> {produced} chunk(s)")

    for source in dependent_sources():
        matched = produced = 0
        for record in store.iter_json(source.prefix):
            if source.entity_id(record) not in entity_ids:
                continue
            if not source.is_ingestable(record):
                continue
            matched += 1
            rendered = source.to_chunks(record)
            chunks.extend(rendered)
            produced += len(rendered)
        print(f"  {source.name}: {matched} record(s) for known entities -> {produced} chunk(s)")

    print(f"Total chunks: {len(chunks)}")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N entity records")
    parser.add_argument("--dry-run", action="store_true", help="Print chunking summary; no embed/upsert")
    parser.add_argument("--bucket", default=gcs_bucket(), help="GCS bucket (default: $KMP_GCS_BUCKET)")
    parser.add_argument(
        "--index",
        default=None,
        help="Pinecone index name (default: $PINECONE_INDEX_NAME, else shared.config)",
    )
    args = parser.parse_args()

    if not args.bucket:
        sys.exit("error: --bucket or KMP_GCS_BUCKET env required")

    store = GCSStore(args.bucket)
    print(f"Sources: {', '.join(s.name for s in SOURCES)}")
    all_chunks = _collect_chunks(store, args.limit)
    if not all_chunks:
        print("Nothing to ingest.")
        return

    if args.dry_run:
        print("\n--- DRY RUN: sample chunks ---")
        for c in all_chunks[:3]:
            print(f"\n[{c.vector_id}]")
            print(c.text[:300] + ("..." if len(c.text) > 300 else ""))
        return

    # Validate every setting and credential before the first paid call, so a
    # misconfigured run fails in seconds rather than part-way through embedding.
    try:
        settings = IngestSettings.from_env(bucket=args.bucket, index_name=args.index)
        # Pace the document side: a full ingest is thousands of chunks and the
        # free tier is per-minute. The serving path builds the same embedder
        # unpaced, so a single query embed adds no latency.
        embedder = build_embedder(
            settings.embed_provider,
            pace_seconds=EMBED_RATE_LIMIT_SLEEP_SECONDS,
        )
        if embedder.api_key_env:
            require_env(embedder.api_key_env)
    except (MissingSettingError, ValueError) as e:
        sys.exit(f"error: {e}")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    pinecone_store = PineconeStore.get_or_create(
        pc,
        settings.index_name,
        embedder.dim,
        settings.cloud,
        settings.region,
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

    print(f"Embedding + upserting ({embedder.model}, {embedder.dim}-dim, batched)...")
    for i in range(0, len(pending), EMBED_SLICE):
        batch = pending[i : i + EMBED_SLICE]
        vectors = embedder.embed_texts([c.text for c in batch])
        pinecone_store.upsert_in_batches(batch, vectors)
        print(f"  {i + len(batch)}/{len(pending)} chunks ingested")

    print("Done.")


if __name__ == "__main__":
    main()

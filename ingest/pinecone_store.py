"""Pinecone index management and batched upsert."""

from __future__ import annotations

import time

from pinecone import Pinecone, ServerlessSpec

from chunking import Chunk
from config import FETCH_BATCH_SIZE, UPSERT_BATCH_SIZE


def fetch_existing_vector_ids(index) -> set[str]:
    """Return the set of vector IDs already present in the index."""
    existing: set[str] = set()
    for batch in index.list():
        existing.update(batch)
    return existing


def fetch_existing_hashes(index, ids: list[str]) -> dict[str, str]:
    """Map vector_id -> stored content_hash for the given IDs already in the index.

    IDs not present in the index (or lacking a content_hash) are simply absent
    from the result, so a plain ``.get(id) != new_hash`` check treats them as
    needing (re-)embedding.
    """
    hashes: dict[str, str] = {}
    unique = list(dict.fromkeys(ids))
    for i in range(0, len(unique), FETCH_BATCH_SIZE):
        batch = unique[i : i + FETCH_BATCH_SIZE]
        resp = index.fetch(ids=batch)
        vectors = getattr(resp, "vectors", None) or {}
        for vid, vec in vectors.items():
            meta = getattr(vec, "metadata", None) or {}
            stored = meta.get("content_hash")
            if stored:
                hashes[vid] = stored
    return hashes


def get_or_create_index(
    pc: Pinecone, name: str, dimension: int, cloud: str, region: str
):
    existing = {idx["name"] for idx in pc.list_indexes()}
    if name not in existing:
        print(f"Creating Pinecone index '{name}' (dim={dimension}, {cloud}/{region})...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        while True:
            desc = pc.describe_index(name)
            if desc.status.get("ready"):
                break
            print("  waiting for index to become ready...")
            time.sleep(3)
    return pc.Index(name)


def upsert_in_batches(index, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    payload = [
        {"id": c.vector_id, "values": v, "metadata": c.metadata}
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(payload), UPSERT_BATCH_SIZE):
        batch = payload[i : i + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch)

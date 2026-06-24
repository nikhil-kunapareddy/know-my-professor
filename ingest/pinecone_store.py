"""Pinecone index management and batched upsert."""

from __future__ import annotations

import time

from pinecone import Pinecone, ServerlessSpec

from chunking import Chunk
from config import UPSERT_BATCH_SIZE


def fetch_existing_vector_ids(index) -> set[str]:
    """Return the set of vector IDs already present in the index."""
    existing: set[str] = set()
    for batch in index.list():
        existing.update(batch)
    return existing


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

"""Pinecone index management and batched upsert."""

from __future__ import annotations

import time

from pinecone import Pinecone, ServerlessSpec

from shared.config import FETCH_BATCH_SIZE, UPSERT_BATCH_SIZE

from .chunking import Chunk


class PineconeStore:
    """Wraps a Pinecone index for hash-aware, batched upserts."""

    def __init__(self, index):
        self.index = index

    @classmethod
    def get_or_create(cls, pc: Pinecone, name: str, dimension: int, cloud: str, region: str) -> "PineconeStore":
        """Return a store for ``name``, creating the serverless index if absent."""
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
        return cls(pc.Index(name))

    def fetch_existing_hashes(self, ids: list[str]) -> dict[str, str]:
        """Map vector_id -> stored content_hash for the given IDs already present.

        IDs not present (or lacking a content_hash) are simply absent from the
        result, so a plain ``.get(id) != new_hash`` check treats them as needing
        (re-)embedding.
        """
        hashes: dict[str, str] = {}
        unique = list(dict.fromkeys(ids))
        for i in range(0, len(unique), FETCH_BATCH_SIZE):
            batch = unique[i : i + FETCH_BATCH_SIZE]
            resp = self.index.fetch(ids=batch)
            vectors = getattr(resp, "vectors", None) or {}
            for vid, vec in vectors.items():
                meta = getattr(vec, "metadata", None) or {}
                stored = meta.get("content_hash")
                if stored:
                    hashes[vid] = stored
        return hashes

    def upsert_in_batches(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Upsert chunk+vector pairs in batches of UPSERT_BATCH_SIZE."""
        payload = [
            {"id": c.vector_id, "values": v, "metadata": c.metadata}
            for c, v in zip(chunks, vectors)
        ]
        for i in range(0, len(payload), UPSERT_BATCH_SIZE):
            self.index.upsert(vectors=payload[i : i + UPSERT_BATCH_SIZE])

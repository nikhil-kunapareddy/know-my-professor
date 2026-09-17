"""Pinecone index management and batched upsert."""

from __future__ import annotations

import time

from pinecone import Pinecone, ServerlessSpec

from shared.config import FETCH_BATCH_SIZE, UPSERT_BATCH_SIZE

from ..sources.base import Chunk


class PineconeStore:
    """Wraps a Pinecone index for hash-aware, batched upserts."""

    def __init__(self, index):
        self.index = index

    @classmethod
    def get_or_create(cls, pc: Pinecone, name: str, dimension: int, cloud: str, region: str) -> PineconeStore:
        """Return a store for ``name``, creating the serverless index if absent.

        If the index already exists, its width is verified up front. An index's
        dimension is fixed at creation, so pointing a differently-sized embedder
        at it cannot work — and without this check the failure arrives mid-run as
        a rejected upsert, after the embedding calls have already been paid for.
        """
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
        else:
            cls.assert_dimension(pc.describe_index(name), name, dimension)

        return cls(pc.Index(name))

    @staticmethod
    def assert_dimension(description, name: str, expected: int) -> None:
        """Fail loudly when a live index cannot hold the configured vectors."""
        actual = getattr(description, "dimension", None)
        if actual is None and isinstance(description, dict):
            actual = description.get("dimension")
        if actual is None:
            print(f"  warning: could not read dimension of index '{name}'; skipping check")
            return
        if int(actual) != expected:
            raise ValueError(
                f"Pinecone index '{name}' is {actual}-dim but the configured embedder "
                f"produces {expected}-dim vectors. An index's dimension is immutable — "
                f"create a new index and re-ingest, or switch back to the matching "
                f"embedding model."
            )

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
        """Upsert chunk+vector pairs in batches of UPSERT_BATCH_SIZE.

        ``strict=True`` because a short vector list would otherwise zip away the
        trailing chunks: they would be silently skipped, yet counted as ingested,
        and the gap would only surface later as missing search results.
        """
        payload = [
            {"id": c.vector_id, "values": v, "metadata": c.metadata}
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(payload), UPSERT_BATCH_SIZE):
            self.index.upsert(vectors=payload[i : i + UPSERT_BATCH_SIZE])

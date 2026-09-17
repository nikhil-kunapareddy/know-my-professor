"""Golden test: chunk rendering must stay byte-identical across refactors.

Chunk text is hashed into ``content_hash``, and ``content_hash`` is what decides
whether ingest re-embeds a chunk. So any change to rendering silently invalidates
the whole Pinecone index and forces a full re-ingest. This test pins the exact
output for a fixture set that exercises every rendering branch.

If it fails, either the change was unintended, or it IS intended and you owe the
index a full re-ingest -- in which case regenerate the golden file deliberately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def records() -> dict:
    return json.loads((FIXTURES / "source_records.json").read_text())


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads((FIXTURES / "chunk_golden.json").read_text())


def _dump(chunks) -> list[dict]:
    return [{"vector_id": c.vector_id, "text": c.text, "metadata": c.metadata} for c in chunks]


def test_profile_chunks_match_golden(records, golden):
    from preprocessing.sources.registry import chunks_for

    for record, expected in zip(records["profiles"], golden["profiles"], strict=True):
        assert record.get("slug") == expected["slug"]
        assert _dump(chunks_for("profiles", record)) == expected["chunks"]


def test_weblinks_chunks_match_golden(records, golden):
    from preprocessing.sources.registry import chunks_for

    for record, expected in zip(records["weblinks"], golden["weblinks"], strict=True):
        assert record.get("slug") == expected["slug"]
        assert _dump(chunks_for("weblinks", record)) == expected["chunks"]


def test_substantive_filter_matches_golden(records, golden):
    from preprocessing.sources.registry import get_source

    source = get_source("profiles")
    for record, expected in zip(records["profiles"], golden["profiles"], strict=True):
        assert source.is_ingestable(record) is expected["is_substantive"]


def test_golden_covers_every_registered_section_type(golden):
    """Guard: a new section type must come with golden coverage, not slip through."""
    from preprocessing.sources.registry import all_section_types

    seen = {
        c["metadata"]["section_type"]
        for group in (golden["profiles"], golden["weblinks"])
        for entry in group
        for c in entry["chunks"]
    }
    missing = set(all_section_types()) - seen
    assert not missing, f"section types with no golden coverage: {sorted(missing)}"

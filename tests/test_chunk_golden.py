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


def test_every_registered_source_has_fixtures(records):
    """A new source must bring fixtures, or it is pinned by nothing."""
    from preprocessing.sources.registry import SOURCES

    missing = sorted({s.name for s in SOURCES} - set(records) - {"_comment"})
    assert not missing, f"registered source(s) with no fixture records: {missing}"


def test_chunks_match_golden(records, golden):
    """Every source's rendering, pinned byte-for-byte.

    Walks the registry rather than naming sources, for the same reason ingest
    does: adding a corpus should not mean editing this file.
    """
    from preprocessing.sources.registry import SOURCES, chunks_for

    for source in SOURCES:
        name = source.name
        assert name in golden, f"no golden entries for source {name!r}"
        for record, expected in zip(records[name], golden[name], strict=True):
            assert record.get("slug") == expected["slug"], name
            assert _dump(chunks_for(name, record)) == expected["chunks"], name


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
        for group in golden.values()
        for entry in group
        for c in entry["chunks"]
    }
    missing = set(all_section_types()) - seen
    assert not missing, f"section types with no golden coverage: {sorted(missing)}"

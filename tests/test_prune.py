"""Ingest prune: deleting vectors a run no longer produces, and refusing when unsafe.

Prune is the only code in the repo that deletes vectors, so every refusal rule
is pinned here, along with the batching and the default-namespace guard.
"""

from __future__ import annotations

import sys

import pytest

from preprocessing.ingest.pinecone_store import PineconeStore
from preprocessing.ingest.prune import PRUNE_FREE_COUNT, plan_prune, section_of
from preprocessing.sources.base import Chunk


def test_section_of_reads_the_section_back_out_of_an_id():
    assert section_of("khoury-jane#biography") == "biography"
    assert section_of("cos-jane#research_funding@3") == "research_funding"
    assert section_of("course-cs1800#course_description") == "course_description"
    assert section_of("no-section-at-all") == "(no section)"


def _ids(section, n, prefix="p"):
    return {f"{prefix}{i}#{section}" for i in range(n)}


def test_ordinary_turnover_is_pruned():
    existing = _ids("biography", 1000) | {"p0#research_funding@2", "p0#research_funding@3"}
    produced = _ids("biography", 1000) - {"p7#biography"}  # one professor left; one section shrank
    plan = plan_prune("people", existing, produced)
    assert plan.refused is None
    assert plan.stale == ("p0#research_funding@2", "p0#research_funding@3", "p7#biography")
    assert plan.by_section == {"biography": (1, 1000), "research_funding": (2, 2)}


def test_prune_refuses_when_the_run_produced_nothing():
    plan = plan_prune("people", _ids("biography", 50), set())
    assert plan.refused and "produced no chunks" in plan.refused


def test_prune_refuses_when_one_section_would_lose_too_much():
    """A source whose read came back empty: the namespace barely moves, one section vanishes."""
    bios = _ids("biography", 5000)
    sites = _ids("website_summary", 200)
    plan = plan_prune("people", bios | sites, bios)
    assert plan.refused and "website_summary 200/200" in plan.refused
    assert len(plan.stale) / len(bios | sites) < 0.05  # which a namespace-wide guard would have let through
    assert plan_prune("people", bios | sites, bios, max_fraction=1.0).refused is None


def test_small_sections_may_always_lose_a_few():
    existing = _ids("students_or_lab_members", PRUNE_FREE_COUNT)
    assert plan_prune("people", existing | {"keep#x"}, {"keep#x"}).refused is None
    one_more = _ids("students_or_lab_members", PRUNE_FREE_COUNT + 1)
    assert plan_prune("people", one_more | {"keep#x"}, {"keep#x"}).refused is not None


# --- PineconeStore.list_ids / delete_ids ------------------------------------------


class _ListingIndex:
    def __init__(self, pages):
        self.pages, self.deleted, self.list_calls = pages, [], []

    def list(self, **kwargs):
        self.list_calls.append(kwargs)
        yield from self.pages

    def delete(self, ids, namespace):
        self.deleted.append((list(ids), namespace))


def test_list_ids_pages_through_the_namespace():
    index = _ListingIndex([["a#x", "b#x"], ["c#x"]])
    assert PineconeStore(index).list_ids("people") == {"a#x", "b#x", "c#x"}
    assert index.list_calls == [{"namespace": "people"}]


def test_delete_ids_batches_and_refuses_the_default_namespace():
    index = _ListingIndex([])
    store = PineconeStore(index)
    store.delete_ids([f"v{i}" for i in range(2500)], namespace="courses")
    assert [len(ids) for ids, _ in index.deleted] == [1000, 1000, 500]
    assert {ns for _, ns in index.deleted} == {"courses"}
    for bad in (None, ""):
        with pytest.raises(ValueError, match="default"):
            store.delete_ids(["v1"], namespace=bad)


# --- the runner -----------------------------------------------------------------------


def _chunk(vid):
    return Chunk(vid, "text", {"content_hash": "h"})


def test_runner_prune_deletes_only_when_allowed_and_never_on_a_dry_run(capsys):
    from preprocessing.ingest.runner import _prune

    existing = {f"p{i}#biography" for i in range(100)} | {"gone#biography"}
    grouped = {"people": [_chunk(f"p{i}#biography") for i in range(100)]}

    dry = _ListingIndex([sorted(existing)])
    assert _prune(PineconeStore(dry), grouped, 0.1, dry_run=True) is False
    assert dry.deleted == [] and "would delete" in capsys.readouterr().out

    wet = _ListingIndex([sorted(existing)])
    assert _prune(PineconeStore(wet), grouped, 0.1, dry_run=False) is False
    assert wet.deleted == [(["gone#biography"], "people")]


def test_runner_prune_reports_a_refusal_and_deletes_nothing(capsys):
    from preprocessing.ingest.runner import _prune

    index = _ListingIndex([[f"p{i}#biography" for i in range(100)]])
    grouped = {"people": [_chunk("p0#biography")]}  # 99 of 100 would go
    assert _prune(PineconeStore(index), grouped, 0.1, dry_run=False) is True
    assert index.deleted == []
    assert "REFUSED" in capsys.readouterr().out


def test_prune_cannot_be_combined_with_limit(monkeypatch):
    from preprocessing.ingest import runner

    monkeypatch.setattr(sys, "argv", ["ingest", "--bucket", "b", "--prune", "--limit", "5"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2  # argparse error, before anything is read


@pytest.mark.parametrize("value, on", [("1", True), ("true", True), ("ON", True), ("0", False), ("", False),
                                       ("prune", False)])
def test_prune_env_flag_is_strict(monkeypatch, value, on):
    from shared.settings import env_flag

    monkeypatch.setenv("KMP_INGEST_PRUNE", value)
    assert env_flag("KMP_INGEST_PRUNE") is on

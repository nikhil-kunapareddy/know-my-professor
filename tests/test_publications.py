"""Offline tests for the OpenAlex publications source and its job."""

from __future__ import annotations

import json

import pytest

from preprocessing.sources.publications import runner as runner_mod
from preprocessing.sources.publications.config import MAX_CANDIDATES, MAX_LISTED_WORKS
from preprocessing.sources.publications.openalex import (
    AuthorIndex,
    BudgetExhausted,
    OpenAlexClient,
    clean_title,
    parse_author,
    parse_work,
    reconstruct_abstract,
)
from preprocessing.sources.publications.runner import (
    PublicationsJob,
    build_prompt,
    build_record,
    input_hash,
    merge_works,
)
from shared.gcs import LocalStore

# --- parsing ------------------------------------------------------------------


def test_reconstruct_abstract_orders_words_and_truncates():
    inverted = {"typing": [1], "Gradual": [0], "is": [2], "sound": [3]}
    assert reconstruct_abstract(inverted) == "Gradual typing is sound"
    assert reconstruct_abstract(inverted, limit=7) == "Gradual"
    assert reconstruct_abstract(None) == ""


def test_clean_title_strips_markup_and_whitespace():
    assert clean_title("  Growth of <i>E. coli</i>\n in  biofilms ") == "Growth of E. coli in biofilms"
    assert clean_title(None) == ""


def test_parse_author_and_work_shorten_ids():
    author = parse_author({
        "id": "https://openalex.org/A5088073215", "display_name": "Olga Vitek",
        "display_name_alternatives": ["O. Vitek"], "works_count": "42",
        "topics": [{"display_name": f"T{i}"} for i in range(9)],
    })
    assert author["id"] == "A5088073215"
    assert author["works_count"] == 42
    assert len(author["topics"]) == 5

    work = parse_work({
        "id": "https://openalex.org/W1", "title": "<b>Proteomics</b>", "publication_year": 2025,
        "primary_location": {"source": {"display_name": "Nature Methods"}},
    })
    assert (work["id"], work["title"], work["venue"], work["abstract"]) == ("W1", "Proteomics", "Nature Methods", "")


def test_parse_work_tolerates_a_missing_source():
    assert parse_work({"id": "W2", "primary_location": {"source": None}})["venue"] == ""


# --- author index -------------------------------------------------------------


def _author(id_, name, works, alternatives=()):
    return {"id": id_, "name": name, "alternatives": list(alternatives), "works_count": works, "topics": []}


def test_author_index_matches_alternative_names_largest_first():
    index = AuthorIndex([
        _author("A2", "Olga Vitek", 10),
        _author("A1", "O. Vitek", 50, alternatives=["Olga Vitek"]),
        _author("A3", "Olga Vitek", 10),
        _author("A4", "Oleg Vitek", 99),
    ])
    assert index.size == 4
    assert [a["id"] for a in index.candidates("Dr. Olga Vitek")] == ["A1", "A2", "A3"]
    assert index.candidates(None) == []


def test_author_index_caps_candidates():
    index = AuthorIndex([_author(f"A{i}", "Wei Wang", i) for i in range(MAX_CANDIDATES + 3)])
    assert len(index.candidates("Wei Wang")) == MAX_CANDIDATES


# --- record building ----------------------------------------------------------


def _work(id_, title, date, year=2025):
    return {"id": id_, "title": title, "year": year, "date": date, "venue": "", "doi": "", "type": "article", "abstract": ""}


def test_merge_works_dedupes_by_id_and_title_newest_first():
    a = [_work("W1", "Sound Gradual Migration", "2025-01-01"), _work("W2", "Old Paper", "2021-05-05")]
    b = [_work("W1", "Sound Gradual Migration", "2025-01-01"),
         _work("W3", "sound gradual migration", "2024-06-01"),  # the preprint of W1
         _work("W4", "", "2026-01-01")]
    merged = merge_works([a, b])
    assert [w["id"] for w in merged] == ["W1", "W2"]


def test_merge_works_caps_the_list():
    works = [_work(f"W{i}", f"Paper {i}", f"2025-01-{i + 1:02d}") for i in range(MAX_LISTED_WORKS + 5)]
    assert len(merge_works([works])) == MAX_LISTED_WORKS


PROFILE = {"slug": "rosen-david", "college": "coe", "name": "David Rosen", "title": "Assistant Professor",
           "research_interests": ["robot perception"], "biography": "Builds certifiable estimators."}
CANDIDATES = [_author("A1", "David Rosen", 80), _author("A2", "David Rosen", 12)]


def test_build_record_keeps_only_offered_authors_in_candidate_order():
    works = {"A1": [_work("W1", "Certifiable SLAM", "2025-02-01")], "A2": [_work("W2", "Bond Yields", "2024-01-01")]}
    decision = {"matching_author_ids": ["A9", "A1"], "excluded_work_ids": [], "themes": " Certifiable estimation. "}
    record = build_record(PROFILE, CANDIDATES, works, decision, "sha256:x", "m")

    assert record["openalex_ids"] == ["A1"]
    assert record["url"] == "https://openalex.org/A1"
    assert [w["id"] for w in record["works"]] == ["W1"]
    assert record["themes"] == "Certifiable estimation."
    assert record["college"] == "coe"
    assert record["candidates_considered"] == ["A1", "A2"]


def test_build_record_drops_works_the_model_excluded_from_a_merged_record():
    works = {"A1": [_work("W1", "Offshore Wind Fatigue", "2026-01-01"),
                    _work("W2", "Bog Turtle Nesting Ecology", "2024-05-01")]}
    decision = {"matching_author_ids": ["A1"], "excluded_work_ids": ["W2", "W404"], "themes": "Wind structures."}
    record = build_record(PROFILE, CANDIDATES, works, decision, "h", "m")

    assert [w["id"] for w in record["works"]] == ["W1"]
    # Only ids that were really offered are recorded; W404 never was.
    assert record["excluded_work_ids"] == ["W2"]


def test_build_record_without_accepted_works_has_no_themes():
    decision = {"matching_author_ids": [], "excluded_work_ids": [], "themes": "Should not survive."}
    record = build_record(PROFILE, CANDIDATES, {}, decision, "h", "m")
    assert (record["works"], record["themes"], record["url"]) == ([], "", "")


def test_build_prompt_labels_each_work_with_its_id():
    works = [(CANDIDATES[0], [_work("W1", "Certifiable SLAM", "2025-02-01")]), (CANDIDATES[1], [])]
    prompt = build_prompt(PROFILE, works)
    assert "- [W1] 2025: Certifiable SLAM" in prompt
    assert "[A2] David Rosen -- 12 works" in prompt
    assert "(no works in the last" in prompt
    assert "robot perception" in prompt


def test_input_hash_moves_with_works_count_only():
    same = input_hash(PROFILE, CANDIDATES)
    assert input_hash(dict(PROFILE, biography="changed"), CANDIDATES) == same  # works are the signal, not prose
    grown = [dict(CANDIDATES[0], works_count=81), CANDIDATES[1]]
    assert input_hash(PROFILE, grown) != same


# --- network ------------------------------------------------------------------


class _Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


def test_client_stops_on_a_spent_budget_without_retrying():
    session = _Session([_Resp(429, headers={"x-ratelimit-remaining": "0", "x-ratelimit-credits-used": "1"})])
    client = OpenAlexClient(session=session, delay=0)
    with pytest.raises(BudgetExhausted):
        client.get("/works", {})
    assert len(session.calls) == 1
    assert client.credits_used == 1


def test_client_pages_authors_with_the_cursor_and_counts_credits():
    page1 = _Resp(body={"results": [{"id": "A1", "display_name": "Ann Lee"}], "meta": {"next_cursor": "c2"}},
                  headers={"x-ratelimit-credits-used": "1"})
    page2 = _Resp(body={"results": [{"id": "A2", "display_name": "Bo Kim"}], "meta": {"next_cursor": "c3"}},
                  headers={"x-ratelimit-credits-used": "1"})
    page3 = _Resp(body={"results": [], "meta": {"next_cursor": None}})
    session = _Session([page1, page2, page3])
    client = OpenAlexClient(session=session, api_key="k", delay=0)

    assert [a["id"] for a in client.northeastern_authors()] == ["A1", "A2"]
    assert [p["cursor"] for _, p in session.calls] == ["*", "c2", "c3"]
    assert all(p["api_key"] == "k" and p["mailto"] for _, p in session.calls)
    assert client.credits_used == 2


# --- the job ------------------------------------------------------------------


class _Client:
    def __init__(self, works=None, raise_budget=False):
        self.works = works or {}
        self.raise_budget = raise_budget

    def recent_works(self, author_id, since):
        if self.raise_budget:
            raise BudgetExhausted("spent")
        return self.works.get(author_id, [])


class _Claude:
    model = "fake-model"

    def __init__(self, decision):
        self.decision = decision

    def json(self, prompt, schema, max_tokens):
        return self.decision


def test_job_writes_the_record_under_the_entity_key(tmp_path):
    store = LocalStore(tmp_path)
    client = _Client({"A1": [_work("W1", "Certifiable SLAM", "2025-02-01")]})
    decision = {"matching_author_ids": ["A1"], "excluded_work_ids": [], "themes": "SLAM."}
    job = PublicationsJob(client, _Claude(decision), store, since="2021-01-01")

    assert job.process(PROFILE, CANDIDATES, "sha256:x") == "matched"
    record = json.loads((tmp_path / "publications" / "coe-rosen-david.json").read_text())
    assert record["input_hash"] == "sha256:x"
    assert record["model"] == "fake-model"


def test_job_writes_nothing_when_the_model_fails(tmp_path):
    job = PublicationsJob(_Client(), _Claude(None), LocalStore(tmp_path), since="2021-01-01")
    assert job.process(PROFILE, CANDIDATES, "h") == "model_failed"
    assert not (tmp_path / "publications").exists()


def test_job_stops_everyone_once_the_budget_is_spent(tmp_path):
    job = PublicationsJob(_Client(raise_budget=True), _Claude({}), LocalStore(tmp_path), since="2021-01-01")
    assert job.process(PROFILE, CANDIDATES, "h") == "budget_exhausted"
    assert job.stop.is_set()
    assert job.process(PROFILE, CANDIDATES, "h") == "not_reached"


def test_select_profiles_keeps_ingestable_profiles_of_the_asked_colleges(tmp_path):
    store = LocalStore(tmp_path)
    rows = {
        "jane-doe": {"slug": "jane-doe", "name": "Jane Doe", "biography": "Types."},
        "cos-ann-lee": {"slug": "ann-lee", "college": "cos", "name": "Ann Lee", "biography": "Cells."},
        "cos-empty": {"slug": "empty", "college": "cos", "name": "Empty Person"},
    }
    for key, row in rows.items():
        store.write_text(f"profiles/{key}.json", json.dumps(row))

    assert {p["slug"] for p in runner_mod.select_profiles(store, None)} == {"jane-doe", "ann-lee"}
    assert [p["slug"] for p in runner_mod.select_profiles(store, {"cos"})] == ["ann-lee"]

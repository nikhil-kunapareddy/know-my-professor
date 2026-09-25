"""Offline tests for the NSF/NIH grants source and its job."""

from __future__ import annotations

from preprocessing.sources.grants.apis import (
    NihClient,
    NsfClient,
    latest_per_project,
    nsf_date,
    parse_nih,
    parse_nsf,
)
from preprocessing.sources.grants.runner import (
    award_key,
    build_record,
    empty_record,
    match_awards,
    summarise,
    summary_hash,
)
from preprocessing.sources.names import ProfileIndex

# --- parsing ------------------------------------------------------------------


def test_nsf_date_converts_to_iso():
    assert nsf_date("07/29/2026") == "2026-07-29"
    assert nsf_date("") == nsf_date(None) == nsf_date("2026-07-29") == ""


def test_parse_nsf_strips_emails_from_co_pis():
    award = parse_nsf({
        "id": "2552085", "title": " Human  Centered AI ", "abstractText": " Abstract. ",
        "piFirstName": "Jane", "piLastName": "Doe",
        "coPDPI": ["Malihe Alikhani m.alikhani@northeastern.edu", "  "],
        "startDate": "07/01/2026", "expDate": "09/30/2029", "estimatedTotalAmt": "845345.00",
        "fundProgramName": "HCC",
    })
    assert award["investigators"] == [{"name": "Jane Doe", "role": "PI"}, {"name": "Malihe Alikhani", "role": "Co-PI"}]
    assert (award["title"], award["abstract"], award["amount"]) == ("Human Centered AI", "Abstract.", 845345)
    assert (award["start"], award["end"], award["amount_basis"]) == ("2026-07-01", "2029-09-30", "total")
    assert award["url"].endswith("AWD_ID=2552085")


def test_parse_nih_marks_the_contact_pi_and_names_the_core_project():
    award = parse_nih({
        "appl_id": 111, "core_project_num": "R01ES033792", "project_num": "5R01ES033792-03",
        "project_title": "Exposure", "fiscal_year": 2025, "award_amount": 412000,
        "project_start_date": "2021-04-01T00:00:00", "project_end_date": "2026-03-31T00:00:00",
        "agency_ic_admin": {"abbreviation": "NIEHS"},
        "principal_investigators": [
            {"first_name": "Ann", "last_name": "Lee", "is_contact_pi": True},
            {"full_name": "Bo Kim", "is_contact_pi": False},
        ],
    })
    assert award["award_id"] == "R01ES033792"
    assert award["investigators"] == [{"name": "Ann Lee", "role": "Contact PI"}, {"name": "Bo Kim", "role": "PI"}]
    assert (award["end"], award["amount_basis"], award["program"]) == ("2026-03-31", "FY2025", "NIEHS")


def test_latest_per_project_keeps_the_newest_fiscal_year():
    rows = [
        parse_nih({"appl_id": 1, "core_project_num": "R01X", "fiscal_year": 2023, "award_amount": 1}),
        parse_nih({"appl_id": 3, "core_project_num": "R01X", "fiscal_year": 2025, "award_amount": 3}),
        parse_nih({"appl_id": 2, "core_project_num": "R01X", "fiscal_year": 2024, "award_amount": 2}),
        parse_nih({"appl_id": 4, "core_project_num": "K01Y", "fiscal_year": 2024, "award_amount": 4}),
    ]
    latest = {a["award_id"]: a for a in latest_per_project(rows)}
    assert latest["R01X"]["amount"] == 3
    assert set(latest) == {"R01X", "K01Y"}
    assert not any(k.startswith("_") for a in latest.values() for k in a)


# --- pagination ---------------------------------------------------------------


class _Resp:
    def __init__(self, body):
        self.status_code = 200
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        pass


class _Session:
    def __init__(self, bodies):
        self.bodies = list(bodies)
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return _Resp(self.bodies.pop(0))

    def post(self, url, json=None, timeout=None):
        self.calls.append(json)
        return _Resp(self.bodies.pop(0))


def _nsf_page(ids, total):
    return {"response": {"award": [{"id": i, "piFirstName": "A", "piLastName": "B"} for i in ids],
                         "metadata": {"totalCount": total}}}


def test_nsf_pages_with_one_based_offsets_until_the_total():
    session = _Session([_nsf_page(["1", "2"], 3), _nsf_page(["3"], 3)])
    awards = NsfClient(session=session, delay=0).awards("2023-09-25")
    assert [a["award_id"] for a in awards] == ["1", "2", "3"]
    assert [p["offset"] for p in session.calls] == [1, 3]
    assert session.calls[0]["expDateStart"] == "09/25/2023"


def test_nih_pages_until_the_total_then_keeps_the_newest_year():
    rows = [{"appl_id": i, "core_project_num": "R01X", "fiscal_year": 2020 + i} for i in (1, 2)]
    session = _Session([{"results": rows[:1], "meta": {"total": 2}}, {"results": rows[1:], "meta": {"total": 2}}])
    projects = NihClient(session=session, delay=0).projects("2023-09-25")
    assert [p["amount_basis"] for p in projects] == ["FY2022"]
    assert [c["offset"] for c in session.calls] == [0, 1]


# --- matching and records -----------------------------------------------------


def _award(agency, award_id, end, investigators, abstract="An abstract.", title="T"):
    return {"agency": agency, "award_id": award_id, "title": title, "abstract": abstract, "program": "",
            "start": "2024-01-01", "end": end, "amount": 100, "amount_basis": "total", "url": "u",
            "investigators": investigators}


JANE = {"slug": "jane-doe", "name": "Jane Doe", "title": "Professor"}
JANE_COS = {"slug": "jane-doe", "college": "cos", "name": "Jane Doe"}  # a joint appointment
ANN = {"slug": "ann-lee", "college": "coe", "name": "Ann Lee"}


def test_match_awards_attaches_each_award_once_per_professor():
    award = _award("NSF", "1", "2027-01-01", [{"name": "Jane Doe", "role": "PI"},
                                              {"name": "Doe, Jane", "role": "Co-PI"},
                                              {"name": "Ann Lee", "role": "Co-PI"}])
    matched = match_awards([award], ProfileIndex([JANE, JANE_COS, ANN]))

    assert set(matched) == {"jane-doe", "cos-jane-doe", "coe-ann-lee"}
    assert [(a["award_id"], role) for a, role in matched["jane-doe"][1]] == [("1", "PI")]
    assert matched["coe-ann-lee"][1][0][1] == "Co-PI"


def test_build_record_fixes_status_and_sorts_newest_ending_first():
    old = _award("NIH", "R01X", "2025-03-31", [])
    new = _award("NSF", "2", "2029-09-30", [])
    record = build_record(JANE, [(old, "PI"), (new, "Co-PI")], {"NSF-2": "Summary."}, today="2026-09-25")

    assert [(a["award_id"], a["status"], a["summary"]) for a in record["awards"]] == [
        ("2", "active", "Summary."), ("R01X", "ended", "")]
    assert "investigators" not in record["awards"][0] and "abstract" not in record["awards"][0]
    assert (record["college"], record["professor_name"]) == ("khoury", "Jane Doe")


def test_empty_record_keeps_identity_and_drops_awards():
    record = build_record(ANN, [(_award("NSF", "2", "2029-09-30", []), "PI")], {}, today="2026-09-25")
    emptied = empty_record(record)
    assert emptied["awards"] == []
    assert (emptied["slug"], emptied["college"]) == ("ann-lee", "coe")


# --- summaries ----------------------------------------------------------------


class _Claude:
    model = "fake"

    def __init__(self, fail_on=()):
        self.prompts = []
        self.fail_on = fail_on

    def text(self, prompt, max_tokens):
        self.prompts.append(prompt)
        if any(f in prompt for f in self.fail_on):
            raise RuntimeError("boom")
        return "A summary."


def test_summarise_skips_unchanged_awards_and_never_raises():
    cached = _award("NSF", "1", "2027-01-01", [], title="Cached")
    changed = _award("NSF", "2", "2027-01-01", [], title="Changed")
    no_abstract = _award("NIH", "R01X", "2027-01-01", [], abstract="")
    failing = _award("NSF", "3", "2027-01-01", [], title="Explodes")
    cache = {
        award_key(cached): {"hash": summary_hash(cached), "summary": "Old."},
        award_key(changed): {"hash": "stale", "summary": "Old."},
    }
    claude = _Claude(fail_on=("Explodes",))
    outcomes = summarise([cached, changed, no_abstract, failing], cache, claude, workers=2)

    assert outcomes == {"cached": 1, "summarised": 2, "failed": 1}
    assert cache[award_key(cached)]["summary"] == "Old."
    assert cache[award_key(changed)] == {"hash": summary_hash(changed), "summary": "A summary."}
    assert cache[award_key(no_abstract)]["summary"] == ""  # no call for an empty abstract
    assert award_key(failing) not in cache  # retried next run
    assert len(claude.prompts) == 2

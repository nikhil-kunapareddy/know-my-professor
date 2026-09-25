"""NSF Award Search and NIH RePORTER clients, normalised to one award shape.

Both agencies describe the same thing differently, so each parser maps its
payload onto::

    {agency, award_id, title, abstract, program, start, end, amount,
     amount_basis, url, investigators: [{name, role}]}

with ISO dates. The parsers are pure; only the ``*Client`` classes touch the
network.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime

import requests

from shared.retry import with_backoff

from ..pacing import Pacer
from .config import (
    MAX_RETRIES,
    NIH_API,
    NIH_FIELDS,
    NIH_ORG,
    NIH_PAGE_SIZE,
    NIH_PROJECT_URL,
    NSF_API,
    NSF_AWARD_URL,
    NSF_AWARDEE,
    NSF_FIELDS,
    NSF_PAGE_SIZE,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)

_EMAIL = re.compile(r"\s*\S+@\S+\s*")
_SPACE = re.compile(r"\s+")


class _Transient(Exception):
    pass


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (_Transient, requests.ConnectionError, requests.Timeout))


def _clean(text: str | None) -> str:
    return _SPACE.sub(" ", text or "").strip()


def _amount(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# --- NSF ----------------------------------------------------------------------


def nsf_date(value: str | None) -> str:
    """NSF writes ``07/29/2026``; everything downstream compares ISO strings."""
    try:
        return datetime.strptime(value or "", "%m/%d/%Y").date().isoformat()
    except ValueError:
        return ""


def parse_nsf(raw: dict) -> dict:
    pi = _clean(f"{raw.get('piFirstName') or ''} {raw.get('piLastName') or ''}") or _clean(raw.get("pdPIName"))
    investigators = [{"name": pi, "role": "PI"}] if pi else []
    # "Malihe Alikhani m.alikhani@northeastern.edu" -- the email is not part of the name.
    for co in raw.get("coPDPI") or []:
        name = _clean(_EMAIL.sub(" ", co))
        if name:
            investigators.append({"name": name, "role": "Co-PI"})
    return {
        "agency": "NSF",
        "award_id": str(raw.get("id") or ""),
        "title": _clean(raw.get("title")),
        "abstract": (raw.get("abstractText") or "").strip(),
        "program": _clean(raw.get("fundProgramName")),
        "start": nsf_date(raw.get("startDate")),
        "end": nsf_date(raw.get("expDate")),
        "amount": _amount(raw.get("estimatedTotalAmt")),
        "amount_basis": "total",
        "url": NSF_AWARD_URL.format(id=raw.get("id")),
        "investigators": investigators,
    }


# --- NIH ----------------------------------------------------------------------


def parse_nih(raw: dict) -> dict:
    investigators = []
    for pi in raw.get("principal_investigators") or []:
        name = _clean(f"{pi.get('first_name') or ''} {pi.get('last_name') or ''}") or _clean(pi.get("full_name"))
        if name:
            investigators.append({"name": name, "role": "Contact PI" if pi.get("is_contact_pi") else "PI"})
    fiscal_year = raw.get("fiscal_year")
    return {
        "agency": "NIH",
        # The core number (R01ES033792) names the project across its yearly renewals.
        "award_id": raw.get("core_project_num") or raw.get("project_num") or "",
        "title": _clean(raw.get("project_title")),
        "abstract": (raw.get("abstract_text") or "").strip(),
        "program": ((raw.get("agency_ic_admin") or {}).get("abbreviation") or "").strip(),
        "start": (raw.get("project_start_date") or "")[:10],
        "end": (raw.get("project_end_date") or "")[:10],
        "amount": _amount(raw.get("award_amount")),
        "amount_basis": f"FY{fiscal_year}" if fiscal_year else "latest year",
        "url": NIH_PROJECT_URL.format(appl_id=raw.get("appl_id")),
        "investigators": investigators,
        "_fiscal_year": fiscal_year or 0,
        "_appl_id": raw.get("appl_id") or 0,
    }


def latest_per_project(projects: list[dict]) -> list[dict]:
    """RePORTER returns one row per fiscal year; keep each project's newest."""
    best: dict[str, dict] = {}
    for p in projects:
        key = p["award_id"]
        rank = (p["_fiscal_year"], p["_appl_id"])
        if key and (key not in best or rank > (best[key]["_fiscal_year"], best[key]["_appl_id"])):
            best[key] = p
    return [{k: v for k, v in p.items() if not k.startswith("_")} for p in best.values()]


# --- network ------------------------------------------------------------------


class _Http:
    """One paced, retrying session per API."""

    label = "http"

    def __init__(self, session: requests.Session | None = None, delay: float = REQUEST_DELAY_SECONDS):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.pacer = Pacer(delay)
        self.requests = 0

    def _send(self, request: Callable[[], requests.Response]) -> dict:
        def call() -> dict:
            self.pacer.wait()
            response = request()
            self.requests += 1
            if response.status_code == 429 or response.status_code >= 500:
                raise _Transient(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()

        return with_backoff(call, is_retryable=_is_transient, max_attempts=MAX_RETRIES, label=self.label)


class NsfClient(_Http):
    label = "nsf"

    def awards(self, ending_after: str) -> list[dict]:
        """Northeastern awards expiring on or after ``ending_after`` (ISO date)."""
        since = datetime.fromisoformat(ending_after).strftime("%m/%d/%Y")
        awards: list[dict] = []
        offset = 1  # NSF offsets are 1-based
        while True:
            params = {
                "awardeeName": NSF_AWARDEE, "expDateStart": since,
                "printFields": NSF_FIELDS, "rpp": NSF_PAGE_SIZE, "offset": offset,
            }
            body = self._send(lambda p=params: self.session.get(NSF_API, params=p, timeout=REQUEST_TIMEOUT_SECONDS))
            response = body.get("response") or {}
            page = response.get("award") or []
            awards.extend(parse_nsf(a) for a in page)
            total = int((response.get("metadata") or {}).get("totalCount") or 0)
            offset += len(page)
            if not page or offset > total:
                return awards


class NihClient(_Http):
    label = "nih"

    def projects(self, ending_after: str) -> list[dict]:
        """Northeastern projects ending on or after ``ending_after``, newest year each."""
        rows: list[dict] = []
        offset = 0
        while True:
            payload = {
                "criteria": {
                    "org_names": [NIH_ORG],
                    "project_end_date": {"from_date": ending_after, "to_date": "2100-01-01"},
                },
                "include_fields": NIH_FIELDS,
                "offset": offset,
                "limit": NIH_PAGE_SIZE,
            }
            body = self._send(lambda p=payload: self.session.post(NIH_API, json=p, timeout=REQUEST_TIMEOUT_SECONDS))
            page = body.get("results") or []
            rows.extend(parse_nih(p) for p in page)
            offset += len(page)
            if not page or offset >= int((body.get("meta") or {}).get("total") or 0):
                return latest_per_project(rows)

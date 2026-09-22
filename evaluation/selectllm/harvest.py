"""Searching the raw corpus offline, to build ground truth from it.

Ground truth is harvested here rather than from retriever or model output. That
is the whole methodological point: expectations taken from what the retriever
returned would guarantee high recall and measure nothing, and expectations taken
from a model's answer would grade the models on their own work.

Mirrors ``ProfileSource.is_ingestable`` exactly. A person whose profile is in
GCS but too thin to ingest has no vectors, so a question about them is
unanswerable and would score every arm as wrong for a reason that has nothing
to do with the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[2] / ".corpus"

#: Fields a question can be answered from. ``education`` is searchable but is a
#: degree list -- CLAUDE.md records that "PhD from MIT" does not embed near it,
#: so it is a known retrieval gap rather than good question material.
SEARCHABLE = ("biography", "research_interests", "areas_of_interest",
              "labs_and_groups", "projects", "education")

#: Khoury profiles predate the multi-college scrape and carry no ``college``
#: field, so an absent field means khoury. This mirrors
#: ``preprocessing.sources.entities.college_of``, which cannot be imported here:
#: ruff's layering rule keeps ``evaluation`` off ``preprocessing``. The two are
#: pinned to agree by ``tests/test_selectllm.py``, which may import both.
DEFAULT_COLLEGE = "khoury"


def college_of(record: dict) -> str:
    """The college a profile record belongs to.

    Byte-for-byte the upstream behaviour, deliberately including the absence of
    any normalisation: the original does not lower-case, so neither does this.
    Normalising here would have quietly classed a record whose field reads
    "COS" as a college that ``INGESTED_COLLEGES`` does not contain, dropping 498
    College of Science people out of every harvest.
    """
    return record.get("college") or DEFAULT_COLLEGE


#: Colleges with vectors in the index. camd (10) and damore-mckim (12) records
#: exist in GCS but were never scraped into the live corpus, so a question about
#: them is unanswerable.
INGESTED_COLLEGES = {"khoury", "cos"}


@dataclass(frozen=True)
class Person:
    slug: str
    name: str
    title: str
    college: str
    record: dict

    def text(self, fields: tuple[str, ...] = SEARCHABLE) -> str:
        return "\n".join(_flatten(self.record.get(f)) for f in fields)

    def snippet(self, pattern: re.Pattern, width: int = 240) -> str:
        """The first matching window, for auditing why this person was selected."""
        body = self.text()
        m = pattern.search(body)
        if not m:
            return ""
        start = max(0, m.start() - width // 2)
        return " ".join(body[start : start + width].split())


def _flatten(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_flatten(v) for v in value)
    if isinstance(value, dict):
        return "\n".join(_flatten(v) for v in value.values())
    return str(value)


def _ingestable(record: dict) -> bool:
    """The live filter, copied deliberately rather than imported.

    ``ProfileSource.is_ingestable`` takes the same three fields. Kept as its own
    function so this module needs no Source instance, but it must not drift --
    ``test_selectllm.py`` asserts the two agree.
    """
    return bool(
        record.get("biography")
        or record.get("research_interests")
        or record.get("areas_of_interest")
    )


@cache
def people() -> tuple[Person, ...]:
    """Every profile that actually has vectors in the index."""
    out = []
    for path in sorted((CORPUS / "profiles").glob("*.json")):
        record = json.loads(path.read_text())
        college = college_of(record)
        if college not in INGESTED_COLLEGES or not _ingestable(record):
            continue
        out.append(
            Person(
                slug=record.get("slug") or path.stem,
                name=record.get("name") or "",
                title=record.get("title") or "",
                college=college,
                record=record,
            )
        )
    return tuple(out)


@cache
def courses() -> tuple[dict, ...]:
    """Catalog records, if the courses prefix has been pulled."""
    d = CORPUS / "courses"
    if not d.exists():
        return ()
    return tuple(json.loads(p.read_text()) for p in sorted(d.glob("*.json")))


@cache
def schedule() -> tuple[dict, ...]:
    """Banner term records -- who teaches what, this term."""
    d = CORPUS / "schedule"
    if not d.exists():
        return ()
    return tuple(json.loads(p.read_text()) for p in sorted(d.glob("*.json")))


def search(
    *terms: str,
    fields: tuple[str, ...] = SEARCHABLE,
    college: str | None = None,
) -> list[tuple[Person, str]]:
    """Everyone whose searchable text matches ANY term, with an audit snippet.

    Terms are treated as whole-word-ish regexes so that "ai" does not match
    "said" and "ml" does not match "html" -- the failure mode that makes a naive
    keyword harvest produce a ground truth full of people who do not qualify.
    """
    pattern = re.compile("|".join(rf"\b{t}\b" if " " not in t else re.escape(t)
                                  for t in terms), re.I)
    hits = []
    for p in people():
        if college and p.college != college:
            continue
        if pattern.search(p.text(fields)):
            hits.append((p, p.snippet(pattern)))
    return hits

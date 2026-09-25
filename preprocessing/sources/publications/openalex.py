"""OpenAlex REST client, response parsing, and the author name index.

Everything that touches the network is in ``OpenAlexClient``; the parsing
helpers are pure functions so tests can drive them with recorded payloads.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator

import requests

from shared.retry import with_backoff

from ..names import name_key
from ..pacing import Pacer
from .config import (
    ABSTRACT_CHARS,
    AUTHOR_FIELDS,
    AUTHOR_FILTER,
    CONTACT_EMAIL,
    MAX_CANDIDATES,
    MAX_RETRIES,
    OPENALEX_BASE,
    PAGE_SIZE,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    TOPICS_SHOWN,
    USER_AGENT,
    WORK_FIELDS,
    WORK_TYPES,
    WORKS_PER_CANDIDATE,
)

_TAGS = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")


class BudgetExhausted(Exception):
    """OpenAlex's daily credit budget is spent.

    Never retried: the budget resets at midnight UTC, not in seconds. The runner
    stops cleanly on it, and the next run resumes where this one stopped.
    """


class _Transient(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


def _is_transient(exc: BaseException) -> bool:
    return isinstance(exc, (_Transient, requests.ConnectionError, requests.Timeout))


# --- parsing (pure) ---------------------------------------------------------


def short_id(url: str | None) -> str:
    """``"https://openalex.org/A5088073215"`` -> ``"A5088073215"``."""
    return (url or "").rstrip("/").rsplit("/", 1)[-1]


def clean_title(title: str | None) -> str:
    """Titles carry markup (``<i>E. coli</i>``) and stray whitespace."""
    return _SPACE.sub(" ", _TAGS.sub("", title or "")).strip()


def reconstruct_abstract(inverted: dict[str, list[int]] | None, limit: int = ABSTRACT_CHARS) -> str:
    """OpenAlex stores abstracts as word -> positions; put the words back in order."""
    if not inverted:
        return ""
    placed = sorted((pos, word) for word, positions in inverted.items() for pos in positions)
    return " ".join(word for _, word in placed)[:limit]


def parse_author(raw: dict) -> dict:
    return {
        "id": short_id(raw.get("id")),
        "name": raw.get("display_name") or "",
        "alternatives": list(raw.get("display_name_alternatives") or []),
        "works_count": int(raw.get("works_count") or 0),
        "topics": [
            t["display_name"] for t in (raw.get("topics") or [])[:TOPICS_SHOWN] if t.get("display_name")
        ],
    }


def parse_work(raw: dict) -> dict:
    source = (raw.get("primary_location") or {}).get("source") or {}
    return {
        "id": short_id(raw.get("id")),
        "title": clean_title(raw.get("title")),
        "year": raw.get("publication_year"),
        "date": raw.get("publication_date") or "",
        "venue": source.get("display_name") or "",
        "doi": raw.get("doi") or "",
        "type": raw.get("type") or "",
        "abstract": reconstruct_abstract(raw.get("abstract_inverted_index")),
    }


class AuthorIndex:
    """OpenAlex authors grouped by ``name_key`` over every name they go by."""

    def __init__(self, authors: Iterable[dict]):
        self._by_key: dict[str, dict[str, dict]] = defaultdict(dict)
        self.size = 0
        for author in authors:
            self.size += 1
            for name in (author["name"], *author["alternatives"]):
                key = name_key(name)
                if key:
                    self._by_key[key][author["id"]] = author

    def candidates(self, name: str | None, limit: int = MAX_CANDIDATES) -> list[dict]:
        """Same-key authors, most works first (ties by id, so the order is stable)."""
        key = name_key(name)
        found = self._by_key.get(key, {}).values() if key else ()
        return sorted(found, key=lambda a: (-a["works_count"], a["id"]))[:limit]


# --- network ----------------------------------------------------------------


class OpenAlexClient:
    """Paced, retrying GETs against api.openalex.org that track credit spend."""

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None,
                 delay: float = REQUEST_DELAY_SECONDS):
        self.api_key = api_key
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.pacer = Pacer(delay)
        self._lock = threading.Lock()
        self.requests = self.credits_used = 0

    def _record(self, headers) -> None:
        try:
            used = int(headers.get("x-ratelimit-credits-used") or 0)
        except ValueError:
            used = 0
        with self._lock:
            self.requests += 1
            self.credits_used += used

    @staticmethod
    def _budget_spent(headers) -> bool:
        if headers.get("x-ratelimit-remaining") == "0":
            return True
        try:
            return float(headers.get("x-ratelimit-remaining-usd", "1")) <= 0
        except ValueError:
            return False

    def get(self, path: str, params: dict) -> dict:
        params = {**params, "mailto": CONTACT_EMAIL}
        if self.api_key:
            params["api_key"] = self.api_key

        def call() -> dict:
            self.pacer.wait()
            response = self.session.get(
                f"{OPENALEX_BASE}{path}", params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
            self._record(response.headers)
            if response.status_code == 429 and self._budget_spent(response.headers):
                raise BudgetExhausted("OpenAlex daily credit budget spent")
            if response.status_code == 429 or response.status_code >= 500:
                raise _Transient(response.status_code)
            response.raise_for_status()
            return response.json()

        try:
            return with_backoff(call, is_retryable=_is_transient, max_attempts=MAX_RETRIES, label="openalex")
        except _Transient as e:
            # A 429 that outlasts every backoff is the budget, whatever the headers say.
            if e.status == 429:
                raise BudgetExhausted("OpenAlex kept returning 429") from e
            raise

    def northeastern_authors(self) -> Iterator[dict]:
        """Every author ever affiliated with Northeastern, via cursor paging."""
        cursor: str | None = "*"
        while cursor:
            page = self.get("/authors", {
                "filter": AUTHOR_FILTER, "select": AUTHOR_FIELDS,
                "per-page": PAGE_SIZE, "cursor": cursor,
            })
            results = page.get("results") or []
            yield from (parse_author(r) for r in results)
            cursor = (page.get("meta") or {}).get("next_cursor") if results else None

    def recent_works(self, author_id: str, since: str) -> list[dict]:
        """An author's most recent works published on or after ``since`` (ISO date)."""
        page = self.get("/works", {
            "filter": f"author.id:{author_id},from_publication_date:{since},type:{WORK_TYPES}",
            "sort": "publication_date:desc",
            "per-page": WORKS_PER_CANDIDATE,
            "select": WORK_FIELDS,
        })
        return [parse_work(w) for w in page.get("results") or []]

"""Joining a person named by an outside source to a professor in the corpus.

OpenAlex authors and NSF/NIH investigators arrive as bare names; the corpus
knows professors by entity id. Enrichment from those sources is therefore joined
on a normalised name key -- defined once here, because two sources normalising
differently would each attach some records to nobody.

The key is deliberately strict: full first name + last name, accents folded,
middle names and initials dropped. ``"O. Vitek"`` does not match ``"Olga
Vitek"`` -- at a university this size an initial matches too many people.
Nicknames ("Bob" for "Robert") are missed rather than guessed at.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

_EMAIL = re.compile(r"\S+@\S+")
_PARENS = re.compile(r"\([^)]*\)")
_NON_LETTERS = re.compile(r"[^a-z']+")
_HONORIFICS = frozenset({"dr", "prof", "professor", "mr", "mrs", "ms"})
_SUFFIXES = frozenset(
    {"jr", "sr", "ii", "iii", "iv", "phd", "md", "jd", "mba", "esq", "rn", "pharmd", "dnp", "facs"}
)


def _fold(text: str) -> str:
    """Lowercase with accents stripped: ``"Nuñez"`` -> ``"nunez"``."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def name_key(name: str | None) -> str | None:
    """``"Dr. Olga  Vitek, PhD"`` -> ``"olga vitek"``; None if no usable name.

    ``"Vitek, Olga"`` is read last-name-first. Hyphens split, so a surname
    written ``"Smith-Jones"`` in one source and ``"Smith Jones"`` in another
    still meets on its last part.
    """
    if not name:
        return None
    # Dots go first so "Ph.D." reads as the suffix "phd" and "J. Smith" as "j smith".
    text = _fold(_PARENS.sub(" ", _EMAIL.sub(" ", name))).replace(".", "")

    if text.count(",") == 1:
        before, after = (part.strip() for part in text.split(","))
        # "Olga Vitek, PhD" is a suffix, not "Last, First".
        if after not in _SUFFIXES:
            text = f"{after} {before}"

    tokens = [t.strip("'") for t in _NON_LETTERS.split(text)]
    tokens = [
        t for t in tokens
        if len(t) > 1 and t not in _HONORIFICS and t not in _SUFFIXES
    ]
    if len(tokens) < 2:
        return None
    return f"{tokens[0]} {tokens[-1]}"


class ProfileIndex:
    """Profiles grouped by ``name_key``: the corpus side of every name join.

    A key can hold several profiles. Usually that is one person with a joint
    appointment, who has a record in each college's directory and should get
    the enrichment in both.
    """

    def __init__(self, profiles: Iterable[dict]):
        self._by_key: dict[str, list[dict]] = defaultdict(list)
        for profile in profiles:
            key = name_key(profile.get("name"))
            if key:
                self._by_key[key].append(profile)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_key.values())

    def match(self, name: str | None) -> list[dict]:
        """Every profile whose name has the same key as ``name``."""
        key = name_key(name)
        return list(self._by_key.get(key, ())) if key else []

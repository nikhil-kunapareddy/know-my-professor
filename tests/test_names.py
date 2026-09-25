"""The name key both enrichment joins (publications, grants) rely on."""

from __future__ import annotations

import pytest

from preprocessing.sources.names import ProfileIndex, name_key


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("Olga Vitek", "olga vitek"),
        ("Dr. Olga  Vitek, PhD", "olga vitek"),
        ("Olga Vitek, Ph.D.", "olga vitek"),
        ("Vitek, Olga", "olga vitek"),
        ("Amélie Nuñez", "amelie nunez"),
        ("Mary Jo Smith", "mary smith"),
        ("Ann Smith-Jones", "ann jones"),
        ("Robert (Bob) Smith", "robert smith"),
        ("Malihe Alikhani m.alikhani@northeastern.edu", "malihe alikhani"),
        ("Prof. John Smith Jr.", "john smith"),
    ],
)
def test_name_key_normalises(raw, key):
    assert name_key(raw) == key


@pytest.mark.parametrize("raw", [None, "", "Vitek", "O. Vitek", "Dr. Smith"])
def test_name_key_refuses_names_it_cannot_pin_down(raw):
    # An initial matches too many people at a university this size.
    assert name_key(raw) is None


def test_profile_index_returns_every_profile_under_a_key():
    khoury = {"slug": "david-rosen", "name": "David Rosen"}
    coe = {"slug": "rosen-david", "college": "coe", "name": "David M. Rosen"}
    other = {"slug": "jane-doe", "name": "Jane Doe"}
    index = ProfileIndex([khoury, coe, other, {"slug": "x", "name": "X"}])

    assert len(index) == 3  # the unkeyable name is not indexed
    assert index.match("Rosen, David") == [khoury, coe]
    assert index.match("Jane Doe") == [other]
    assert index.match("Nobody Here") == []
    assert index.match(None) == []

"""Shared test harness for the flat-import source folders.

The scraper/, ingest/, and weblinks/ folders are NOT packages — each is meant to
run with its own folder on sys.path (`python scraper.py`), and several share
module names (config.py, gcs.py). So we can't just `import` them. The `load`
fixture loads a module from one source folder with that folder on sys.path,
purging the shared flat names first so e.g. `from config import ...` resolves
against the folder under test.

Some source modules import cloud libs that aren't installed in CI
(trafilatura, google.generativeai, google.cloud.storage); we install light
stubs for the missing ones so the pure logic stays importable and testable.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = {name: ROOT / name for name in ("scraper", "ingest", "weblinks")}

# Every bare module name defined across the three source folders. Purged before
# each load so a fresh import binds to the folder currently on sys.path.
FLAT_NAMES = {
    "config", "models", "fetcher", "profile_parser", "stores", "scraper",
    "chunking", "embedding", "gcs", "pinecone_store", "ingest",
    "crawl", "extract", "weblinks",
}


def _missing(name: str, *attrs: str) -> bool:
    """True if `name` can't be imported, or lacks any of the required attrs.

    The attr check matters when an installed package is the wrong version
    (e.g. an old `pinecone` without the `Pinecone` class).
    """
    try:
        mod = importlib.import_module(name)
    except ImportError:
        return True
    return any(not hasattr(mod, a) for a in attrs)


def _install_stubs() -> None:
    if _missing("pinecone", "Pinecone", "ServerlessSpec"):
        p = types.ModuleType("pinecone")

        class _Pinecone:  # pragma: no cover - not exercised by unit tests
            def __init__(self, *a, **k):
                pass

        class _ServerlessSpec:  # pragma: no cover
            def __init__(self, *a, **k):
                pass

        p.Pinecone = _Pinecone
        p.ServerlessSpec = _ServerlessSpec
        sys.modules["pinecone"] = p

    if _missing("trafilatura"):
        t = types.ModuleType("trafilatura")
        t.extract = lambda html, **kw: None  # tests monkeypatch this
        sys.modules["trafilatura"] = t

    if _missing("google.generativeai"):
        import google  # real namespace package (google.api_core is installed)

        genai = types.ModuleType("google.generativeai")
        genai.configure = lambda **kw: None

        class _GenerativeModel:
            def __init__(self, *a, **k):
                pass

            def generate_content(self, *a, **k):  # pragma: no cover - not called in unit tests
                raise RuntimeError("stub generate_content")

        genai.GenerativeModel = _GenerativeModel
        genai.embed_content = lambda **kw: {"embedding": [0.0]}
        sys.modules["google.generativeai"] = genai
        setattr(google, "generativeai", genai)

    if _missing("google.cloud.storage"):
        import google

        try:
            gcloud = importlib.import_module("google.cloud")
        except ImportError:
            gcloud = types.ModuleType("google.cloud")
            gcloud.__path__ = []  # namespace-style
            sys.modules["google.cloud"] = gcloud
            setattr(google, "cloud", gcloud)

        storage = types.ModuleType("google.cloud.storage")

        class _Client:  # pragma: no cover - not called in unit tests
            def __init__(self, *a, **k):
                pass

        storage.Client = _Client
        sys.modules["google.cloud.storage"] = storage
        setattr(gcloud, "storage", storage)


_install_stubs()


def _load(dir_name: str, module_name: str):
    for n in FLAT_NAMES:
        sys.modules.pop(n, None)
    src = str(SRC[dir_name])
    sys.path.insert(0, src)
    try:
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)
    finally:
        try:
            sys.path.remove(src)
        except ValueError:
            pass


@pytest.fixture
def load():
    """Return loader: load(folder, module) -> imported module from that folder."""
    return _load

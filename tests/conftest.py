"""Test harness for the offline unit tests.

The source is now a proper package (``core``, ``preprocessing``, ``shared``,
``serving``) imported normally — pyproject's ``pythonpath = ["."]`` puts the
repo root on ``sys.path``, so the old flat-import shim is gone.

What remains: some modules import cloud libs that may be absent in a bare
environment (``pinecone``, ``mistralai``, ``trafilatura``,
``google.generativeai``, ``google.cloud.storage``). We install lightweight stubs
for the missing ones so the pure logic stays importable and testable. When the
real libraries are present (e.g. in .venv) the stubs are skipped.
"""

from __future__ import annotations

import importlib
import sys
import types


def _missing(name: str, *attrs: str) -> bool:
    """True if ``name`` can't be imported, or lacks any of the required attrs."""
    try:
        mod = importlib.import_module(name)
    except ImportError:
        return True
    return any(not hasattr(mod, a) for a in attrs)


def _install_stubs() -> None:
    if _missing("pinecone", "Pinecone", "ServerlessSpec"):
        p = types.ModuleType("pinecone")

        class _Pinecone:  # pragma: no cover
            def __init__(self, *a, **k):
                pass

        class _ServerlessSpec:  # pragma: no cover
            def __init__(self, *a, **k):
                pass

        p.Pinecone = _Pinecone
        p.ServerlessSpec = _ServerlessSpec
        sys.modules["pinecone"] = p

    if _missing("mistralai", "Mistral"):
        m = types.ModuleType("mistralai")

        class _Mistral:  # pragma: no cover
            def __init__(self, *a, **k):
                pass

        m.Mistral = _Mistral
        sys.modules["mistralai"] = m

        models = types.ModuleType("mistralai.models")

        class _SDKError(Exception):  # pragma: no cover
            status_code = None

        models.SDKError = _SDKError
        m.models = models
        sys.modules["mistralai.models"] = models

    if _missing("trafilatura"):
        t = types.ModuleType("trafilatura")
        t.extract = lambda html, **kw: None  # tests monkeypatch this
        sys.modules["trafilatura"] = t

    if _missing("google.generativeai"):
        import google  # real namespace package (google.api_core is installed)

        genai = types.ModuleType("google.generativeai")
        genai.configure = lambda **kw: None

        class _GenerativeModel:  # pragma: no cover
            def __init__(self, *a, **k):
                pass

            def generate_content(self, *a, **k):
                raise RuntimeError("stub generate_content")

        genai.GenerativeModel = _GenerativeModel
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

        class _Client:  # pragma: no cover
            def __init__(self, *a, **k):
                pass

        storage.Client = _Client
        sys.modules["google.cloud.storage"] = storage
        setattr(gcloud, "storage", storage)


_install_stubs()

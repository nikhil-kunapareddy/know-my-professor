"""Live end-to-end check of the RAG chain: Mistral embed -> Pinecone -> Llama.

Unlike the rest of tests/ (which is fully offline with stubbed cloud deps), this
hits the REAL APIs and the live Pinecone index, so it is opt-in: skipped unless
KMP_LIVE_E2E is set. Run it from the venv (which has the SDK deps) after an
ingest, to confirm the deployed chain works before/after a cutover:

    KMP_LIVE_E2E=1 .venv/bin/python -m pytest tests/test_e2e_live.py -v -s

Keys are read from ingest/.env (MISTRAL_API_KEY, PINECONE_API_KEY, LLAMA_API_KEY);
environment variables of the same name take precedence if already set.

It reuses production code/constants on purpose:
  - ingest.embedding.embed_texts + ingest.config  (the real embedder + dim/index)
  - api/app.py EMBED_MODEL / DEFAULT_INDEX / DEFAULT_CHAT_MODEL  (query side)
so it also fails loudly if the ingest and API sides ever drift apart.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / "ingest" / ".env"

pytestmark = pytest.mark.skipif(
    not os.environ.get("KMP_LIVE_E2E"),
    reason="live e2e disabled; set KMP_LIVE_E2E=1 to run (hits real APIs + index)",
)

_REQUIRED_KEYS = ["MISTRAL_API_KEY", "PINECONE_API_KEY", "LLAMA_API_KEY"]


def _load_dotenv() -> dict[str, str]:
    """Parse ingest/.env by hand (values may contain '|' etc. that break shell `source`)."""
    env: dict[str, str] = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


@pytest.fixture(scope="module")
def keys() -> dict[str, str]:
    env = _load_dotenv()
    missing = [k for k in _REQUIRED_KEYS if not (os.environ.get(k) or env.get(k))]
    if missing:
        pytest.skip(f"missing keys for live e2e: {missing}")
    # Make keys visible to production code that reads os.environ (e.g. embed client).
    for k in _REQUIRED_KEYS:
        if env.get(k):
            os.environ[k] = env[k]
    return {k: os.environ[k] for k in _REQUIRED_KEYS}


def _load_api_app():
    """Import api/app.py (flat module) for its query-side constants."""
    api_dir = str(ROOT / "api")
    sys.path.insert(0, api_dir)
    try:
        sys.modules.pop("app", None)
        return importlib.import_module("app")
    finally:
        try:
            sys.path.remove(api_dir)
        except ValueError:
            pass


def test_ingest_and_api_agree_on_embedder_and_index(load):
    """Drift guard: the embedder + index must match on both sides or retrieval breaks."""
    config = load("ingest", "config")
    app = _load_api_app()
    assert app.EMBED_MODEL == config.EMBED_MODEL, "ingest vs API embed model drift"
    assert app.DEFAULT_INDEX == config.PINECONE_DEFAULT_INDEX, "ingest vs API index drift"


def test_rag_chain_end_to_end(keys, load):
    config = load("ingest", "config")
    embedding = load("ingest", "embedding")
    app = _load_api_app()

    question = "Who at Khoury works on programming languages or type systems?"

    # 1) Embed the query with the real production embedder; dim must match the index.
    qvec = embedding.embed_texts([question])[0]
    assert len(qvec) == config.EMBED_DIM == 1024

    # 2) Retrieve from the live index.
    from pinecone import Pinecone

    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(config.PINECONE_DEFAULT_INDEX)
    result = index.query(vector=qvec, top_k=5, include_metadata=True)
    matches = result.matches or []
    assert matches, "no vectors retrieved — is the index populated?"

    context = "\n\n".join(
        f"[{i}] {(m.metadata or {}).get('professor_name', '')} "
        f"({(m.metadata or {}).get('professor_title', '')}) — "
        f"{(m.metadata or {}).get('section_type', '')}\n{(m.metadata or {}).get('text', '')}"
        for i, m in enumerate(matches, start=1)
    )

    # 3) Generate with the same Llama model the API uses.
    from llama_api_client import LlamaAPIClient

    client = LlamaAPIClient(api_key=os.environ["LLAMA_API_KEY"])
    resp = client.chat.completions.create(
        model=app.DEFAULT_CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Answer ONLY from the numbered context. Cite sources with [n]."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n"},
        ],
        temperature=0.0,
    )
    answer = (resp.completion_message.content.text or "").strip()
    print("\nE2E answer:", answer)
    assert answer, "empty answer from chat model"
    assert "[" in answer, "answer did not cite any retrieved source"

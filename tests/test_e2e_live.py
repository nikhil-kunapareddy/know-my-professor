"""Live end-to-end check of the RAG chain: Mistral embed -> Pinecone -> Llama.

Unlike the rest of tests/ (fully offline with stubbed cloud deps), this hits the
REAL APIs and the live Pinecone index, so it is opt-in: skipped unless
KMP_LIVE_E2E is set. Run it from the venv after an ingest:

    KMP_LIVE_E2E=1 .venv/bin/python -m pytest tests/test_e2e_live.py -v -s

Keys are read from the repo-root .env (MISTRAL_API_KEY, PINECONE_API_KEY,
LLAMA_API_KEY); env vars of the same name take precedence if already set.

It reuses production code/constants on purpose (the real embedder + shared
config), so it also fails loudly if anything starts hardcoding a divergent
embedder/index instead of pulling from shared.config.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

pytestmark = pytest.mark.skipif(
    not os.environ.get("KMP_LIVE_E2E"),
    reason="live e2e disabled; set KMP_LIVE_E2E=1 to run (hits real APIs + index)",
)

_REQUIRED_KEYS = ["MISTRAL_API_KEY", "PINECONE_API_KEY", "LLAMA_API_KEY"]


def _load_dotenv() -> dict[str, str]:
    """Parse the repo-root .env by hand (values may contain '|' that breaks `source`)."""
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
    for k in _REQUIRED_KEYS:
        if env.get(k):
            os.environ[k] = env[k]
    return {k: os.environ[k] for k in _REQUIRED_KEYS}


def test_embedder_and_index_are_single_sourced():
    """Drift guard: query + document embedders and the index all come from shared.config."""
    from core.query.embedder import QueryEmbedder
    from preprocessing.ingest.embedding import DocumentEmbedder
    from shared.config import EMBED_MODEL, PINECONE_DEFAULT_INDEX

    assert QueryEmbedder(client=None).model == EMBED_MODEL
    assert DocumentEmbedder().model == EMBED_MODEL
    assert PINECONE_DEFAULT_INDEX  # the one place the index name lives


def test_rag_chain_end_to_end(keys):
    from llama_api_client import LlamaAPIClient
    from pinecone import Pinecone

    from core.query.embedder import QueryEmbedder
    from preprocessing.ingest.embedding import DocumentEmbedder
    from shared.config import DEFAULT_CHAT_MODEL, EMBED_DIM, PINECONE_DEFAULT_INDEX

    question = "Who at Khoury works on programming languages or type systems?"

    # 1) Embed the query with the real production embedder; dim must match the index.
    qvec = DocumentEmbedder().embed_texts([question])[0]
    assert len(qvec) == EMBED_DIM == 1024

    # 2) Retrieve from the live index.
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(PINECONE_DEFAULT_INDEX)
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
    client = LlamaAPIClient(api_key=os.environ["LLAMA_API_KEY"])
    resp = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
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

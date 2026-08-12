"""Live end-to-end check of the RAG chain: embed -> Pinecone -> generate.

Unlike the rest of tests/ (fully offline with stubbed cloud deps), this hits the
REAL APIs and the live Pinecone index, so it is opt-in: skipped unless
KMP_LIVE_E2E is set. Run it from the venv after an ingest:

    KMP_LIVE_E2E=1 .venv/bin/python -m pytest tests/test_e2e_live.py -v -s

Keys are read from the repo-root .env (MISTRAL_API_KEY, PINECONE_API_KEY,
LLAMA_API_KEY); env vars of the same name take precedence if already set.

It drives the real ``RAGPipeline`` with real providers, so it exercises the same
objects the deployed service builds — not a parallel reimplementation.
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


def test_rag_chain_end_to_end(keys):
    """The deployed pipeline, assembled the same way the API assembles it."""
    from pinecone import Pinecone

    from core.llm import build_generator
    from core.pipeline import NO_ANSWER, RAGPipeline
    from core.retrieval.pinecone_retriever import PineconeRetriever
    from shared.config import EMBED_DIM, PINECONE_DEFAULT_INDEX
    from shared.embeddings import build_embedder

    embedder = build_embedder()
    question = "Who at Khoury works on programming languages or type systems?"

    # 1) The query embedding must match the index width.
    qvec = embedder.embed_query(question)
    assert len(qvec) == EMBED_DIM

    # 2) The index must actually hold vectors.
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(PINECONE_DEFAULT_INDEX)
    retriever = PineconeRetriever(index)
    assert retriever.retrieve(qvec, top_k=5), "no vectors retrieved — is the index populated?"

    # 3) The full pipeline answers and cites.
    pipeline = RAGPipeline(embedder, retriever, build_generator())
    result = pipeline.answer(question)
    print("\nE2E answer:", result.answer)
    print("E2E timings:", result.timings_ms)

    assert result.answer, "empty answer from chat model"
    assert result.answer != NO_ANSWER, "pipeline found nothing above the score floor"
    assert "[" in result.answer, "answer did not cite any retrieved source"
    assert result.sources


def test_live_index_dimension_matches_config(keys):
    """The index the service queries must be the width the embedder produces."""
    from pinecone import Pinecone

    from shared.config import EMBED_DIM, PINECONE_DEFAULT_INDEX

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    assert pc.describe_index(PINECONE_DEFAULT_INDEX).dimension == EMBED_DIM

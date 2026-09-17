"""Single source of truth for cross-cutting constants.

Only things BOTH the serving side (``core``) and the data side (``preprocessing``)
must agree on belong here — above all the embedding model/dimension and the
Pinecone index. Keeping them in one module is what stops the query embedder and
the document embedder drifting apart (the old layout duplicated these in
``api/app.py`` and ``ingest/config.py`` and relied on a live test to notice).

Config that only one component cares about lives with that component:
  - per-source scrape/extract knobs -> ``preprocessing/sources/<name>/config.py``
  - section taxonomy               -> the ``Source`` that emits it (see the registry)
  - env-var wiring                 -> ``shared/settings.py``
"""

from __future__ import annotations

import os

# --- Embeddings (shared by the query side and preprocessing.ingest) --------

# Query and document vectors MUST be produced by the same model/dim or
# similarity search is meaningless. Both sides resolve the embedder through
# shared.providers, which reads these.
DEFAULT_EMBED_PROVIDER = "mistral"
EMBED_MODEL = "mistral-embed-2312"
EMBED_DIM = 1024

# Mistral's endpoint embeds a list of texts per request, so we batch instead of
# one-call-per-text — collapsing hundreds of requests into a handful.
EMBED_BATCH_SIZE = 64
EMBED_RATE_LIMIT_SLEEP_SECONDS = 1.0
EMBED_MAX_RETRIES = 6

# --- Pinecone --------------------------------------------------------------

# 1024-dim index for Mistral vectors. The old 3072-dim "know-my-professor"
# index (Gemini embeddings) is kept intact for rollback only.
PINECONE_DEFAULT_INDEX = "know-my-professor-m1024"
PINECONE_DEFAULT_CLOUD = "aws"
PINECONE_DEFAULT_REGION = "us-east-1"
UPSERT_BATCH_SIZE = 100
FETCH_BATCH_SIZE = 100

# --- Generation / retrieval (core) -----------------------------------------

# Which Generator the /chat service builds when CHAT_PROVIDER is unset. The
# model itself is NOT here: an id like "claude-opus-5" means nothing to the
# Llama provider, so each Generator carries its own ``default_model`` and this
# file only names the provider. Override the model per deployment with
# CHAT_MODEL, which is only meaningful together with CHAT_PROVIDER.
DEFAULT_CHAT_PROVIDER = "anthropic"
DEFAULT_TOP_K = 8

# Cosine similarity below this is treated as "not really about the question".
# Without a floor, vector search always returns top_k rows, so the pipeline's
# no-answer path could never fire and an unrelated chunk would still be cited.
MIN_RETRIEVAL_SCORE = 0.35

# --- GCS -------------------------------------------------------------------


def gcs_bucket() -> str | None:
    """The configured GCS bucket name (env KMP_GCS_BUCKET), or None."""
    return os.environ.get("KMP_GCS_BUCKET")

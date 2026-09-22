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

# The namespace holding the people corpus (profiles + weblinks). It lives here,
# not on the Source, because BOTH ends must name the same partition: ingest
# writes it and /chat reads it. A mismatch is silent -- the query succeeds
# against an empty partition and every answer becomes the no-answer string --
# so the string exists once rather than twice.
#
# "people", not "professors": the directories call postdocs, PhD students and
# staff "faculty" too, and the corpus keeps whoever has substantive content.
#
PEOPLE_NAMESPACE = "people"

# The course corpus. Lives here for the same reason: serving now names it too.
COURSES_NAMESPACE = "courses"

# Namespaces /chat searches, in order. EVERY listed namespace gets its own
# top_k -- they are not merged into one top_k -- because a cosine score cannot
# tell a person from a course. Measured 2026-09-21 on the golden set: course
# descriptions are *about topics*, so "who works on formal verification?" scores
# a syllabus (0.769) above the right professor's bio (0.729). Merging into one
# top_k displaced the first correct chunk on 3 of 10 people questions. Giving
# each namespace its own slots makes that arithmetically impossible.
CHAT_NAMESPACES: tuple[str, ...] = (PEOPLE_NAMESPACE, COURSES_NAMESPACE)

# --- Generation / retrieval (core) -----------------------------------------

# Which Generator the /chat service builds when CHAT_PROVIDER is unset. The
# model itself is NOT here: an id like "claude-opus-5" means nothing to the
# Llama provider, so each Generator carries its own ``default_model`` and this
# file only names the provider. Override the model per deployment with
# CHAT_MODEL, which is only meaningful together with CHAT_PROVIDER.
DEFAULT_CHAT_PROVIDER = "anthropic"

# Chunks retrieved PER NAMESPACE, not per request: /chat runs one query against
# each namespace in CHAT_NAMESPACES, so the model actually sees
# DEFAULT_TOP_K * len(CHAT_NAMESPACES) chunks -- 22 today, not 11. Raised 8 -> 11
# to widen recall on broad questions ("who works on NLP?"), where the corpus
# holds far more valid people than 8 slots can carry. Override per deployment
# with TOP_K.
DEFAULT_TOP_K = 11

# Cosine similarity below this is treated as "not really about the question".
# Without a floor, vector search always returns top_k rows, so the pipeline's
# no-answer path could never fire and an unrelated chunk would still be cited.
MIN_RETRIEVAL_SCORE = 0.35

# --- Rerank (optional second-stage scoring) --------------------------------

# Reranking is ON. Measured 2026-09-22 over 60 questions
# (evaluation/results/2026-09-22-rerank/notes.md): MRR 0.852 -> 0.900,
# precision@8 59.6% -> 62.7%, concentrated in narrow course questions which
# gain +0.152 MRR from a base of 15.2% precision -- the worst stratum in the
# corpus. A cross-encoder reads the query and a chunk TOGETHER, so it sees
# interaction cosine cannot, which is why no value of top_k removes that noise.
#
# Being on by default is affordable only because losing it is safe: the free
# tier allows 500 rerank requests a month and one /chat question spends one, so
# exhaustion is an EXPECTED operating state, not a failure. FailOpenReranker
# degrades to plain cosine order -- exactly the behaviour shipped before this
# -- and latches off rather than retrying into a wall.
#
# Set RERANK_PROVIDER to one of DISABLED_VALUES below to turn it off without a
# rebuild. That escape hatch is load-bearing now that the default is on.
DEFAULT_RERANK_PROVIDER: str | None = "pinecone"

# Env values that mean "no reranker". Needed because the usual
# ``os.environ.get(X) or DEFAULT`` idiom cannot express "off" once DEFAULT is
# truthy -- an empty string falls straight back to the default, so without
# these the feature could only be disabled by a code change.
RERANK_DISABLED_VALUES = frozenset({"none", "off", "false", "0", "disabled"})

# Rerank score below this is dropped. 0.0 means "keep everything", which is the
# default ON PURPOSE: a cross-encoder score is a different scale from cosine
# and the distribution on this corpus has not been measured yet. Picking a
# number first is exactly how MIN_RETRIEVAL_SCORE above ended up inert.
#
# This floor is applied ONLY to chunks that actually carry a rerank score. A
# degraded reranker returns them unscored, and filtering those would empty the
# context and refuse every question. See RAGPipeline.answer.
DEFAULT_RERANK_MIN_SCORE = 0.0

# Chunks kept after reranking; None keeps all. This is the knob that makes a
# reranker worth having -- reranking without truncating only renumbers the
# citations. Over-fetch by raising TOP_K and cut back here.
DEFAULT_RERANK_TOP_N: int | None = None

# How long the fail-open breaker stays latched after the provider's allowance
# runs out. The free Pinecone tier resets monthly while a Cloud Run revision
# can outlive that, so the latch re-arms hourly and retries rather than
# degrading until the next deploy. 0 latches until the process restarts.
DEFAULT_RERANK_RETRY_AFTER_SECONDS = 3600.0

# --- GCS -------------------------------------------------------------------


def gcs_bucket() -> str | None:
    """The configured GCS bucket name (env KMP_GCS_BUCKET), or None."""
    return os.environ.get("KMP_GCS_BUCKET")

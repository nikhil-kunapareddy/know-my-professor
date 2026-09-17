"""The judge: the model that scores, and the embeddings it scores with.

The only module here that needs credentials, and the only one that imports Ragas
or a provider SDK at module scope — import it and you are committing to the
``eval`` extra being installed. The CLI imports it inside ``main`` for exactly
that reason, so ``--help`` works without it.

Two deliberate choices:

**The judge is not built through ``core.llm``.** Every judged metric asks for a
*structured* verdict (a list of extracted claims, a verdict per claim), which
Ragas gets through Instructor. ``Generator.generate`` returns prose and cannot
express that, so wrapping it would mean re-implementing Instructor's
schema-coercion inside this package. The judge therefore talks to the SDK
directly, while still reading the same ``ANTHROPIC_API_KEY`` the service does.

**Judge embeddings are the corpus embeddings.** The metrics that need vectors
(answer_relevancy, semantic_similarity, answer_correctness) go through
``shared.embeddings``, so similarity is measured in the same space the index was
built in. A second embedding model here would make those scores incomparable to
retrieval's, for no gain.

Three findings from wiring this up, each of which is a silent or confusing
failure if forgotten — hence the code below rather than a note in a README:

1. Metrics score through ``ascore``, which calls ``agenerate``. Instructor
   raises ``TypeError: Cannot use agenerate() with a synchronous client``, so the
   client must be ``AsyncAnthropic``.
2. Ragas' ``llm_factory`` defaults to ``temperature=0.01, top_p=0.1``, and the
   Anthropic SDK's ``Messages.create()`` no longer accepts sampling parameters
   at all — every judged call fails with ``unexpected keyword argument
   'temperature'``. They are stripped after construction.
3. A judged case is dozens of judge calls, so the default model is a cheap one
   and the disk cache is on by default. Re-running an unchanged evaluation
   should not cost anything.
"""

from __future__ import annotations

import asyncio
import typing as t

import anthropic
from ragas.cache import DiskCacheBackend
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import llm_factory

from shared.config import EMBED_RATE_LIMIT_SLEEP_SECONDS
from shared.embeddings import build_embedder
from shared.embeddings.base import Embedder
from shared.settings import require_env

#: Judging is a classification job over short text, not a reasoning job, and it
#: runs dozens of times per case. Override with --judge-model when comparing a
#: stronger judge; expect the absolute numbers to shift when you do, so do not
#: compare across judges.
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"

#: Only Anthropic is wired up: Ragas reaches structured output through
#: Instructor, which has no adapter for the native Llama API client that
#: ``core.llm.llama`` uses. Naming the provider anyway keeps the CLI honest
#: about what it does and does not support.
JUDGE_PROVIDERS = ("anthropic",)
DEFAULT_JUDGE_PROVIDER = "anthropic"

#: Enough for a claim-decomposition verdict; Ragas' own default (1024) truncates
#: the longer ones, which surfaces as an unparseable response rather than an error.
JUDGE_MAX_TOKENS = 4096

#: Sampling parameters the Anthropic SDK rejects outright (see module docstring).
_UNSUPPORTED_MODEL_ARGS = ("temperature", "top_p")


def build_judge(
    model: str | None = None,
    provider: str = DEFAULT_JUDGE_PROVIDER,
    cache_dir: str | None = None,
) -> t.Any:
    """Build the Ragas judge LLM (an ``InstructorBaseRagasLLM``).

    ``cache_dir`` turns on Ragas' disk cache, keyed on the prompt — so a changed
    pipeline still re-judges, but a re-run of the same evaluation does not.
    """
    if provider not in JUDGE_PROVIDERS:
        raise ValueError(
            f"unknown judge provider {provider!r}; available: {JUDGE_PROVIDERS}. "
            "Ragas needs structured output, which it gets through Instructor; "
            "there is no Instructor adapter for the native Llama API client."
        )

    client = anthropic.AsyncAnthropic(api_key=require_env("ANTHROPIC_API_KEY"))
    judge = llm_factory(
        model or DEFAULT_JUDGE_MODEL,
        provider=provider,
        client=client,
        cache=DiskCacheBackend(cache_dir=cache_dir) if cache_dir else None,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    return drop_unsupported_model_args(judge)


def drop_unsupported_model_args(judge: t.Any) -> t.Any:
    """Remove the sampling parameters the Anthropic Messages API rejects.

    Separate from ``build_judge`` so a test can prove it happened without
    needing a key: if a Ragas upgrade renames ``model_args``, that test fails
    here instead of every judged call failing at runtime.
    """
    model_args = getattr(judge, "model_args", None)
    if not isinstance(model_args, dict):
        raise TypeError(
            f"expected the Ragas judge to expose a model_args dict, got "
            f"{type(model_args).__name__}. Ragas' InstructorLLM contract changed; "
            "check whether sampling parameters still need stripping."
        )
    for name in _UNSUPPORTED_MODEL_ARGS:
        model_args.pop(name, None)
    return judge


def build_judge_embeddings(embedder: Embedder | None = None) -> t.Any:
    """Wrap the corpus embedder in the interface Ragas' metrics accept.

    Paced on purpose. Mistral's free tier reports
    ``x-ratelimit-limit-req-minute: 60``, and a measured judged case costs 6
    embedding requests (answer_relevancy 2, answer_correctness 2,
    semantic_similarity 2) — so a 60-case run would ask for ~360 of them plus
    the 60 query embeds retrieval already spent. Unpaced, that is several times
    the allowance and the run degrades into 429 backoff.

    ``pace_seconds`` is the knob ingest already uses for exactly this reason
    (see ``shared/embeddings/mistral.py``); one second per request is the free
    tier's own ceiling. Serving still passes 0 — this tax is the evaluation's
    alone.
    """
    return EmbedderAdapter(
        embedder or build_embedder(pace_seconds=EMBED_RATE_LIMIT_SLEEP_SECONDS)
    )


class EmbedderAdapter(BaseRagasEmbedding):
    """A ``shared.embeddings.Embedder`` as a Ragas embedding.

    Ragas' metrics ``isinstance``-check their embeddings against
    ``BaseRagasEmbedding``, so the adapter inherits from it rather than merely
    matching its shape.

    Ragas' default batch behaviour embeds one text per request. The provider
    batches, so the batch methods are overridden to use it — a metric scoring
    ten strings costs one request, not ten. The sync API is bridged to async with
    ``to_thread`` rather than duplicated.

    The lock is not decoration. Several metrics embed at once while the runner
    scores concurrently, and driving one ``mistralai.Mistral`` client from
    several threads intermittently fails with ``AttributeError: 'NoneType'
    object has no attribute 'build_request'`` — the same calls made one at a time
    always reach the API. Embedding is batched and fast, so serialising it costs
    almost nothing and leaves the judge calls (the actual bottleneck) parallel.
    """

    def __init__(self, embedder: Embedder):
        super().__init__()
        self.embedder = embedder
        self._lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self.embedder.model

    def embed_text(self, text: str, **kwargs: t.Any) -> list[float]:
        return self.embedder.embed_query(text)

    async def aembed_text(self, text: str, **kwargs: t.Any) -> list[float]:
        async with self._lock:
            return await asyncio.to_thread(self.embedder.embed_query, text)

    def embed_texts(self, texts: list[str], **kwargs: t.Any) -> list[list[float]]:
        return self.embedder.embed_texts(list(texts))

    async def aembed_texts(self, texts: list[str], **kwargs: t.Any) -> list[list[float]]:
        async with self._lock:
            return await asyncio.to_thread(self.embedder.embed_texts, list(texts))

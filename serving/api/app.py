"""
/chat API for Know My Professor — a thin FastAPI wrapper around core.RAGPipeline.

POST /v1/chat  { "question": "..." }   (POST /chat is a deprecated alias)
GET  /health                           liveness
GET  /ready                            readiness: can we actually reach the index?

Providers are resolved through the registries, so this module names neither
Mistral nor Llama; swapping either is a settings change.

Required env:
  PINECONE_API_KEY, plus the selected providers' keys
  (MISTRAL_API_KEY for embedding, LLAMA_API_KEY for generation by default)
Optional env:
  PINECONE_INDEX_NAME, PINECONE_NAMESPACES, EMBED_PROVIDER, CHAT_PROVIDER,
  LLAMA_CHAT_MODEL, TOP_K, MIN_RETRIEVAL_SCORE, REQUEST_BUDGET_SECONDS,
  RERANK_MODEL, RERANK_MIN_SCORE, RERANK_TOP_N, RERANK_RETRY_AFTER_SECONDS
Reranking is ON by default (RERANK_PROVIDER=pinecone). Set RERANK_PROVIDER to
"none"/"off"/"false"/"0"/"disabled" to turn it off without a rebuild; it reuses
PINECONE_API_KEY, so it needs no new secret.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pinecone import Pinecone

from core.llm import build_generator
from core.pipeline import RAGPipeline
from core.rerank import FailOpenReranker, build_reranker
from core.retrieval.pinecone_retriever import PineconeRetriever
from shared.embeddings import build_embedder
from shared.schemas import API_VERSION, ChatRequest, ChatResponse, ErrorResponse
from shared.settings import ApiSettings, require_env

from .citations import build_citations
from .errors import classify_upstream_error
from .observability import configure_logging, log, new_request_id, request_id_var

#: Wall-clock budget for one /chat request. Cloud Run's own timeout is minutes;
#: this returns a clean 504 long before a browser gives up.
REQUEST_BUDGET_SECONDS = float(os.environ.get("REQUEST_BUDGET_SECONDS", "45"))

#: How long a successful readiness probe is trusted before re-checking upstream.
READINESS_TTL_SECONDS = 30.0

logger = logging.getLogger("kmp.api")
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build the pipeline once, failing fast on missing config."""
    global logger
    logger = configure_logging()

    settings = ApiSettings.from_env()
    embedder = build_embedder(settings.embed_provider)
    generator = build_generator(settings.chat_provider, model=settings.chat_model)

    # Optional second stage: absent unless a deployment names a provider, and
    # wrapped so that losing it degrades the answer instead of the service.
    reranker = None
    if settings.rerank_provider:
        reranker = FailOpenReranker(
            build_reranker(settings.rerank_provider, model=settings.rerank_model),
            retry_after_seconds=settings.rerank_retry_after_seconds,
        )

    # Providers read their keys lazily, which would turn a missing key into a
    # 502 on the first user request. Check now so the revision fails to boot.
    # The reranker is filtered out when absent -- None has no api_key_env.
    for provider in (p for p in (embedder, generator, reranker) if p is not None):
        if provider.api_key_env:
            require_env(provider.api_key_env)

    index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.index_name)

    state["settings"] = settings
    state["index"] = index
    state["ready_until"] = 0.0
    # One query reads exactly ONE namespace, so serving both corpora means one
    # query each -- and each gets its own top_k, because a course description
    # outscores the bio of the person who researches that topic. See
    # RAGPipeline.answer.
    state["pipeline"] = RAGPipeline(
        embedder=embedder,
        retriever=PineconeRetriever(index),
        generator=generator,
        top_k=settings.top_k,
        min_score=settings.min_score,
        namespaces=settings.namespaces,
        reranker=reranker,
        rerank_min_score=settings.rerank_min_score,
        rerank_top_n=settings.rerank_top_n,
    )

    log(
        logger, logging.INFO, "api ready",
        embed_provider=settings.embed_provider,
        embed_model=embedder.model,
        chat_provider=settings.chat_provider,
        chat_model=generator.model,
        index=settings.index_name,
        namespaces=list(settings.namespaces),
        top_k=settings.top_k,
        min_score=settings.min_score,
        rerank_provider=settings.rerank_provider,
        rerank_model=reranker.model if reranker else None,
        rerank_min_score=settings.rerank_min_score if reranker else None,
        rerank_top_n=settings.rerank_top_n if reranker else None,
    )
    yield


app = FastAPI(title="Know My Professor — /chat", lifespan=lifespan)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Tag every log line from this request, and echo the id to the client."""
    request_id = request.headers.get("X-Request-ID") or new_request_id()
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness only: the process is up. Deliberately does no upstream I/O."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    """Readiness: the pipeline is built AND the vector index answers.

    Result is cached briefly so a probe loop doesn't turn into an API-call bill.
    """
    if "pipeline" not in state:
        return JSONResponse(status_code=503, content={"status": "starting"})

    now = asyncio.get_event_loop().time()
    if now < state.get("ready_until", 0.0):
        return JSONResponse(status_code=200, content={"status": "ready", "cached": True})

    try:
        await run_in_threadpool(state["index"].describe_index_stats)
    except Exception as e:
        log(logger, logging.ERROR, "readiness check failed", error=type(e).__name__)
        return JSONResponse(status_code=503, content={"status": "index_unreachable"})

    state["ready_until"] = now + READINESS_TTL_SECONDS
    return JSONResponse(status_code=200, content={"status": "ready", "cached": False})


router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def chat(req: ChatRequest) -> Any:
    """Answer a question from the indexed corpus, with citations."""
    pipeline: RAGPipeline = state["pipeline"]
    started = time.perf_counter()

    try:
        result = await asyncio.wait_for(
            run_in_threadpool(pipeline.answer, req.question),
            timeout=REQUEST_BUDGET_SECONDS,
        )
    except TimeoutError:
        # The worker thread cannot be cancelled; it finishes and is discarded.
        # The client gets a clean 504 instead of hanging.
        log(logger, logging.ERROR, "chat timed out", budget_s=REQUEST_BUDGET_SECONDS)
        return _error_response(504, "upstream_timeout", "The answer took too long to generate. Try again.")
    except Exception as e:
        classified = classify_upstream_error(e)
        log(
            logger, logging.ERROR, "chat failed",
            error=type(e).__name__, code=classified.code, status=classified.status_code,
        )
        logger.exception("chat pipeline exception")  # full trace to logs, never to the client
        return _error_response(classified.status_code, classified.code, classified.detail)

    citations = build_citations(result.answer, result.sources)
    if result.sources and not citations:
        log(logger, logging.WARNING, "answer cited no sources", retrieved=result.retrieved)

    log(
        logger, logging.INFO, "chat ok",
        retrieved=result.retrieved,
        kept=len(result.sources),
        cited=len(citations),
        # False here while a reranker IS configured means it degraded -- out of
        # quota or unreachable. The one-off WARNING says why; this field is how
        # you tell which answers were affected.
        reranked=result.reranked,
        answer_chars=len(result.answer),
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
        **{f"stage_{k}_ms": v for k, v in result.timings_ms.items()},
    )
    return ChatResponse(answer=result.answer, citations=citations)


def _error_response(status_code: int, code: str, detail: str) -> JSONResponse:
    """An ErrorResponse carrying the request id, and no upstream text."""
    body = ErrorResponse(error=code, detail=detail, request_id=request_id_var.get())
    return JSONResponse(status_code=status_code, content=body.model_dump())


app.include_router(router, prefix=f"/{API_VERSION}")
# Unversioned alias: the frontend deploys separately from the API, so the old
# path has to keep working across the window where only one of them has shipped.
app.include_router(router, include_in_schema=False)

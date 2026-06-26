"""
/chat API for Know My Professor — a thin FastAPI wrapper around core.RAGPipeline.

POST /chat   { "question": "..." }
GET  /health

Required env:
  LLAMA_API_KEY      (chat model — Meta Llama API)
  MISTRAL_API_KEY    (query embedding — must match the ingest embedder)
  PINECONE_API_KEY
Optional env:
  PINECONE_INDEX_NAME  (default: know-my-professor-m1024)
  LLAMA_CHAT_MODEL     (default from shared.config)
  TOP_K                (default from shared.config)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from llama_api_client import LlamaAPIClient
from mistralai import Mistral
from pinecone import Pinecone
from pydantic import BaseModel, Field

from core.llm.generator import AnswerGenerator
from core.pipeline import RAGPipeline
from core.query.embedder import QueryEmbedder
from core.retrieval.pinecone_retriever import PineconeRetriever
from shared.config import DEFAULT_CHAT_MODEL, DEFAULT_TOP_K, PINECONE_DEFAULT_INDEX


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


class Citation(BaseModel):
    number: int
    professor_name: str
    professor_title: str
    section_type: str
    url: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


state: dict[str, Any] = {}


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required env var: {name}")
    return value


@asynccontextmanager
async def lifespan(_app: FastAPI):
    mistral = Mistral(api_key=_require_env("MISTRAL_API_KEY"))
    llama = LlamaAPIClient(api_key=_require_env("LLAMA_API_KEY"))
    pc = Pinecone(api_key=_require_env("PINECONE_API_KEY"))

    index_name = os.environ.get("PINECONE_INDEX_NAME", PINECONE_DEFAULT_INDEX)
    chat_model = os.environ.get("LLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    top_k = int(os.environ.get("TOP_K", DEFAULT_TOP_K))

    state["pipeline"] = RAGPipeline(
        embedder=QueryEmbedder(mistral),
        retriever=PineconeRetriever(pc.Index(index_name)),
        generator=AnswerGenerator(llama, model=chat_model),
        top_k=top_k,
    )
    print(f"Ready. chat_model={chat_model} index={index_name} top_k={top_k}")
    yield


app = FastAPI(title="Know My Professor — /chat", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        result = state["pipeline"].answer(req.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"chat pipeline failed: {e}") from e

    citations = [
        Citation(
            number=i,
            professor_name=src.metadata.get("professor_name", ""),
            professor_title=src.metadata.get("professor_title", ""),
            section_type=src.metadata.get("section_type", ""),
            url=src.metadata.get("url", ""),
            score=src.score,
        )
        for i, src in enumerate(result.sources, start=1)
    ]
    return ChatResponse(answer=result.answer, citations=citations)

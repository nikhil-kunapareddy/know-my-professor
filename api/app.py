"""
/chat API for Know My Professor.

POST /chat   { "question": "..." }
GET  /health

Required env:
  GEMINI_API_KEY
  PINECONE_API_KEY
Optional env:
  PINECONE_INDEX_NAME  (default: know-my-professor)
  GEMINI_CHAT_MODEL    (default: models/gemini-2.5-flash)
  TOP_K                (default: 8)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pinecone import Pinecone
from pydantic import BaseModel, Field

EMBED_MODEL = "models/gemini-embedding-001"
DEFAULT_CHAT_MODEL = "models/gemini-2.5-flash"
DEFAULT_INDEX = "know-my-professor"
DEFAULT_TOP_K = 8

SYSTEM_INSTRUCTION = """\
You answer questions about faculty at Northeastern University's Khoury College of Computer Sciences.

Rules:
- Use ONLY the numbered context entries below to answer. Do not invent facts.
- Cite the professors you used with their bracketed numbers, e.g. [1], [3].
- If multiple professors are relevant, list them.
- If the context does not contain the answer, say "I don't have that information in my data."
- Be concise. Two or three sentences is usually enough.
"""


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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    gemini_key = _require_env("GEMINI_API_KEY")
    pinecone_key = _require_env("PINECONE_API_KEY")

    genai.configure(api_key=gemini_key)
    chat_model_name = os.environ.get("GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    state["chat_model"] = genai.GenerativeModel(
        chat_model_name,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    pc = Pinecone(api_key=pinecone_key)
    index_name = os.environ.get("PINECONE_INDEX_NAME", DEFAULT_INDEX)
    state["index"] = pc.Index(index_name)
    state["top_k"] = int(os.environ.get("TOP_K", DEFAULT_TOP_K))

    print(f"Ready. chat_model={chat_model_name} index={index_name} top_k={state['top_k']}")
    yield


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required env var: {name}")
    return value


app = FastAPI(title="Know My Professor — /chat", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        query_embedding = genai.embed_content(
            model=EMBED_MODEL,
            content=req.question,
            task_type="retrieval_query",
        )["embedding"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"embedding failed: {e}") from e

    try:
        results = state["index"].query(
            vector=query_embedding,
            top_k=state["top_k"],
            include_metadata=True,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vector search failed: {e}") from e

    matches = results.matches or []
    if not matches:
        return ChatResponse(
            answer="I don't have that information in my data.", citations=[]
        )

    context_blocks: list[str] = []
    citations: list[Citation] = []
    for i, match in enumerate(matches, start=1):
        md = match.metadata or {}
        context_blocks.append(
            f"[{i}] {md.get('professor_name', '')} "
            f"({md.get('professor_title', '')}) — {md.get('section_type', '')}\n"
            f"{md.get('text', '')}"
        )
        citations.append(
            Citation(
                number=i,
                professor_name=md.get("professor_name", ""),
                professor_title=md.get("professor_title", ""),
                section_type=md.get("section_type", ""),
                url=md.get("url", ""),
                score=float(match.score),
            )
        )

    prompt = (
        "Context:\n"
        + "\n\n".join(context_blocks)
        + f"\n\nQuestion: {req.question}\n"
    )

    try:
        response = state["chat_model"].generate_content(prompt)
        answer = (response.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"chat model failed: {e}") from e

    if not answer:
        answer = "I don't have that information in my data."

    return ChatResponse(answer=answer, citations=citations)

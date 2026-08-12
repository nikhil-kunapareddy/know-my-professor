"""The /chat wire contract, imported by BOTH ends.

The API validates responses against these models and the frontend parses into
them, so the contract is checked on both sides instead of existing twice — once
as FastAPI models and once as ``data.get("answer")`` in the UI. A field rename
now breaks at import/parse time rather than rendering as a blank in the browser.

Kept in ``shared`` (not ``serving/api``) so the frontend can depend on it without
pulling in FastAPI.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

#: Bumped when the response shape changes incompatibly; surfaced at /v1/chat.
API_VERSION = "v1"


class ChatRequest(BaseModel):
    """A question for the RAG pipeline."""

    question: str = Field(min_length=2, max_length=2000)


class Citation(BaseModel):
    """One source the answer was grounded in.

    ``number`` matches the bracketed marker in the answer text, so ``[2]`` in the
    prose refers to the citation with ``number == 2``.
    """

    number: int
    professor_name: str = ""
    professor_title: str = ""
    section_type: str = ""
    url: str = ""
    score: float = 0.0


class ChatResponse(BaseModel):
    """An answer plus the sources it actually cited."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """A failure the caller can act on.

    ``error`` is a short stable code for the client to branch on; ``request_id``
    is echoed so a user-reported failure can be found in the logs. Upstream
    provider text is deliberately absent — it can carry internal detail.
    """

    error: str
    detail: str
    request_id: str

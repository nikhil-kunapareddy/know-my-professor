"""Semantic similarity between the answer and the reference — no judge involved.

DeepEval ships no equivalent, and this is the one metric in the table worth
writing by hand rather than dropping: it is the only reading in the whole report
that does not come from an LLM's opinion. When every judged metric moves at once
after a prompt change, this one says whether the answers actually changed or
only the judge's mood did.

It is also free of judge cost and judge variance — two embeddings and a dot
product — so it runs on every case at no quota expense beyond the embedder's.

**The embedding space is the corpus's own.** Similarity is measured with
``shared.embeddings``, the same model the index was built with, so a score here
is commensurable with a retrieval score. A second embedding model would produce
numbers that look like the others and mean something else.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams

from shared.embeddings.base import Embedder


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity, clamped to [0, 1].

    Clamped because a negative cosine is not a meaningful "less similar than
    nothing" reading for a mean, and because every other metric in the report is
    on [0, 1] — one column that can go negative invites misreading the table.
    """
    if len(left) != len(right):
        raise ValueError(f"embedding widths differ: {len(left)} vs {len(right)}")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / norm))


class SemanticSimilarityMetric(BaseMetric):
    """Cosine similarity between ``actual_output`` and ``expected_output``.

    Implements DeepEval's ``BaseMetric`` so the runner, the report, and the
    ``--min`` gates treat it exactly like a judged metric — the fact that no LLM
    is involved is an implementation detail everywhere except the cost.

    The lock is not decoration, and it is deliberately shared by every instance
    rather than held per-instance. The runner builds a fresh metric for each
    (sample, metric) pair — DeepEval metrics keep their score and reason as
    instance state, so a shared instance would race — which means a per-instance
    lock would serialise nothing at all. Driving one ``mistralai.Mistral``
    client from several threads intermittently fails with ``AttributeError:
    'NoneType' object has no attribute 'build_request'``, while the same calls
    made one at a time always reach the API. Embedding is batched and fast, so
    serialising it costs almost nothing.
    """

    _required_params = [SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT]

    #: Shared across instances on purpose — see the class docstring. Safe to
    #: build at class-definition time: since Python 3.10 an asyncio.Lock binds
    #: to the running loop on first use, not at construction.
    _LOCK = asyncio.Lock()

    def __init__(self, embedder: Embedder, threshold: float = 0.5, **_: Any):
        self.embedder = embedder
        self.threshold = threshold
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None
        self.error: str | None = None
        self.skipped = False
        self.evaluation_model = getattr(embedder, "model", "embeddings")
        self.async_mode = False
        self.include_reason = True
        self.strict_mode = False
        self.verbose_mode = False

    @property
    def __name__(self) -> str:
        return "semantic_similarity"

    def _score(self, test_case: LLMTestCase) -> float:
        left, right = self.embedder.embed_texts(
            [test_case.actual_output or "", test_case.expected_output or ""]
        )
        value = cosine(left, right)
        self.score = value
        self.success = value >= self.threshold
        self.reason = f"cosine similarity in {self.evaluation_model} space"
        return value

    def measure(self, test_case: LLMTestCase, *_: Any, **__: Any) -> float:
        return self._score(test_case)

    async def a_measure(self, test_case: LLMTestCase, *_: Any, **__: Any) -> float:
        async with self._LOCK:
            return await asyncio.to_thread(self._score, test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

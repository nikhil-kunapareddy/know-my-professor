"""Running every arm over every case, and recording what each call actually did.

The output is a dump (``samples.jsonl``), not a score. Ranking reads the dump,
so a rubric change never re-pays for generation — which is the whole reason the
dump exists rather than scoring inline.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.pipeline import NO_ANSWER
from evaluation.live import LiveSystem
from shared.embeddings.base import Embedder

from .arms import ARMS, Arm
from .cases import Case


class CachingEmbedder(Embedder):
    """Memoises ``embed_query`` and serialises calls to the wrapped client.

    Two reasons, both load-bearing:

    - **The same 50 questions are embedded once per arm per repeat.** Without a
      cache that is 400 identical embeddings against a free-tier key, for no
      information: the query text does not depend on which model answers it.
    - **One Mistral client driven from several threads intermittently raises**
      ``AttributeError: 'NoneType' object has no attribute 'build_request'``
      while the same calls made sequentially reach the API. CLAUDE.md records
      this, and it is why the lock is held across the client call rather than
      only around the dict.

    Wrapping the embedder rather than reaching past the pipeline keeps the
    measured path identical to the one ``serving/api`` runs.
    """

    def __init__(self, inner: Embedder):
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            return self._inner.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            hit = self._cache.get(text)
            if hit is None:
                hit = self._inner.embed_query(text)
                self._cache[text] = hit
            return hit


@dataclass
class Sample:
    """One (case, arm, repeat) call, with everything needed to score it later."""

    case_id: str
    arm: str
    repeat: int
    model: str
    effort: str | None
    answer: str
    declined: bool
    #: Provider accounting. ``output_tokens`` is the billed figure and includes
    #: thinking; ``answer_chars`` is the visible answer, so the two together show
    #: how much of a bill was spent on reasoning the user never sees.
    input_tokens: int = 0
    output_tokens: int = 0
    answer_chars: int = 0
    stop_reason: str | None = None
    refusal_category: str | None = None
    refused: bool = False
    truncated: bool = False
    cost_usd: float = 0.0
    latency_ms: dict[str, float] = field(default_factory=dict)
    retrieved: int = 0
    #: The chunks the model was shown, so ranking can check a named entity was
    #: actually retrievable rather than invented.
    chunks: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.case_id, self.arm, self.repeat)


def _declined(answer: str) -> bool:
    """Whether the answer is a refusal to answer.

    Matches the prompt's phrasing rather than the exact ``NO_ANSWER`` string:
    CLAUDE.md records that exact equality scored caveated refusals as failures,
    7/7 correct refusals reading as 0/7, because the model opens with the
    requested sentence and then honestly names the nearest material.
    """
    head = answer.strip().lower()[:160]
    if not head:
        return True
    return any(
        phrase in head
        for phrase in (
            NO_ANSWER.strip().lower()[:40],
            "i don't have that information",
            "i do not have that information",
            "not in my data",
            "no information about",
        )
    )


def existing(path: Path) -> dict[tuple[str, str, int], Sample]:
    """Samples already on disk, so a crashed run resumes instead of re-paying."""
    if not path.exists():
        return {}
    out: dict[tuple[str, str, int], Sample] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            s = Sample(**json.loads(line))
            out[s.key] = s
    return out


def run(
    system: LiveSystem,
    cases: list[Case],
    out_path: Path,
    top_k: int,
    min_score: float,
    repeats: int = 2,
    arms: dict[str, Arm] | None = None,
) -> list[Sample]:
    """Generate an answer per (case, arm, repeat), appending each to the dump.

    Retrieval runs inside the pipeline for every call rather than once per case.
    It is free (Pinecone + a cached embedding), it keeps the measured path
    identical to serving, and it lets the report *verify* that every arm saw the
    same chunks instead of assuming it.
    """
    arms = arms or ARMS
    system = LiveSystem(
        settings=system.settings,
        embedder=CachingEmbedder(system.embedder),
        retriever=system.retriever,
    )
    done = existing(out_path)
    results = list(done.values())

    with out_path.open("a") as sink:
        for arm_name, arm in arms.items():
            pipeline = system.pipeline(
                top_k=top_k,
                min_score=min_score,
                chat_provider="anthropic",
                chat_model=arm.model,
                effort=arm.effort,
            )
            for case in cases:
                for repeat in range(1, repeats + 1):
                    if (case.id, arm_name, repeat) in done:
                        continue
                    sample = _one(pipeline, case, arm_name, arm, repeat)
                    sink.write(json.dumps(asdict(sample)) + "\n")
                    sink.flush()
                    results.append(sample)
    return results


def _one(pipeline, case: Case, arm_name: str, arm: Arm, repeat: int) -> Sample:
    started = time.perf_counter()
    try:
        # No namespace argument: this is the blended path /chat actually serves,
        # which no existing harness measures.
        result = pipeline.answer(case.question)
    except Exception as exc:  # noqa: BLE001 - one bad call must not void the run
        return Sample(
            case_id=case.id, arm=arm_name, repeat=repeat, model=arm.model,
            effort=arm.effort, answer="", declined=False,
            latency_ms={"total": round((time.perf_counter() - started) * 1000, 1)},
            error=f"{type(exc).__name__}: {exc}",
        )

    gen = result.generation
    return Sample(
        case_id=case.id,
        arm=arm_name,
        repeat=repeat,
        model=arm.model,
        effort=arm.effort,
        answer=result.answer,
        declined=_declined(result.answer),
        input_tokens=gen.input_tokens if gen else 0,
        output_tokens=gen.output_tokens if gen else 0,
        answer_chars=len(result.answer),
        stop_reason=gen.stop_reason if gen else None,
        refusal_category=gen.refusal_category if gen else None,
        refused=bool(gen and gen.refused),
        truncated=bool(gen and gen.truncated),
        cost_usd=arm.cost(gen.input_tokens, gen.output_tokens) if gen else 0.0,
        latency_ms=dict(result.timings_ms),
        retrieved=result.retrieved,
        chunks=[
            {"document_id": s.document_id, "score": round(s.score, 4),
             "rank": i, "text": s.metadata.get("text", "")}
            for i, s in enumerate(result.sources, start=1)
        ],
    )

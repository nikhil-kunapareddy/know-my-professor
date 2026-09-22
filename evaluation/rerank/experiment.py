"""Retrieve once, rerank once, judge once, score both orderings.

The design rests on one property: reranking REORDERS a set, it does not change
it. So a chunk's verdict is a property of (question, chunk), not of the arm it
appears in — judge the set once and both arms are scored for free. That is also
why ``evaluation.topk``'s cache still applies: its key is
(rubric, model, effort, question, chunk_id, text), none of which reranking
touches.

Nothing here decides anything; ``report.py`` turns these records into tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.rerank.base import Reranker
from core.retrieval.base import RetrievalResult
from evaluation.harness import EvalCase
from evaluation.topk.experiment import JudgedCase, JudgedChunk

#: Question strata for THIS set. `evaluation.topk` has its own three-value
#: tuple and is deliberately left alone, so "blended" is added here instead of
#: there — the topk experiment is a recorded result, not a shared library of
#: constants.
STRATA = ("broad", "narrow", "noans", "blended")


def stratum_of(case: EvalCase) -> str:
    """Which stratum a case belongs to, read from ``{corpus}-{stratum}-{slug}``."""
    parts = case.id.split("-")
    return parts[1] if len(parts) > 1 and parts[1] in STRATA else ""


def corpus_of(case: EvalCase) -> str:
    """"people", "courses", or "blended" — what the case actually searches."""
    return case.namespace if case.namespace is not None else "blended"


@dataclass
class RerankedCase:
    """One question's chunks, held in both orderings.

    ``chunks`` is the cosine order and is also the canonical set: the two
    ordering lists hold the SAME ``JudgedChunk`` objects, so filling a verdict
    in one fills it in the other. Naming it ``chunks`` (with ``case`` alongside)
    is what lets ``topk.experiment.judge_cases`` consume this class unchanged.
    """

    case: EvalCase
    namespaces: tuple[str | None, ...]
    chunks: list[JudgedChunk] = field(default_factory=list)
    reranked: list[JudgedChunk] = field(default_factory=list)
    #: document_id -> cross-encoder score. Empty when the provider degraded.
    rerank_scores: dict[str, float] = field(default_factory=dict)
    #: True when the reranker could not rank — the arm is then a copy of the
    #: baseline and must not be read as "reranking made no difference".
    degraded: bool = False

    def before(self) -> JudgedCase:
        """The baseline arm, shaped for ``topk.experiment.metrics_at``."""
        return JudgedCase(case=self.case, namespace=corpus_of(self.case), chunks=self.chunks)

    def after(self) -> JudgedCase:
        return JudgedCase(case=self.case, namespace=corpus_of(self.case), chunks=self.reranked)

    def to_dict(self) -> dict:
        return {
            "id": self.case.id,
            "question": self.case.question,
            "namespaces": list(self.namespaces),
            "expect_no_answer": self.case.expect_no_answer,
            "expected_slugs": list(self.case.expected_slugs),
            "degraded": self.degraded,
            "rerank_scores": self.rerank_scores,
            # Written once, in cosine order; the reranked arm is rebuilt from
            # rerank_scores on load, so the file cannot hold two orderings that
            # disagree about what was retrieved.
            "chunks": [c.to_dict() for c in self.chunks],
        }


def targets_for(case: EvalCase, namespaces: tuple[str, ...]) -> tuple[str | None, ...]:
    """Which namespaces this case searches.

    An explicit ``None`` namespace means the blended path — every namespace,
    one query each, exactly as ``RAGPipeline.answer`` fans out.
    """
    return (case.namespace,) if case.namespace is not None else tuple(namespaces)


def retrieve_cases(live, cases: list[EvalCase], k: int) -> list[RerankedCase]:
    """Embed and search each case, ``k`` per namespace, union sorted by cosine.

    Goes through ``live.retriever`` rather than ``live.retrieve`` on purpose:
    the latter applies a reranker when one is configured, which would rerank
    the baseline arm and collapse the experiment into comparing a thing with
    itself. The arms are built explicitly here instead.

    No score floor. It is measured to be inert on this corpus, and applying it
    would conflate "the floor removed it" with "retrieval never found it".
    """
    out: list[RerankedCase] = []
    for case in cases:
        targets = targets_for(case, live.settings.namespaces)
        vector = live.embedder.embed_query(case.question)

        results: list[RetrievalResult] = []
        for namespace in targets:
            results.extend(live.retriever.retrieve(vector, k, namespace=namespace))
        results.sort(key=lambda r: r.score, reverse=True)

        out.append(
            RerankedCase(
                case=case,
                namespaces=targets,
                chunks=[
                    JudgedChunk(
                        document_id=r.document_id,
                        score=r.score,
                        rank=rank,
                        text=str(r.metadata.get("text", "")),
                    )
                    for rank, r in enumerate(results, start=1)
                ],
            )
        )
        label = "+".join(str(t) for t in targets)
        print(f"  retrieved {len(out[-1].chunks):>2} from {label:16} for [{case.id}]")
    return out


def rerank_cases(reranker: Reranker, cases: list[RerankedCase]) -> int:
    """Fill each case's reranked ordering. Returns how many degraded.

    One rerank request per case, which is the unit the free tier bills.
    """
    degraded = 0
    for case in cases:
        if not case.chunks:
            continue
        results = [
            RetrievalResult(
                document_id=c.document_id, score=c.score, metadata={"text": c.text}
            )
            for c in case.chunks
        ]
        try:
            ranked = reranker.rerank(case.case.question, results)
        except Exception as e:  # noqa: BLE001 - recorded per case, never fatal
            print(f"    ERROR [{case.case.id}]: {type(e).__name__}: {e}")
            ranked = []

        scored = {r.document_id: r.rerank_score for r in ranked if r.rerank_score is not None}
        if not scored:
            # Either the provider failed or FailOpenReranker handed the list
            # back unranked. Either way the arm is a copy, and saying so is
            # what stops it being read as "the reranker changed nothing".
            case.degraded = True
            case.reranked = list(case.chunks)
            degraded += 1
            continue

        case.rerank_scores = scored
        by_id = {c.document_id: c for c in case.chunks}
        case.reranked = [by_id[r.document_id] for r in ranked if r.document_id in by_id]
        print(
            f"  reranked {len(case.reranked):>2} for [{case.case.id}]"
            f"  top={max(scored.values()):.3f} bottom={min(scored.values()):.3f}"
        )
    return degraded


def score_pairs(cases: list[RerankedCase]) -> list[tuple[float, bool]]:
    """Every (rerank score, judged relevant) pair, for choosing the cutoff.

    Unjudged chunks and degraded cases are excluded rather than defaulted:
    a missing verdict counted as irrelevant would drag the "irrelevant"
    distribution toward the relevant one and make any cutoff look better than
    it is.
    """
    pairs: list[tuple[float, bool]] = []
    for case in cases:
        if case.degraded:
            continue
        for chunk in case.chunks:
            score = case.rerank_scores.get(chunk.document_id)
            if score is not None and chunk.relevant is not None:
                pairs.append((score, chunk.relevant))
    return pairs


def write_cases(path: Path, cases: list[RerankedCase], run: dict) -> None:
    """Record a run so the report can be rebuilt without re-buying anything."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"_run": run})]
    lines += [json.dumps(c.to_dict()) for c in cases]
    path.write_text("\n".join(lines) + "\n")


def read_cases(path: Path, by_id: dict[str, EvalCase]) -> tuple[list[RerankedCase], dict]:
    """Replay a recorded run, rejoining it to the live question set by case id."""
    run: dict = {}
    cases: list[RerankedCase] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if "_run" in raw:
            run = raw["_run"]
            continue
        case = by_id.get(raw["id"])
        if case is None:
            print(f"  skipping [{raw['id']}]: no longer in the question set")
            continue

        chunks = [JudgedChunk.from_dict(c) for c in raw["chunks"]]
        scores = raw.get("rerank_scores", {})
        by_doc = {c.document_id: c for c in chunks}
        reranked = (
            list(chunks)
            if raw.get("degraded")
            else [
                by_doc[d]
                for d in sorted(scores, key=lambda d: scores[d], reverse=True)
                if d in by_doc
            ]
        )
        cases.append(
            RerankedCase(
                case=case,
                namespaces=tuple(raw.get("namespaces", ())),
                chunks=chunks,
                reranked=reranked,
                rerank_scores=scores,
                degraded=bool(raw.get("degraded")),
            )
        )
    return cases, run

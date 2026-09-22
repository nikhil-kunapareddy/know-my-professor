"""Retrieve once at the largest k, judge once, derive every smaller k.

The whole experiment rests on one property: Pinecone returns a *ranked* list for
a deterministic query vector, so the chunks a top-3 query returns are the first
three a top-11 query returns. Five k values therefore need one retrieval and one
pass of judging, not five of each — 550 judge calls instead of 1,750.

Nothing here decides anything. It produces ``JudgedCase`` records and the
per-k arithmetic over them; ``report.py`` applies the rule, and the raw records
are written to disk so a rule change never costs another judge call.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from evaluation.harness import CaseOutcome, EvalCase, EvalReport

from .judge import RelevanceJudge, Verdict

#: Fable 5.1, USD per million tokens. Used only to report what a run actually
#: cost; nothing branches on it.
INPUT_USD_PER_MTOK = 10.0
OUTPUT_USD_PER_MTOK = 50.0


#: Question strata, read from the case id (``{namespace}-{stratum}-{slug}``).
#:
#: The distinction is not cosmetic, and the smoke run proved it: a narrow
#: question has exactly one correct entity, so its relevant-chunk count pins at
#: ~1 and every marginal gain is +0.00 no matter what k is. A set made only of
#: narrow questions answers "k=3" by construction and measures nothing. k is
#: decided by the broad questions — the ones with more valid answers than any
#: top-k can hold — so the two are reported apart as well as pooled.
STRATA = ("broad", "narrow", "noans")


def stratum_of(case: EvalCase) -> str:
    """Which stratum a case belongs to, or "" if its id does not say."""
    parts = case.id.split("-")
    return parts[1] if len(parts) > 1 and parts[1] in STRATA else ""


@dataclass
class JudgedChunk:
    """One retrieved chunk, its rank, and the judge's verdict on it."""

    document_id: str
    score: float
    rank: int
    text: str
    relevant: bool | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "score": self.score,
            "rank": self.rank,
            "text": self.text,
            "relevant": self.relevant,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> JudgedChunk:
        return cls(**raw)


@dataclass
class JudgedCase:
    """One question, its namespace, and its ranked chunks with verdicts."""

    case: EvalCase
    namespace: str
    chunks: list[JudgedChunk] = field(default_factory=list)

    def at(self, k: int) -> list[JudgedChunk]:
        """The chunks a top-k query would have returned — the prefix trick."""
        return self.chunks[:k]

    def to_dict(self) -> dict:
        return {
            "id": self.case.id,
            "question": self.case.question,
            "namespace": self.namespace,
            "expect_no_answer": self.case.expect_no_answer,
            "expected_slugs": list(self.case.expected_slugs),
            "chunks": [c.to_dict() for c in self.chunks],
        }


def retrieve_cases(live, cases: list[EvalCase], k_max: int) -> list[JudgedCase]:
    """Query each case against its OWN namespace, once, at ``k_max``.

    ``min_score=0``: the floor is measured to be inert on this corpus (nothing
    scores below ~0.70), and applying it here would silently conflate "the
    floor removed it" with "retrieval never found it".
    """
    out: list[JudgedCase] = []
    for case in cases:
        namespace = case.namespace
        results = live.retrieve(case.question, k_max, 0.0, namespace=namespace)
        out.append(
            JudgedCase(
                case=case,
                namespace=namespace,
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
        print(f"  retrieved {len(out[-1].chunks):>2} for [{case.id}]")
    return out


@dataclass
class JudgingStats:
    """What a judging pass cost, measured rather than estimated."""

    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def usd(self) -> float:
        return (
            self.input_tokens / 1e6 * INPUT_USD_PER_MTOK
            + self.output_tokens / 1e6 * OUTPUT_USD_PER_MTOK
        )

    def format(self) -> str:
        return (
            f"{self.calls} pair(s): {self.cache_hits} cached, {self.errors} error(s); "
            f"{self.input_tokens:,} in + {self.output_tokens:,} out = ${self.usd:.2f}"
        )


def judge_cases(
    judge: RelevanceJudge, cases: list[JudgedCase], concurrency: int = 8
) -> JudgingStats:
    """Fill in every chunk's verdict, in parallel, mutating ``cases`` in place.

    A pair that errors is left with ``relevant=None`` and counted, never
    defaulted to False: a silent False reads as a retrieval miss and would bias
    every k downward by the same amount, which is invisible in the output.
    """
    stats = JudgingStats()
    work = [(case, chunk) for case in cases for chunk in case.chunks]

    def run(item) -> tuple[JudgedChunk, Verdict | None]:
        case, chunk = item
        try:
            return chunk, judge.judge(case.case.question, chunk.document_id, chunk.text)
        except Exception as e:  # noqa: BLE001 - recorded per pair, never fatal
            print(f"    ERROR {chunk.document_id}: {type(e).__name__}: {e}")
            return chunk, None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for done, (chunk, verdict) in enumerate(pool.map(run, work), start=1):
            stats.calls += 1
            if verdict is None:
                stats.errors += 1
            else:
                chunk.relevant = verdict.relevant
                chunk.reason = verdict.reason
                stats.cache_hits += int(verdict.cached)
                stats.input_tokens += verdict.input_tokens
                stats.output_tokens += verdict.output_tokens
            if done % 25 == 0 or done == len(work):
                print(f"  judged {done}/{len(work)}")
    return stats


@dataclass(frozen=True)
class KMetrics:
    """Everything one value of k scored, over one namespace."""

    k: int
    cases: int
    #: Mean chunks retrieved. Below k only where the corpus held fewer.
    mean_retrieved: float
    #: THE headline: mean judged-relevant chunks per answerable question.
    mean_relevant: float
    #: Relevant / retrieved, over answerable questions — how much of what the
    #: model is handed earns its place.
    precision: float
    recall_at_k: float
    mrr: float
    #: Share of no-answer questions where the judge found NOTHING relevant.
    #: A larger k passing more plausible junk shows up here first.
    no_answer_clean: float | None
    #: Pairs the judge failed on, excluded from the means above.
    unjudged: int


def metrics_at(cases: list[JudgedCase], k: int) -> KMetrics:
    """Score every case at one k, using only the first k chunks of each."""
    positives = [c for c in cases if not c.case.expect_no_answer]
    negatives = [c for c in cases if c.case.expect_no_answer]

    judged = [ch for c in positives for ch in c.at(k) if ch.relevant is not None]
    unjudged = sum(1 for c in cases for ch in c.at(k) if ch.relevant is None)
    relevant_total = sum(1 for ch in judged if ch.relevant)

    per_case = [
        sum(1 for ch in c.at(k) if ch.relevant) for c in positives
    ]

    # Recall and MRR come from the existing slug-based harness rather than a
    # second implementation here, so the two families of metric cannot drift
    # apart on what counts as a hit.
    report = EvalReport(
        outcomes=[
            CaseOutcome(case=c.case, retrieved_ids=[ch.document_id for ch in c.at(k)])
            for c in cases
        ],
        top_k=k,
    )

    return KMetrics(
        k=k,
        cases=len(cases),
        mean_retrieved=(
            sum(len(c.at(k)) for c in positives) / len(positives) if positives else 0.0
        ),
        mean_relevant=(sum(per_case) / len(per_case) if per_case else 0.0),
        precision=(relevant_total / len(judged) if judged else 0.0),
        recall_at_k=report.recall_at_k,
        mrr=report.mrr,
        no_answer_clean=(
            sum(1 for c in negatives if not any(ch.relevant for ch in c.at(k))) / len(negatives)
            if negatives
            else None
        ),
        unjudged=unjudged,
    )


def write_cases(path: Path, cases: list[JudgedCase]) -> None:
    """Record the run so any later analysis is free of the judge."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def read_cases(path: Path, by_id: dict[str, EvalCase]) -> list[JudgedCase]:
    """Replay a recorded run, re-joining it to the question file by case id.

    Re-joining rather than trusting the dump's own copy means the expectations
    can be corrected after a run without re-judging a single pair.
    """
    out: list[JudgedCase] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        case = by_id.get(raw["id"])
        if case is None:
            print(f"  skipping [{raw['id']}]: no longer in the question set")
            continue
        out.append(
            JudgedCase(
                case=case,
                namespace=raw["namespace"],
                chunks=[JudgedChunk.from_dict(c) for c in raw["chunks"]],
            )
        )
    return out

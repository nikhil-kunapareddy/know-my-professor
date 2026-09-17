"""Ragas evaluation: LLM-judged quality at each point in the RAG pipeline.

``evaluation/harness.py`` answers the questions a slug list can answer — did the
right professor come back, and how high up. It cannot answer the ones that
decide whether the product is good: is the answer *supported* by what was
retrieved, does it actually address the question, is it *correct*. Those need a
judge, and that is what Ragas provides.

The framework is split by the point in the pipeline it probes, because a single
"quality" number tells you nothing about where to spend your time:

    retrieval   did the right context come back?        (no generator involved)
    generation  given that context, is the answer grounded and on-topic?
    end_to_end  is the final answer actually right?     (needs a reference)

Layout mirrors that split, and keeps Ragas at the edges:

    samples.py   pure: golden case + pipeline output -> one judged record
    metrics.py   the stage -> metric table, resolved lazily by name
    judge.py     the judge LLM and embeddings (the only module needing keys)
    runner.py    scores samples x metrics concurrently; records, never raises
    report.py    pure: aggregate, format, and gate on minimums

Only ``judge.py`` and ``metrics.build_metric`` import Ragas, so everything that
decides *what* gets scored is unit-testable with no SDK and no network.

Cost: a judged run is dozens of LLM calls per case. ``run_ragas.py`` defaults to
a cheap judge model, caches judge calls on disk, and can replay a recorded run
(``--from-dump``) so iterating on metrics costs nothing.
"""

from __future__ import annotations

from .samples import (
    RagasSample,
    context_texts,
    read_samples,
    sample_from_result,
    sample_from_retrieval,
    write_samples,
)

__all__ = [
    "RagasSample",
    "context_texts",
    "read_samples",
    "sample_from_result",
    "sample_from_retrieval",
    "write_samples",
]

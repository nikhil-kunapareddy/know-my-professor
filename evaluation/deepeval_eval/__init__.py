"""DeepEval evaluation: LLM-judged quality at each point in the RAG pipeline.

``evaluation/harness.py`` answers the questions a slug list can answer — did the
right professor come back, and how high up. It cannot answer the ones that
decide whether the product is good: is the answer *supported* by what was
retrieved, does it actually address the question, is it *correct*. Those need a
judge, and that is what DeepEval provides.

The framework is split by the point in the pipeline it probes, because a single
"quality" number tells you nothing about where to spend your time:

    retrieval   did the right context come back?        (no generator involved)
    generation  given that context, is the answer grounded and on-topic?
    end_to_end  is the final answer actually right?     (needs a reference)

Layout mirrors that split, and keeps DeepEval at the edges:

    samples.py     pure: golden case + pipeline output -> one judged record
    metrics.py     the stage -> metric table, resolved lazily by name
    judge.py       the judge model and its disk cache (the only module needing keys)
    similarity.py  the one metric DeepEval lacks, computed on the corpus embedder
    runner.py      scores samples x metrics concurrently; records, never raises
    report.py      pure: aggregate, format, and gate on minimums

Only ``judge.py``, ``similarity.py`` and ``metrics.build_metric`` import
DeepEval, so everything that decides *what* gets scored is unit-testable with no
SDK and no network.

Cost: a judged case is dozens of LLM calls. ``run_deepeval.py`` defaults to a
cheap judge model, caches judge calls on disk, and can replay a recorded run
(``--from-dump``) so iterating on metrics costs nothing.
"""

from __future__ import annotations

import os

# Set before anything imports deepeval, which reads it at import time. DeepEval
# ships PostHog telemetry that is on by default; this is a local evaluation tool
# for a private corpus, so it is turned off at the package boundary rather than
# left to whoever remembers to export it. setdefault, so an explicit "0" in the
# environment still wins.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "1")

from .samples import (  # noqa: E402 - must follow the telemetry opt-out above
    JudgedSample,
    SampleSet,
    context_texts,
    read_samples,
    sample_from_result,
    sample_from_retrieval,
    write_samples,
)

__all__ = [
    "JudgedSample",
    "SampleSet",
    "context_texts",
    "read_samples",
    "sample_from_result",
    "sample_from_retrieval",
    "write_samples",
]

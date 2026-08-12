"""Turning retrieved sources into the citations the answer actually used.

Retrieval returns ``top_k`` chunks; a two-sentence answer typically leans on two
of them. Returning all of them labelled "Sources" overstates what the answer was
grounded in, so the bracketed markers in the text decide what ships.

Numbers are NOT renumbered: ``[3]`` in the prose has to keep pointing at citation
3, so the surviving citations are sparse by design.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.llm.prompts import cited_numbers
from core.retrieval.base import RetrievalResult
from shared.schemas import Citation


def build_citations(answer: str, sources: Sequence[RetrievalResult]) -> list[Citation]:
    """Citations for the sources the answer cited, in retrieval order.

    If the model cited nothing, every source is returned rather than none —
    losing provenance entirely is worse than showing too much, and the caller
    logs the case so it stays visible.
    """
    all_citations = [
        Citation(
            number=i,
            professor_name=src.metadata.get("professor_name", ""),
            professor_title=src.metadata.get("professor_title", ""),
            section_type=src.metadata.get("section_type", ""),
            url=src.metadata.get("url", ""),
            score=src.score,
        )
        for i, src in enumerate(sources, start=1)
    ]

    used = cited_numbers(answer)
    if not used:
        return all_citations
    return [c for c in all_citations if c.number in used]

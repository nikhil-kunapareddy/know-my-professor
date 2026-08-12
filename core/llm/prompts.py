"""System instruction, prompt assembly, and citation parsing for the answer step.

The prompt tells the model to cite with ``[n]``, so the parser for those markers
lives here too — the convention and the code that reads it back stay together,
and the API and the eval harness share one definition instead of each keeping a
regex that can drift.
"""

from __future__ import annotations

import re

from core.retrieval.base import RetrievalResult

#: Matches the bracketed citation markers the system instruction asks for.
CITATION_MARKER = re.compile(r"\[(\d+)\]")

SYSTEM_INSTRUCTION = """\
You answer questions about faculty at Northeastern University's Khoury College of Computer Sciences.

Rules:
- Use ONLY the numbered context entries below to answer. Do not invent facts.
- Cite the professors you used with their bracketed numbers, e.g. [1], [3].
- If multiple professors are relevant, list them.
- If the context does not contain the answer, say "I don't have that information in my data."
- Be concise. Two or three sentences is usually enough.
"""


def cited_numbers(answer: str) -> set[int]:
    """The distinct ``[n]`` markers present in an answer."""
    return {int(m) for m in CITATION_MARKER.findall(answer)}


class PromptBuilder:
    """Formats retrieved chunks into the numbered context the model reads.

    The numbering here ([1], [2], …) is the same ordering the caller uses to
    build citations, so a bracketed reference in the answer maps back to a
    retrieved source.
    """

    @staticmethod
    def context_block(number: int, result: RetrievalResult) -> str:
        """Render one numbered context entry from a retrieval result."""
        md = result.metadata
        return (
            f"[{number}] {md.get('professor_name', '')} "
            f"({md.get('professor_title', '')}) — {md.get('section_type', '')}\n"
            f"{md.get('text', '')}"
        )

    def build_user_message(self, question: str, results: list[RetrievalResult]) -> str:
        """Assemble the user turn: numbered context followed by the question."""
        blocks = [self.context_block(i, r) for i, r in enumerate(results, start=1)]
        return "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n"

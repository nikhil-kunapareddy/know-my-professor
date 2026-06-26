"""Query analysis and preprocessing.

Not yet wired into the pipeline — this is the seam for future intent routing
(e.g. people-vs-course queries) and hybrid-vs-semantic strategy selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class QueryAnalysis:
    """Result of query analysis."""

    original: str
    cleaned: str
    intent: str
    keywords: List[str]
    entities: List[str]
    length: int
    is_question: bool


class QueryAnalyzer:
    """Analyze queries for better retrieval."""

    def __init__(self):
        self.entity_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "url": r"https?://[^\s]+",
            "course_code": r"\b[A-Z]{2,4}\s?\d{3,4}\b",
            "number": r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
        }

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze query and extract components.

        Returns:
            QueryAnalysis with detailed breakdown
        """
        cleaned = self._clean_text(query)
        keywords = self._extract_keywords(cleaned)
        entities = self._extract_entities(query)
        intent = self._detect_intent(cleaned)
        is_question = query.rstrip().endswith("?")

        return QueryAnalysis(
            original=query,
            cleaned=cleaned,
            intent=intent,
            keywords=keywords,
            entities=entities,
            length=len(cleaned.split()),
            is_question=is_question,
        )

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning: collapse whitespace, drop punctuation, lowercase."""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\w\s\-\?]", "", text)
        return text.lower()

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords (words longer than 3 chars)."""
        tokens = text.split()
        keywords = [t for t in tokens if len(t) > 3 and not t.startswith("-")]
        return list(set(keywords))[:10]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (emails, URLs, course codes, proper nouns)."""
        entities: List[str] = []
        for pattern in self.entity_patterns.values():
            entities.extend(re.findall(pattern, text))
        # Capitalized sequences (likely professor names).
        entities.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text))
        return list(set(entities))

    def _detect_intent(self, query: str) -> str:
        """Detect query intent against the faculty-directory domain."""
        q = query.lower()
        if any(w in q for w in ["who", "professor", "faculty", "teaches", "advisor"]):
            return "person"
        if any(w in q for w in ["course", "class", "teach", "syllabus"]):
            return "course"
        if any(w in q for w in ["research", "work on", "area", "interest", "lab"]):
            return "research_area"
        if any(w in q for w in ["where", "location", "campus"]):
            return "location"
        return "informational"

    def suggest_strategy(self, query: str) -> str:
        """Suggest a retrieval strategy for the query.

        Entity-bearing queries (names, course codes) benefit from lexical
        matching; everything else defaults to semantic search.
        """
        analysis = self.analyze(query)
        if analysis.length < 3:
            return "expand"
        if analysis.entities:
            return "hybrid"
        return "semantic"

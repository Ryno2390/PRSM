"""
Query Processor
===============

Query processing and understanding for PRSM.
Implements real keyword extraction and intent classification.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Processed query with extracted features."""
    original_text: str
    normalized_text: str
    intent: str
    entities: Dict[str, Any]
    keywords: List[str]


class QueryProcessor:
    """Process and understand user queries with real NLP techniques."""

    # Common stopwords to filter out
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
        "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
        "she", "her", "in", "on", "at", "by", "for", "with", "about", "to",
        "from", "of", "and", "or", "but", "if", "then", "else", "so", "because"
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    async def process(self, query: str) -> ProcessedQuery:
        """Process a query with real keyword extraction and intent classification."""
        normalized = query.lower().strip()

        # Real keyword extraction: remove stopwords, extract meaningful terms
        keywords = [w for w in normalized.split() if w not in self.STOPWORDS and len(w) > 2]

        # Intent classification based on query structure
        intent = self._classify_intent(normalized)

        # Extract entities using simple patterns
        entities = self._extract_entities(query)

        return ProcessedQuery(
            original_text=query,
            normalized_text=normalized,
            intent=intent,
            entities=entities,
            keywords=keywords,
        )

    def _classify_intent(self, text: str) -> str:
        """Classify intent from query structure."""
        text_lower = text.lower()

        # Definition queries
        if any(text_lower.startswith(w) for w in ("what is", "what are", "define", "explain")):
            return "definition"

        # Procedure queries
        if any(text_lower.startswith(w) for w in ("how do", "how to", "how can", "steps", "procedure")):
            return "procedure"

        # Explanation queries
        if any(text_lower.startswith(w) for w in ("why", "reason", "cause", "because")):
            return "explanation"

        # Comparison queries
        if any(w in text_lower for w in ("compare", "difference", "versus", "vs", "better")):
            return "comparison"

        # Analysis queries
        if any(w in text_lower for w in ("analyze", "analysis", "evaluate", "assess")):
            return "analysis"

        # List queries
        if any(text_lower.startswith(w) for w in ("list", "show me", "give me")):
            return "list"

        return "query"

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities using simple pattern matching."""
        import re

        entities = {
            "dates": [],
            "numbers": [],
            "urls": [],
            "proper_nouns": []
        }

        # Extract dates (simple pattern)
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
        entities["dates"] = re.findall(date_pattern, text)

        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        entities["numbers"] = re.findall(number_pattern, text)

        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        entities["urls"] = re.findall(url_pattern, text)

        # Extract potential proper nouns (capitalized words not at sentence start)
        words = text.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and word not in self.STOPWORDS:
                entities["proper_nouns"].append(word)

        return entities

    async def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        normalized = text.lower().strip()
        return [w for w in normalized.split() if w not in self.STOPWORDS and len(w) > 2]

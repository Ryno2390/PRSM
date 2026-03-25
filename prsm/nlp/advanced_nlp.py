"""
Advanced NLP Processor
======================

Advanced natural language processing for PRSM queries.
Uses QueryProcessor internally for text analysis.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class NLPResult:
    """Result from NLP processing"""
    text: str
    tokens: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    intent: str = ""
    confidence: float = 0.0


class AdvancedNLPProcessor:
    """Advanced NLP processor for query understanding."""

    # Common stopwords
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
        self._initialized = False
        self._query_processor = None

    async def initialize(self) -> bool:
        """Initialize the NLP processor."""
        try:
            from prsm.nlp.query_processor import QueryProcessor
            self._query_processor = QueryProcessor(self.config)
        except Exception as e:
            logger.debug(f"Could not initialize QueryProcessor: {e}")
        self._initialized = True
        return True

    async def process(self, text: str) -> NLPResult:
        """Process text and extract features."""
        # Tokenize
        tokens = text.split()

        # Extract entities
        entities = await self.extract_entities(text)

        # Classify intent
        intent = await self.classify_intent(text)

        # Calculate confidence based on processing quality
        confidence = self._calculate_confidence(tokens, entities, intent)

        return NLPResult(
            text=text,
            tokens=tokens,
            entities=entities,
            intent=intent,
            confidence=confidence
        )

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text using regex patterns."""
        entities = []

        # Extract dates
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
        for match in re.finditer(date_pattern, text):
            entities.append({
                "type": "date",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        for match in re.finditer(number_pattern, text):
            entities.append({
                "type": "number",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

        # Extract URLs
        url_pattern = r'https?://[^\s]+'
        for match in re.finditer(url_pattern, text):
            entities.append({
                "type": "url",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                "type": "email",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })

        # Extract proper nouns (capitalized words not at sentence start)
        words = text.split()
        for i, word in enumerate(words):
            clean_word = word.strip('.,!?;:"\'')
            if i > 0 and clean_word and clean_word[0].isupper() and clean_word.lower() not in self.STOPWORDS:
                entities.append({
                    "type": "proper_noun",
                    "value": clean_word,
                    "position": i
                })

        return entities

    async def classify_intent(self, text: str) -> str:
        """Classify the intent of the text."""
        text_lower = text.lower()

        if any(text_lower.startswith(w) for w in ("what is", "what are", "define", "explain")):
            return "definition"
        if any(text_lower.startswith(w) for w in ("how do", "how to", "how can", "steps")):
            return "procedure"
        if any(text_lower.startswith(w) for w in ("why", "reason", "cause")):
            return "explanation"
        if any(w in text_lower for w in ("compare", "difference", "versus", "vs")):
            return "comparison"
        if any(w in text_lower for w in ("analyze", "analysis", "evaluate")):
            return "analysis"

        return "query"

    def _calculate_confidence(self, tokens: List[str], entities: List[Dict], intent: str) -> float:
        """Calculate processing confidence based on results."""
        base_confidence = 0.5

        # Boost for more tokens
        if len(tokens) > 3:
            base_confidence += 0.1

        # Boost for entities found
        if entities:
            base_confidence += 0.1 * min(len(entities), 3)

        # Boost for specific intents
        if intent in ("definition", "procedure", "comparison"):
            base_confidence += 0.1

        return min(base_confidence, 0.95)

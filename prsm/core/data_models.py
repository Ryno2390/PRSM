"""
PRSM Data Models - Query and Response Types
============================================

Core data models for query processing and response generation.
These models define the interfaces for the PRSM query pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from uuid import uuid4


@dataclass
class QueryRequest:
    """
    Request model for PRSM queries.

    Attributes:
        user_id: Unique identifier for the user making the request
        query_text: The user's query text
        context: Optional context dictionary with user preferences, domain info, etc.
        session_id: Optional session ID for conversation continuity
        query_id: Unique identifier for this specific query
    """

    user_id: str
    query_text: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    query_id: str = field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "query_text": self.query_text,
            "context": self.context,
            "session_id": self.session_id,
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class QueryResponse:
    """
    Response model for PRSM queries.

    Attributes:
        query_id: Unique identifier matching the request
        response_text: The generated response text
        confidence_score: Confidence score (0.0 to 1.0)
        sources: List of source identifiers used in the response
        metadata: Additional metadata about the response
        processing_time: Time taken to process the query in seconds
    """

    query_id: str
    response_text: str
    confidence_score: float = 0.0
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_id": self.query_id,
            "response_text": self.response_text,
            "confidence_score": self.confidence_score,
            "sources": self.sources,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SearchResult:
    """
    Result from knowledge base search.

    Attributes:
        doc_id: Document identifier
        content: Document content
        score: Relevance/similarity score
        metadata: Document metadata
    """

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata
        }

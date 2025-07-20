#!/usr/bin/env python3
"""
NWTN Data Models
================

Common data structures used across the NWTN system for semantic search and embeddings.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Any
from datetime import datetime


@dataclass
class PaperData:
    """Structured paper data for embedding"""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    domain: str
    categories: List[str]
    published_date: str
    file_path: str
    
    def get_text_content(self) -> str:
        """Get combined text content for embedding"""
        return f"{self.title}\n\n{self.abstract}"


@dataclass
class PaperEmbedding:
    """Paper with its embedding vector"""
    paper_data: PaperData
    embedding: np.ndarray
    embedding_model: str
    created_at: str


@dataclass
class SemanticSearchResult:
    """Result from semantic search"""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    similarity_score: float
    domain: str
    categories: List[str]
    published_date: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'similarity_score': self.similarity_score,
            'domain': self.domain,
            'categories': self.categories,
            'published_date': self.published_date
        }


@dataclass
class RetrievedPaper:
    """Paper retrieved from semantic search"""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    content: str
    relevance_score: float
    metadata: dict
    
    def __post_init__(self):
        """Ensure content is set"""
        if not self.content:
            self.content = f"{self.title}\n\n{self.abstract}"
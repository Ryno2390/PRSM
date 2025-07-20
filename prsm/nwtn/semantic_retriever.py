#!/usr/bin/env python3
"""
Semantic Retriever for NWTN System 1 → System 2 → Attribution Pipeline
====================================================================

This module implements embedding-based semantic search for the NWTN system,
replacing simple keyword search with sophisticated semantic retrieval.

Part of Phase 1.1 of the NWTN System 1 → System 2 → Attribution roadmap.
"""

import asyncio
import structlog
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
from prsm.nwtn.external_storage_config import ExternalStorageManager, ExternalKnowledgeBase

logger = structlog.get_logger(__name__)


@dataclass
class RetrievedPaper:
    """Represents a paper retrieved through semantic search"""
    paper_id: str
    title: str
    authors: str
    abstract: str
    arxiv_id: str
    publish_date: str
    relevance_score: float
    similarity_score: float
    retrieval_method: str = "semantic_embedding"
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SemanticSearchResult:
    """Complete result of semantic search operation"""
    query: str
    retrieved_papers: List[RetrievedPaper]
    search_time_seconds: float
    total_papers_searched: int
    retrieval_method: str
    embedding_model: str
    search_id: str = field(default_factory=lambda: str(uuid4()))


class TextEmbeddingGenerator:
    """Generates text embeddings for semantic search"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = 384  # Default for MiniLM
        
    async def initialize(self):
        """Initialize the embedding model"""
        try:
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info("Text embedding model initialized",
                       model=self.model_name,
                       dimension=self.embedding_dimension)
            return True
        except ImportError:
            logger.warning("sentence-transformers not available, using mock embeddings")
            self.model = None
            return False
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if self.model is None:
            # Mock embedding for testing
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.rand(self.embedding_dimension).tolist()
            return embedding
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return mock embedding as fallback
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.rand(self.embedding_dimension).tolist()
            return embedding


class SemanticRetriever:
    """
    Advanced semantic retrieval system for NWTN
    
    Implements embedding-based search with relevance scoring and configurable
    retrieval parameters for System 1 candidate generation.
    """
    
    def __init__(self, 
                 external_knowledge_base: ExternalKnowledgeBase,
                 embedding_generator: Optional[TextEmbeddingGenerator] = None):
        self.external_knowledge_base = external_knowledge_base
        self.embedding_generator = embedding_generator or TextEmbeddingGenerator()
        self.initialized = False
        
        # Configurable retrieval parameters - optimized for more diverse results
        self.default_top_k = 25
        self.default_similarity_threshold = 0.2
        self.max_papers_to_search = 150000  # Scale to full 150K corpus
        
        # Performance tracking
        self.retrieval_stats = {
            'total_retrievals': 0,
            'successful_retrievals': 0,
            'average_search_time': 0.0,
            'total_papers_retrieved': 0
        }
    
    async def initialize(self):
        """Initialize the semantic retriever"""
        try:
            # Initialize embedding generator
            embedding_ready = await self.embedding_generator.initialize()
            
            # Ensure external knowledge base is ready
            if not self.external_knowledge_base.initialized:
                await self.external_knowledge_base.initialize()
            
            self.initialized = True
            logger.info("Semantic retriever initialized",
                       embedding_model=self.embedding_generator.model_name,
                       embedding_ready=embedding_ready,
                       knowledge_base_ready=self.external_knowledge_base.initialized)
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic retriever: {e}")
            return False
    
    async def semantic_search(self, 
                            query: str,
                            top_k: Optional[int] = None,
                            similarity_threshold: Optional[float] = None,
                            search_method: str = "hybrid") -> SemanticSearchResult:
        """
        Perform semantic search for papers relevant to the query
        
        Args:
            query: Natural language query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            search_method: "semantic", "keyword", or "hybrid"
            
        Returns:
            SemanticSearchResult with retrieved papers and metadata
        """
        start_time = datetime.now(timezone.utc)
        
        if not self.initialized:
            await self.initialize()
        
        top_k = top_k or self.default_top_k
        similarity_threshold = similarity_threshold or self.default_similarity_threshold
        
        try:
            retrieved_papers = []
            
            if search_method in ["semantic", "hybrid"]:
                # Semantic search using embeddings
                semantic_papers = await self._semantic_search_embeddings(
                    query, top_k, similarity_threshold
                )
                retrieved_papers.extend(semantic_papers)
            
            if search_method in ["keyword", "hybrid"]:
                # Keyword search as fallback/supplement
                keyword_papers = await self._keyword_search_fallback(
                    query, top_k
                )
                retrieved_papers.extend(keyword_papers)
            
            # Remove duplicates and sort by relevance
            retrieved_papers = self._deduplicate_and_rank(retrieved_papers)
            retrieved_papers = retrieved_papers[:top_k]
            
            # Calculate search time
            search_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update statistics
            self._update_retrieval_stats(search_time, len(retrieved_papers))
            
            result = SemanticSearchResult(
                query=query,
                retrieved_papers=retrieved_papers,
                search_time_seconds=search_time,
                total_papers_searched=self.max_papers_to_search,
                retrieval_method=search_method,
                embedding_model=self.embedding_generator.model_name
            )
            
            logger.info("Semantic search completed",
                       query=query[:50],
                       papers_found=len(retrieved_papers),
                       search_time=search_time,
                       method=search_method)
            
            return result
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            # Return empty result on failure
            return SemanticSearchResult(
                query=query,
                retrieved_papers=[],
                search_time_seconds=0.0,
                total_papers_searched=0,
                retrieval_method="failed",
                embedding_model=self.embedding_generator.model_name
            )
    
    async def _semantic_search_embeddings(self, 
                                        query: str,
                                        top_k: int,
                                        similarity_threshold: float) -> List[RetrievedPaper]:
        """Perform embedding-based semantic search"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_embedding(query)
            
            # Search embeddings in external knowledge base
            if hasattr(self.external_knowledge_base.storage_manager, 'search_embeddings'):
                embedding_results = await self.external_knowledge_base.storage_manager.search_embeddings(
                    query_embedding, max_results=top_k
                )
                
                papers = []
                for result in embedding_results:
                    if result['similarity'] >= similarity_threshold:
                        # Convert embedding result to paper
                        metadata = result.get('metadata', {})
                        paper = RetrievedPaper(
                            paper_id=metadata.get('id', f"embed_{result['batch_id']}_{result['index']}"),
                            title=metadata.get('title', 'Unknown Title'),
                            authors=metadata.get('authors', 'Unknown Authors'),
                            abstract=metadata.get('abstract', ''),
                            arxiv_id=metadata.get('arxiv_id', ''),
                            publish_date=metadata.get('publish_date', ''),
                            relevance_score=result['similarity'],
                            similarity_score=result['similarity'],
                            retrieval_method="semantic_embedding"
                        )
                        papers.append(paper)
                
                return papers
            
        except Exception as e:
            logger.warning(f"Embedding search failed: {e}")
        
        return []
    
    async def _keyword_search_fallback(self, query: str, top_k: int) -> List[RetrievedPaper]:
        """Perform keyword search as fallback"""
        try:
            # Use existing keyword search from external knowledge base
            papers = await self.external_knowledge_base.search_papers(query, max_results=top_k)
            
            retrieved_papers = []
            for paper in papers:
                retrieved_paper = RetrievedPaper(
                    paper_id=paper.get('id', ''),
                    title=paper.get('title', ''),
                    authors=paper.get('authors', ''),
                    abstract=paper.get('abstract', ''),
                    arxiv_id=paper.get('arxiv_id', ''),
                    publish_date=paper.get('publish_date', ''),
                    relevance_score=paper.get('relevance_score', 0.5),
                    similarity_score=paper.get('relevance_score', 0.5),
                    retrieval_method="keyword_search"
                )
                retrieved_papers.append(retrieved_paper)
            
            return retrieved_papers
            
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []
    
    def _deduplicate_and_rank(self, papers: List[RetrievedPaper]) -> List[RetrievedPaper]:
        """Remove duplicates and rank by relevance"""
        # Remove duplicates by paper_id
        seen_ids = set()
        unique_papers = []
        
        for paper in papers:
            if paper.paper_id not in seen_ids:
                seen_ids.add(paper.paper_id)
                unique_papers.append(paper)
        
        # Sort by relevance score (descending)
        unique_papers.sort(key=lambda p: p.relevance_score, reverse=True)
        
        return unique_papers
    
    def _update_retrieval_stats(self, search_time: float, papers_retrieved: int):
        """Update retrieval statistics"""
        self.retrieval_stats['total_retrievals'] += 1
        self.retrieval_stats['successful_retrievals'] += 1 if papers_retrieved > 0 else 0
        self.retrieval_stats['total_papers_retrieved'] += papers_retrieved
        
        # Update average search time
        total_time = (self.retrieval_stats['average_search_time'] * 
                     (self.retrieval_stats['total_retrievals'] - 1) + search_time)
        self.retrieval_stats['average_search_time'] = total_time / self.retrieval_stats['total_retrievals']
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval performance statistics"""
        return {
            **self.retrieval_stats,
            'success_rate': (self.retrieval_stats['successful_retrievals'] / 
                           max(1, self.retrieval_stats['total_retrievals'])),
            'average_papers_per_retrieval': (self.retrieval_stats['total_papers_retrieved'] / 
                                           max(1, self.retrieval_stats['successful_retrievals']))
        }
    
    async def configure_retrieval_params(self, 
                                       top_k: Optional[int] = None,
                                       similarity_threshold: Optional[float] = None,
                                       max_papers_to_search: Optional[int] = None):
        """Configure retrieval parameters"""
        if top_k is not None:
            self.default_top_k = top_k
        if similarity_threshold is not None:
            self.default_similarity_threshold = similarity_threshold
        if max_papers_to_search is not None:
            self.max_papers_to_search = max_papers_to_search
        
        logger.info("Retrieval parameters updated",
                   top_k=self.default_top_k,
                   similarity_threshold=self.default_similarity_threshold,
                   max_papers_to_search=self.max_papers_to_search)


# Factory function for easy instantiation
async def create_semantic_retriever(external_knowledge_base: ExternalKnowledgeBase) -> SemanticRetriever:
    """Create and initialize a semantic retriever"""
    embedding_generator = TextEmbeddingGenerator()
    retriever = SemanticRetriever(external_knowledge_base, embedding_generator)
    await retriever.initialize()
    return retriever
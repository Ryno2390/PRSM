"""
Abstract base classes for PRSM vector store implementations

Defines the unified interface that all vector database backends must implement,
ensuring seamless transitions between different vector stores during scaling.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np
from pydantic import BaseModel, Field


class VectorStoreType(str, Enum):
    """Supported vector database types"""
    PGVECTOR = "pgvector"
    MILVUS = "milvus" 
    QDRANT = "qdrant"
    PINECONE = "pinecone"  # For initial prototyping only


class ContentType(str, Enum):
    """Types of content that can be stored"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    RESEARCH_PAPER = "research_paper"
    DATASET = "dataset"


@dataclass
class VectorStoreConfig:
    """Configuration for vector database connection"""
    store_type: VectorStoreType
    host: str
    port: int
    database: str
    username: Optional[str] = None
    password: Optional[str] = None
    collection_name: str = "prsm_content"
    
    # Performance settings
    max_connections: int = 100
    connection_timeout: int = 30
    query_timeout: int = 10
    
    # Vector settings
    vector_dimension: int = 1536  # OpenAI ada-002 default
    similarity_metric: str = "cosine"
    
    # IPFS integration
    ipfs_gateway: str = "https://ipfs.prsm.ai"
    enable_ipfs_verification: bool = True


class ContentMatch(BaseModel):
    """Result from vector similarity search"""
    content_cid: str = Field(..., description="IPFS content identifier")
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Provenance information
    creator_id: Optional[str] = None
    royalty_rate: float = Field(default=0.08, description="Creator royalty percentage")
    content_type: ContentType = ContentType.TEXT
    
    # Usage tracking
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = None
    
    # Quality metrics
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    peer_review_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    citation_count: int = Field(default=0)


class SearchFilters(BaseModel):
    """Filters for vector search queries"""
    content_types: Optional[List[ContentType]] = None
    creator_ids: Optional[List[str]] = None
    date_range: Optional[tuple[datetime, datetime]] = None
    min_quality_score: Optional[float] = None
    max_royalty_rate: Optional[float] = None
    require_open_license: bool = False
    exclude_content_cids: Optional[List[str]] = None


class PRSMVectorStore(ABC):
    """
    Abstract base class for all PRSM vector database implementations
    
    Provides unified interface for vector operations with built-in:
    - IPFS content addressing integration
    - Provenance tracking for creator royalties
    - Multi-modal content support
    - Performance monitoring
    - Migration support between different vector stores
    """
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.is_connected = False
        self.performance_metrics = {
            "total_queries": 0,
            "total_storage_operations": 0,
            "average_query_time": 0.0,
            "average_storage_time": 0.0,
            "error_count": 0
        }
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to vector database"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to vector database"""
        pass
    
    @abstractmethod
    async def create_collection(self, collection_name: str, 
                              vector_dimension: int,
                              metadata_schema: Dict[str, Any]) -> bool:
        """Create a new collection/table for storing vectors"""
        pass
    
    @abstractmethod
    async def store_content_with_embeddings(self, 
                                          content_cid: str,
                                          embeddings: np.ndarray,
                                          metadata: Dict[str, Any]) -> str:
        """
        Store content embeddings with IPFS provenance tracking
        
        Args:
            content_cid: IPFS content identifier
            embeddings: Vector embeddings for the content
            metadata: Content metadata including creator info and royalty rates
            
        Returns:
            vector_id: Unique identifier for the stored vector
        """
        pass
    
    @abstractmethod
    async def search_similar_content(self, 
                                   query_embedding: np.ndarray,
                                   filters: Optional[SearchFilters] = None,
                                   top_k: int = 10) -> List[ContentMatch]:
        """
        Search for similar content with provenance tracking
        
        Args:
            query_embedding: Vector embedding of the search query
            filters: Optional filters to narrow search results
            top_k: Maximum number of results to return
            
        Returns:
            List of ContentMatch objects with similarity scores and metadata
        """
        pass
    
    @abstractmethod
    async def update_content_metadata(self, 
                                    content_cid: str,
                                    metadata_updates: Dict[str, Any]) -> bool:
        """Update metadata for existing content"""
        pass
    
    @abstractmethod
    async def delete_content(self, content_cid: str) -> bool:
        """Remove content from vector store (for DMCA compliance)"""
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        pass
    
    # Concrete methods for common functionality
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health and performance of the vector store"""
        try:
            start_time = asyncio.get_event_loop().time()
            stats = await self.get_collection_stats()
            response_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "connection_status": self.is_connected,
                "collection_stats": stats,
                "performance_metrics": self.performance_metrics
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection_status": self.is_connected
            }
    
    async def batch_store_content(self, 
                                content_batch: List[tuple[str, np.ndarray, Dict[str, Any]]],
                                batch_size: int = 100) -> List[str]:
        """
        Store multiple content items efficiently
        
        Args:
            content_batch: List of (content_cid, embeddings, metadata) tuples
            batch_size: Number of items to process in each batch
            
        Returns:
            List of vector IDs for stored content
        """
        vector_ids = []
        
        for i in range(0, len(content_batch), batch_size):
            batch = content_batch[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.store_content_with_embeddings(cid, emb, meta)
                for cid, emb, meta in batch
            ])
            vector_ids.extend(batch_results)
        
        return vector_ids
    
    async def search_with_royalty_tracking(self,
                                         query_embedding: np.ndarray,
                                         user_id: str,
                                         filters: Optional[SearchFilters] = None,
                                         top_k: int = 10) -> List[ContentMatch]:
        """
        Search content and automatically track access for royalty payments
        
        This method performs the search and then records the access for
        later FTNS royalty distribution to content creators.
        """
        # Perform the search
        results = await self.search_similar_content(query_embedding, filters, top_k)
        
        # Track access for royalty calculation
        # Note: This will integrate with the FTNS service in Phase 1
        for result in results:
            await self._track_content_access(result.content_cid, user_id, result.similarity_score)
        
        return results
    
    async def _track_content_access(self, content_cid: str, user_id: str, relevance_score: float):
        """Track content access for royalty calculation"""
        # Placeholder for FTNS integration
        # Will be implemented in Phase 1C when FTNS service is integrated
        pass
    
    def _update_performance_metrics(self, operation_type: str, duration: float, success: bool):
        """Update internal performance tracking"""
        if operation_type == "query":
            self.performance_metrics["total_queries"] += 1
            current_avg = self.performance_metrics["average_query_time"]
            total_queries = self.performance_metrics["total_queries"]
            self.performance_metrics["average_query_time"] = (
                (current_avg * (total_queries - 1) + duration) / total_queries
            )
        elif operation_type == "storage":
            self.performance_metrics["total_storage_operations"] += 1
            current_avg = self.performance_metrics["average_storage_time"]
            total_ops = self.performance_metrics["total_storage_operations"]
            self.performance_metrics["average_storage_time"] = (
                (current_avg * (total_ops - 1) + duration) / total_ops
            )
        
        if not success:
            self.performance_metrics["error_count"] += 1


class VectorStoreBenchmark:
    """Benchmarking utilities for vector store performance testing"""
    
    @staticmethod
    async def benchmark_search_performance(store: PRSMVectorStore,
                                         test_embeddings: List[np.ndarray],
                                         iterations: int = 100) -> Dict[str, Any]:
        """Benchmark search performance across multiple queries"""
        total_time = 0.0
        successful_queries = 0
        
        for i in range(iterations):
            embedding = test_embeddings[i % len(test_embeddings)]
            
            start_time = asyncio.get_event_loop().time()
            try:
                results = await store.search_similar_content(embedding, top_k=10)
                duration = asyncio.get_event_loop().time() - start_time
                total_time += duration
                successful_queries += 1
            except Exception as e:
                print(f"Query {i} failed: {e}")
        
        avg_query_time = total_time / successful_queries if successful_queries > 0 else 0
        
        return {
            "total_queries": iterations,
            "successful_queries": successful_queries,
            "success_rate": successful_queries / iterations,
            "average_query_time_ms": avg_query_time * 1000,
            "queries_per_second": successful_queries / total_time if total_time > 0 else 0
        }
    
    @staticmethod
    async def benchmark_storage_performance(store: PRSMVectorStore,
                                          test_data: List[tuple[str, np.ndarray, Dict]],
                                          batch_size: int = 10) -> Dict[str, Any]:
        """Benchmark storage performance for batch operations"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            vector_ids = await store.batch_store_content(test_data, batch_size)
            total_time = asyncio.get_event_loop().time() - start_time
            
            return {
                "total_items": len(test_data),
                "successful_items": len(vector_ids),
                "success_rate": len(vector_ids) / len(test_data),
                "total_time_seconds": total_time,
                "items_per_second": len(vector_ids) / total_time if total_time > 0 else 0,
                "average_storage_time_ms": (total_time / len(vector_ids)) * 1000 if vector_ids else 0
            }
        except Exception as e:
            return {
                "error": str(e),
                "total_items": len(test_data),
                "successful_items": 0,
                "success_rate": 0.0
            }
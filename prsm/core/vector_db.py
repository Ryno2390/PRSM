"""
PRSM Vector Database Integration - Model Embeddings and Semantic Search

🎯 PURPOSE IN PRSM:
This module provides comprehensive vector database integration for PRSM, implementing
semantic search, model similarity matching, and intelligent model recommendation
across the distributed ecosystem.

🔧 INTEGRATION POINTS:
- Model discovery: Semantic search for similar models based on capabilities
- Content similarity: Find related research papers and datasets
- Query matching: Route queries to optimal models based on semantic similarity  
- Knowledge clustering: Group similar concepts and research domains
- Recommendation engine: Suggest relevant models and content to users
- Quality assessment: Compare model outputs for consistency validation

🚀 REAL-WORLD CAPABILITIES:
- Multi-provider support (Pinecone, Weaviate, Chroma, Qdrant)
- High-dimensional embedding storage and retrieval
- Similarity search with configurable distance metrics
- Batch operations for efficient bulk processing
- Metadata filtering for precise search refinement
- Real-time indexing of new models and content
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List, Tuple, Union, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import structlog

import numpy as np
from prsm.core.config import get_settings
from prsm.core.redis_client import get_model_cache

logger = structlog.get_logger(__name__)
settings = get_settings()


class VectorProvider(Enum):
    """Supported vector database providers"""
    PINECONE = "pinecone"
    WEAVIATE = "weaviate" 
    CHROMA = "chroma"
    QDRANT = "qdrant"
    LOCAL = "local"


class DistanceMetric(Enum):
    """Distance metrics for similarity search"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dotproduct"
    MANHATTAN = "manhattan"


@dataclass
class VectorSearchResult:
    """Result from vector similarity search"""
    id: str
    score: float
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass
class EmbeddingBatch:
    """Batch of embeddings for bulk operations"""
    ids: List[str]
    vectors: List[List[float]]
    metadata: List[Dict[str, Any]]


class BaseVectorDB(ABC):
    """
    Abstract base class for vector database implementations
    
    🎯 PURPOSE: Provides consistent interface across different vector
    database providers while allowing for provider-specific optimizations
    """
    
    def __init__(self, provider: VectorProvider, **config):
        self.provider = provider
        self.config = config
        self.connected = False
        self.client = None
    
    @abstractmethod
    async def initialize(self):
        """Initialize connection to vector database"""
        pass
    
    @abstractmethod
    async def create_index(self, index_name: str, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE) -> bool:
        """Create a new vector index"""
        pass
    
    @abstractmethod
    async def upsert_vectors(self, index_name: str, vectors: EmbeddingBatch) -> bool:
        """Insert or update vectors in the index"""
        pass
    
    @abstractmethod
    async def search_similar(
        self, 
        index_name: str, 
        query_vector: List[float], 
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    async def delete_vectors(self, index_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors by ID"""
        pass
    
    @abstractmethod
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get index statistics"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check database health"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up connections"""
        pass


class PineconeVectorDB(BaseVectorDB):
    """
    Pinecone vector database implementation
    
    🌲 PINECONE INTEGRATION:
    Production-grade vector database with excellent performance
    and managed infrastructure for high-scale deployments
    """
    
    def __init__(self, **config):
        super().__init__(VectorProvider.PINECONE, **config)
        self.api_key = config.get("api_key") or settings.pinecone_api_key
        self.environment = config.get("environment") or settings.pinecone_environment
        
    async def initialize(self):
        """Initialize Pinecone client"""
        try:
            import pinecone
            
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            self.client = pinecone
            self.connected = True
            
            logger.info("Pinecone client initialized successfully")
            
        except ImportError:
            raise RuntimeError("Pinecone client not installed. Run: pip install pinecone-client")
        except Exception as e:
            logger.error("Failed to initialize Pinecone", error=str(e))
            raise
    
    async def create_index(self, index_name: str, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE) -> bool:
        """Create Pinecone index"""
        try:
            if not self.connected:
                return False
            
            # Map our metric to Pinecone metric
            pinecone_metrics = {
                DistanceMetric.COSINE: "cosine",
                DistanceMetric.EUCLIDEAN: "euclidean", 
                DistanceMetric.DOT_PRODUCT: "dotproduct"
            }
            
            metric_name = pinecone_metrics.get(metric, "cosine")
            
            # Check if index already exists
            if index_name in self.client.list_indexes():
                logger.info("Pinecone index already exists", index_name=index_name)
                return True
            
            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric_name,
                pod_type="p1.x1"  # Starter pod type
            )
            
            # Wait for index to be ready
            while not self.client.describe_index(index_name).status['ready']:
                await asyncio.sleep(1)
            
            logger.info("Pinecone index created", 
                       index_name=index_name,
                       dimension=dimension,
                       metric=metric_name)
            
            return True
            
        except Exception as e:
            logger.error("Failed to create Pinecone index",
                        index_name=index_name,
                        error=str(e))
            return False
    
    async def upsert_vectors(self, index_name: str, vectors: EmbeddingBatch) -> bool:
        """Upsert vectors to Pinecone index"""
        try:
            if not self.connected:
                return False
            
            index = self.client.Index(index_name)
            
            # Prepare vectors for Pinecone format
            pinecone_vectors = []
            for i, (vec_id, vector, metadata) in enumerate(zip(vectors.ids, vectors.vectors, vectors.metadata)):
                pinecone_vectors.append({
                    "id": vec_id,
                    "values": vector,
                    "metadata": metadata
                })
            
            # Batch upsert (Pinecone handles batching internally)
            index.upsert(vectors=pinecone_vectors)
            
            logger.debug("Vectors upserted to Pinecone",
                        index_name=index_name,
                        count=len(pinecone_vectors))
            
            return True
            
        except Exception as e:
            logger.error("Failed to upsert vectors to Pinecone",
                        index_name=index_name,
                        error=str(e))
            return False
    
    async def search_similar(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search similar vectors in Pinecone"""
        try:
            if not self.connected:
                return []
            
            index = self.client.Index(index_name)
            
            # Perform similarity search
            search_kwargs = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
                "include_values": False
            }
            
            if filter_metadata:
                search_kwargs["filter"] = filter_metadata
            
            results = index.query(**search_kwargs)
            
            # Convert to standard format
            search_results = []
            for match in results["matches"]:
                search_results.append(VectorSearchResult(
                    id=match["id"],
                    score=match["score"],
                    metadata=match.get("metadata", {}),
                    vector=match.get("values")
                ))
            
            logger.debug("Pinecone similarity search completed",
                        index_name=index_name,
                        results_count=len(search_results),
                        top_k=top_k)
            
            return search_results
            
        except Exception as e:
            logger.error("Failed to search Pinecone",
                        index_name=index_name,
                        error=str(e))
            return []
    
    async def delete_vectors(self, index_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors from Pinecone index"""
        try:
            if not self.connected:
                return False
            
            index = self.client.Index(index_name)
            index.delete(ids=vector_ids)
            
            logger.debug("Vectors deleted from Pinecone",
                        index_name=index_name,
                        count=len(vector_ids))
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete vectors from Pinecone",
                        index_name=index_name,
                        error=str(e))
            return False
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        try:
            if not self.connected:
                return {}
            
            index = self.client.Index(index_name)
            stats = index.describe_index_stats()
            
            return {
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "index_fullness": stats.get("index_fullness", 0.0),
                "namespaces": stats.get("namespaces", {})
            }
            
        except Exception as e:
            logger.error("Failed to get Pinecone index stats",
                        index_name=index_name,
                        error=str(e))
            return {}
    
    async def health_check(self) -> bool:
        """Check Pinecone health"""
        try:
            if not self.connected:
                return False
            
            # List indexes as a health check
            indexes = self.client.list_indexes()
            return True
            
        except Exception as e:
            logger.error("Pinecone health check failed", error=str(e))
            return False
    
    async def cleanup(self):
        """Clean up Pinecone connections"""
        try:
            self.connected = False
            self.client = None
            logger.info("Pinecone cleanup completed")
        except Exception as e:
            logger.error("Error during Pinecone cleanup", error=str(e))


class WeaviateVectorDB(BaseVectorDB):
    """
    Weaviate vector database implementation
    
    🕸️ WEAVIATE INTEGRATION:
    Open-source vector database with GraphQL API and built-in
    semantic search capabilities for complex queries
    """
    
    def __init__(self, **config):
        super().__init__(VectorProvider.WEAVIATE, **config)
        self.url = config.get("url") or settings.weaviate_url
        self.api_key = config.get("api_key") or settings.weaviate_api_key
    
    async def initialize(self):
        """Initialize Weaviate client"""
        try:
            import weaviate
            
            auth_config = None
            if self.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.api_key)
            
            self.client = weaviate.Client(
                url=self.url,
                auth_client_secret=auth_config
            )
            
            # Test connection
            self.client.schema.get()
            self.connected = True
            
            logger.info("Weaviate client initialized successfully")
            
        except ImportError:
            raise RuntimeError("Weaviate client not installed. Run: pip install weaviate-client")
        except Exception as e:
            logger.error("Failed to initialize Weaviate", error=str(e))
            raise
    
    async def create_index(self, index_name: str, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE) -> bool:
        """Create Weaviate class (index)"""
        try:
            if not self.connected:
                return False
            
            # Map distance metric
            weaviate_metrics = {
                DistanceMetric.COSINE: "cosine",
                DistanceMetric.EUCLIDEAN: "l2-squared",
                DistanceMetric.DOT_PRODUCT: "dot",
                DistanceMetric.MANHATTAN: "manhattan"
            }
            
            class_schema = {
                "class": index_name,
                "description": f"PRSM vector index for {index_name}",
                "vectorizer": "none",  # We provide vectors directly
                "vectorIndexConfig": {
                    "distance": weaviate_metrics.get(metric, "cosine")
                },
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Content or description"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Additional metadata"
                    }
                ]
            }
            
            # Check if class already exists
            existing_schema = self.client.schema.get()
            existing_classes = [cls["class"] for cls in existing_schema.get("classes", [])]
            
            if index_name in existing_classes:
                logger.info("Weaviate class already exists", class_name=index_name)
                return True
            
            self.client.schema.create_class(class_schema)
            
            logger.info("Weaviate class created",
                       class_name=index_name,
                       dimension=dimension,
                       metric=metric.value)
            
            return True
            
        except Exception as e:
            logger.error("Failed to create Weaviate class",
                        class_name=index_name,
                        error=str(e))
            return False
    
    async def upsert_vectors(self, index_name: str, vectors: EmbeddingBatch) -> bool:
        """Upsert vectors to Weaviate"""
        try:
            if not self.connected:
                return False
            
            # Batch import vectors
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for vec_id, vector, metadata in zip(vectors.ids, vectors.vectors, vectors.metadata):
                    data_object = {
                        "content": metadata.get("content", ""),
                        "metadata": metadata
                    }
                    
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=index_name,
                        uuid=vec_id,
                        vector=vector
                    )
            
            logger.debug("Vectors upserted to Weaviate",
                        class_name=index_name,
                        count=len(vectors.ids))
            
            return True
            
        except Exception as e:
            logger.error("Failed to upsert vectors to Weaviate",
                        class_name=index_name,
                        error=str(e))
            return False
    
    async def search_similar(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search similar vectors in Weaviate"""
        try:
            if not self.connected:
                return []
            
            # Build GraphQL query
            query = (
                self.client.query
                .get(index_name, ["content", "metadata"])
                .with_near_vector({"vector": query_vector})
                .with_limit(top_k)
                .with_additional(["id", "distance"])
            )
            
            # Add filters if provided
            if filter_metadata:
                where_filter = self._build_weaviate_filter(filter_metadata)
                if where_filter:
                    query = query.with_where(where_filter)
            
            results = query.do()
            
            # Convert to standard format
            search_results = []
            objects = results.get("data", {}).get("Get", {}).get(index_name, [])
            
            for obj in objects:
                additional = obj.get("_additional", {})
                search_results.append(VectorSearchResult(
                    id=additional.get("id", ""),
                    score=1.0 - additional.get("distance", 1.0),  # Convert distance to similarity
                    metadata=obj.get("metadata", {})
                ))
            
            logger.debug("Weaviate similarity search completed",
                        class_name=index_name,
                        results_count=len(search_results),
                        top_k=top_k)
            
            return search_results
            
        except Exception as e:
            logger.error("Failed to search Weaviate",
                        class_name=index_name,
                        error=str(e))
            return []
    
    def _build_weaviate_filter(self, filter_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build Weaviate where filter from metadata"""
        # Simple implementation - extend as needed
        if not filter_metadata:
            return None
        
        # Example: {"model_type": "teacher"} -> where filter
        conditions = []
        for key, value in filter_metadata.items():
            conditions.append({
                "path": ["metadata", key],
                "operator": "Equal",
                "valueText": str(value)
            })
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {
                "operator": "And",
                "operands": conditions
            }
        
        return None
    
    async def delete_vectors(self, index_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors from Weaviate"""
        try:
            if not self.connected:
                return False
            
            for vec_id in vector_ids:
                self.client.data_object.delete(uuid=vec_id, class_name=index_name)
            
            logger.debug("Vectors deleted from Weaviate",
                        class_name=index_name,
                        count=len(vector_ids))
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete vectors from Weaviate",
                        class_name=index_name,
                        error=str(e))
            return False
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get Weaviate class statistics"""
        try:
            if not self.connected:
                return {}
            
            # Get object count
            result = (
                self.client.query
                .aggregate(index_name)
                .with_meta_count()
                .do()
            )
            
            count = result.get("data", {}).get("Aggregate", {}).get(index_name, [{}])[0].get("meta", {}).get("count", 0)
            
            return {
                "total_vector_count": count,
                "class_name": index_name
            }
            
        except Exception as e:
            logger.error("Failed to get Weaviate class stats",
                        class_name=index_name,
                        error=str(e))
            return {}
    
    async def health_check(self) -> bool:
        """Check Weaviate health"""
        try:
            if not self.connected:
                return False
            
            # Get schema as health check
            self.client.schema.get()
            return True
            
        except Exception as e:
            logger.error("Weaviate health check failed", error=str(e))
            return False
    
    async def cleanup(self):
        """Clean up Weaviate connections"""
        try:
            self.connected = False
            self.client = None
            logger.info("Weaviate cleanup completed")
        except Exception as e:
            logger.error("Error during Weaviate cleanup", error=str(e))


class ChromaVectorDB(BaseVectorDB):
    """
    Chroma vector database implementation
    
    🎨 CHROMA INTEGRATION:
    Lightweight, open-source vector database perfect for
    development and smaller-scale deployments
    """
    
    def __init__(self, **config):
        super().__init__(VectorProvider.CHROMA, **config)
        self.persist_directory = config.get("persist_directory", "./chroma_db")
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 8000)
    
    async def initialize(self):
        """Initialize Chroma client"""
        try:
            import chromadb
            
            # Use persistent client for data persistence
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.connected = True
            
            logger.info("Chroma client initialized successfully",
                       persist_directory=self.persist_directory)
            
        except ImportError:
            raise RuntimeError("Chroma client not installed. Run: pip install chromadb")
        except Exception as e:
            logger.error("Failed to initialize Chroma", error=str(e))
            raise
    
    async def create_index(self, index_name: str, dimension: int, metric: DistanceMetric = DistanceMetric.COSINE) -> bool:
        """Create Chroma collection (index)"""
        try:
            if not self.connected:
                return False
            
            # Map distance metric
            chroma_metrics = {
                DistanceMetric.COSINE: "cosine",
                DistanceMetric.EUCLIDEAN: "l2",
                DistanceMetric.DOT_PRODUCT: "ip"  # inner product
            }
            
            distance_function = chroma_metrics.get(metric, "cosine")
            
            try:
                # Try to get existing collection
                collection = self.client.get_collection(name=index_name)
                logger.info("Chroma collection already exists", collection_name=index_name)
                return True
            except Exception:
                # Collection doesn't exist, create it
                collection = self.client.create_collection(
                    name=index_name,
                    metadata={"hnsw:space": distance_function}
                )
                
                logger.info("Chroma collection created",
                           collection_name=index_name,
                           dimension=dimension,
                           distance_function=distance_function)
                
                return True
            
        except Exception as e:
            logger.error("Failed to create Chroma collection",
                        collection_name=index_name,
                        error=str(e))
            return False
    
    async def upsert_vectors(self, index_name: str, vectors: EmbeddingBatch) -> bool:
        """Upsert vectors to Chroma collection"""
        try:
            if not self.connected:
                return False
            
            collection = self.client.get_collection(name=index_name)
            
            # Prepare documents and metadata for Chroma
            documents = []
            metadatas = []
            
            for metadata in vectors.metadata:
                documents.append(metadata.get("content", ""))
                # Chroma requires flat metadata (no nested objects)
                flat_metadata = {}
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        flat_metadata[k] = v
                    else:
                        flat_metadata[k] = json.dumps(v)
                metadatas.append(flat_metadata)
            
            collection.upsert(
                ids=vectors.ids,
                embeddings=vectors.vectors,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.debug("Vectors upserted to Chroma",
                        collection_name=index_name,
                        count=len(vectors.ids))
            
            return True
            
        except Exception as e:
            logger.error("Failed to upsert vectors to Chroma",
                        collection_name=index_name,
                        error=str(e))
            return False
    
    async def search_similar(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """Search similar vectors in Chroma"""
        try:
            if not self.connected:
                return []
            
            collection = self.client.get_collection(name=index_name)
            
            # Build where clause for filtering
            where_clause = None
            if filter_metadata:
                where_clause = {}
                for key, value in filter_metadata.items():
                    where_clause[key] = {"$eq": value}
            
            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "distances", "documents"]
            )
            
            # Convert to standard format
            search_results = []
            if results["ids"] and len(results["ids"]) > 0:
                ids = results["ids"][0]
                distances = results["distances"][0] if "distances" in results else []
                metadatas = results["metadatas"][0] if "metadatas" in results else []
                
                for i, vec_id in enumerate(ids):
                    # Convert distance to similarity score (Chroma returns distances)
                    distance = distances[i] if i < len(distances) else 1.0
                    score = 1.0 / (1.0 + distance)  # Simple conversion
                    
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    search_results.append(VectorSearchResult(
                        id=vec_id,
                        score=score,
                        metadata=metadata
                    ))
            
            logger.debug("Chroma similarity search completed",
                        collection_name=index_name,
                        results_count=len(search_results),
                        top_k=top_k)
            
            return search_results
            
        except Exception as e:
            logger.error("Failed to search Chroma",
                        collection_name=index_name,
                        error=str(e))
            return []
    
    async def delete_vectors(self, index_name: str, vector_ids: List[str]) -> bool:
        """Delete vectors from Chroma collection"""
        try:
            if not self.connected:
                return False
            
            collection = self.client.get_collection(name=index_name)
            collection.delete(ids=vector_ids)
            
            logger.debug("Vectors deleted from Chroma",
                        collection_name=index_name,
                        count=len(vector_ids))
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete vectors from Chroma",
                        collection_name=index_name,
                        error=str(e))
            return False
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get Chroma collection statistics"""
        try:
            if not self.connected:
                return {}
            
            collection = self.client.get_collection(name=index_name)
            count = collection.count()
            
            return {
                "total_vector_count": count,
                "collection_name": index_name
            }
            
        except Exception as e:
            logger.error("Failed to get Chroma collection stats",
                        collection_name=index_name,
                        error=str(e))
            return {}
    
    async def health_check(self) -> bool:
        """Check Chroma health"""
        try:
            if not self.connected:
                return False
            
            # List collections as health check
            self.client.list_collections()
            return True
            
        except Exception as e:
            logger.error("Chroma health check failed", error=str(e))
            return False
    
    async def cleanup(self):
        """Clean up Chroma connections"""
        try:
            self.connected = False
            self.client = None
            logger.info("Chroma cleanup completed")
        except Exception as e:
            logger.error("Error during Chroma cleanup", error=str(e))


# === Vector Database Manager ===

class VectorDBManager:
    """
    Centralized vector database manager for PRSM
    
    🎯 PURPOSE: Manages multiple vector database providers with
    intelligent routing, fallback strategies, and unified operations
    """
    
    def __init__(self):
        self.providers: Dict[VectorProvider, BaseVectorDB] = {}
        self.primary_provider = None
        self.fallback_providers = []
        
        # Standard PRSM indexes
        self.indexes = {
            "models": {"dimension": 1536, "metric": DistanceMetric.COSINE},
            "content": {"dimension": 1536, "metric": DistanceMetric.COSINE},
            "queries": {"dimension": 1536, "metric": DistanceMetric.COSINE}
        }
    
    async def initialize(self):
        """
        Initialize vector database providers
        
        🚀 INITIALIZATION SEQUENCE:
        1. Detect available providers based on configuration
        2. Initialize connections to each provider
        3. Set up primary and fallback strategies
        4. Create standard PRSM indexes
        """
        try:
            # Initialize Pinecone if configured
            if settings.pinecone_api_key and settings.pinecone_environment:
                pinecone_db = PineconeVectorDB(
                    api_key=settings.pinecone_api_key,
                    environment=settings.pinecone_environment
                )
                await pinecone_db.initialize()
                self.providers[VectorProvider.PINECONE] = pinecone_db
                if not self.primary_provider:
                    self.primary_provider = VectorProvider.PINECONE
                logger.info("Pinecone provider initialized")
            
            # Initialize Weaviate if configured
            if settings.weaviate_url:
                weaviate_db = WeaviateVectorDB(
                    url=settings.weaviate_url,
                    api_key=settings.weaviate_api_key
                )
                await weaviate_db.initialize()
                self.providers[VectorProvider.WEAVIATE] = weaviate_db
                if not self.primary_provider:
                    self.primary_provider = VectorProvider.WEAVIATE
                logger.info("Weaviate provider initialized")
            
            # Initialize Chroma as fallback (always available)
            chroma_db = ChromaVectorDB(persist_directory="./data/chroma_db")
            await chroma_db.initialize()
            self.providers[VectorProvider.CHROMA] = chroma_db
            if not self.primary_provider:
                self.primary_provider = VectorProvider.CHROMA
            logger.info("Chroma provider initialized")
            
            # Set up fallback strategy
            self.fallback_providers = [p for p in self.providers.keys() if p != self.primary_provider]
            
            # Create standard indexes
            await self._create_standard_indexes()
            
            logger.info("Vector database manager initialized",
                       primary_provider=self.primary_provider.value,
                       total_providers=len(self.providers))
            
        except Exception as e:
            logger.error("Failed to initialize vector database manager", error=str(e))
            raise
    
    async def _create_standard_indexes(self):
        """Create standard PRSM indexes across all providers"""
        for index_name, config in self.indexes.items():
            for provider_type, provider in self.providers.items():
                try:
                    await provider.create_index(
                        index_name=index_name,
                        dimension=config["dimension"],
                        metric=config["metric"]
                    )
                    logger.debug("Index created",
                               index_name=index_name,
                               provider=provider_type.value)
                except Exception as e:
                    logger.warning("Failed to create index",
                                 index_name=index_name,
                                 provider=provider_type.value,
                                 error=str(e))
    
    async def upsert_embedding(
        self,
        index_name: str,
        vector_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        provider: Optional[VectorProvider] = None
    ) -> bool:
        """
        Upsert single embedding with automatic provider selection
        
        🎯 SMART ROUTING:
        - Uses primary provider by default
        - Automatic fallback on failures
        - Caches embeddings in Redis for fast retrieval
        """
        try:
            # Cache embedding in Redis
            model_cache = get_model_cache()
            if model_cache:
                text_content = metadata.get("content", "")
                if text_content:
                    text_hash = hashlib.sha256(text_content.encode()).hexdigest()
                    await model_cache.store_embedding(text_hash, embedding)
            
            # Prepare batch for upsert
            batch = EmbeddingBatch(
                ids=[vector_id],
                vectors=[embedding],
                metadata=[metadata]
            )
            
            # Try primary provider first
            target_provider = provider or self.primary_provider
            if target_provider in self.providers:
                success = await self.providers[target_provider].upsert_vectors(index_name, batch)
                if success:
                    logger.debug("Embedding upserted",
                               vector_id=vector_id,
                               index_name=index_name,
                               provider=target_provider.value)
                    return True
            
            # Try fallback providers
            for fallback_provider in self.fallback_providers:
                try:
                    success = await self.providers[fallback_provider].upsert_vectors(index_name, batch)
                    if success:
                        logger.info("Embedding upserted via fallback",
                                   vector_id=vector_id,
                                   provider=fallback_provider.value)
                        return True
                except Exception as e:
                    logger.warning("Fallback provider failed",
                                 provider=fallback_provider.value,
                                 error=str(e))
            
            return False
            
        except Exception as e:
            logger.error("Failed to upsert embedding",
                        vector_id=vector_id,
                        error=str(e))
            return False
    
    async def search_similar_models(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        model_type: Optional[str] = None,
        specialization: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar models using semantic similarity
        
        🔍 MODEL DISCOVERY:
        - Semantic search across model embeddings
        - Filtering by model type and specialization
        - Intelligent ranking based on multiple factors
        """
        try:
            # Build metadata filter
            filter_metadata = {}
            if model_type:
                filter_metadata["model_type"] = model_type
            if specialization:
                filter_metadata["specialization"] = specialization
            
            # Try primary provider first
            if self.primary_provider in self.providers:
                results = await self.providers[self.primary_provider].search_similar(
                    index_name="models",
                    query_vector=query_embedding,
                    top_k=top_k,
                    filter_metadata=filter_metadata if filter_metadata else None
                )
                
                if results:
                    logger.debug("Model similarity search completed",
                               provider=self.primary_provider.value,
                               results_count=len(results))
                    return results
            
            # Try fallback providers
            for fallback_provider in self.fallback_providers:
                try:
                    results = await self.providers[fallback_provider].search_similar(
                        index_name="models",
                        query_vector=query_embedding,
                        top_k=top_k,
                        filter_metadata=filter_metadata if filter_metadata else None
                    )
                    
                    if results:
                        logger.info("Model search via fallback provider",
                                   provider=fallback_provider.value,
                                   results_count=len(results))
                        return results
                except Exception as e:
                    logger.warning("Fallback search failed",
                                 provider=fallback_provider.value,
                                 error=str(e))
            
            return []
            
        except Exception as e:
            logger.error("Failed to search similar models", error=str(e))
            return []
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all vector database providers"""
        health_status = {}
        
        for provider_type, provider in self.providers.items():
            try:
                health_status[provider_type.value] = await provider.health_check()
            except Exception as e:
                health_status[provider_type.value] = False
                logger.error("Vector DB health check failed",
                           provider=provider_type.value,
                           error=str(e))
        
        return health_status
    
    async def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all providers"""
        stats = {}
        
        for provider_type, provider in self.providers.items():
            provider_stats = {}
            for index_name in self.indexes.keys():
                try:
                    index_stats = await provider.get_index_stats(index_name)
                    provider_stats[index_name] = index_stats
                except Exception as e:
                    logger.warning("Failed to get index stats",
                                 provider=provider_type.value,
                                 index_name=index_name,
                                 error=str(e))
            
            stats[provider_type.value] = provider_stats
        
        return stats
    
    async def cleanup(self):
        """Clean up all vector database connections"""
        for provider_type, provider in self.providers.items():
            try:
                await provider.cleanup()
                logger.debug("Provider cleaned up", provider=provider_type.value)
            except Exception as e:
                logger.error("Error cleaning up provider",
                           provider=provider_type.value,
                           error=str(e))
        
        self.providers.clear()
        logger.info("Vector database manager cleanup completed")


# === Global Vector Database Manager ===

vector_db_manager = VectorDBManager()


# === Helper Functions ===

async def init_vector_databases():
    """Initialize vector databases for PRSM"""
    await vector_db_manager.initialize()


async def close_vector_databases():
    """Close vector database connections"""
    await vector_db_manager.cleanup()


def get_vector_db_manager() -> VectorDBManager:
    """Get vector database manager instance"""
    return vector_db_manager


# === Embedding Generation Utilities ===

class EmbeddingGenerator:
    """
    Generate embeddings for PRSM content
    
    🧠 EMBEDDING GENERATION:
    Creates high-quality embeddings for models, content, and queries
    using OpenAI's embedding models with caching optimization
    """
    
    def __init__(self):
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimensions
        self.client = None
    
    async def initialize(self):
        """Initialize embedding client"""
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info("Embedding generator initialized", model=self.model_name)
        except ImportError:
            raise RuntimeError("OpenAI client not installed. Run: pip install openai")
        except Exception as e:
            logger.error("Failed to initialize embedding generator", error=str(e))
            raise
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text with caching"""
        try:
            if not self.client:
                await self.initialize()
            
            # Check cache first
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            model_cache = get_model_cache()
            
            if model_cache:
                cached_embedding = await model_cache.get_embedding(text_hash)
                if cached_embedding:
                    logger.debug("Embedding cache hit", text_hash=text_hash)
                    return cached_embedding
            
            # Generate new embedding
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            
            embedding = response.data[0].embedding
            
            # Cache the embedding
            if model_cache:
                await model_cache.store_embedding(text_hash, embedding)
            
            logger.debug("Embedding generated",
                        text_length=len(text),
                        embedding_dimension=len(embedding))
            
            return embedding
            
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            return None
    
    async def generate_model_embedding(self, model_metadata: Dict[str, Any]) -> Optional[List[float]]:
        """Generate embedding for model metadata"""
        try:
            # Combine relevant metadata into text
            text_parts = []
            
            if "name" in model_metadata:
                text_parts.append(f"Model: {model_metadata['name']}")
            
            if "description" in model_metadata:
                text_parts.append(f"Description: {model_metadata['description']}")
            
            if "specialization" in model_metadata:
                text_parts.append(f"Specialization: {model_metadata['specialization']}")
            
            if "capabilities" in model_metadata:
                capabilities = model_metadata["capabilities"]
                if isinstance(capabilities, list):
                    text_parts.append(f"Capabilities: {', '.join(capabilities)}")
                else:
                    text_parts.append(f"Capabilities: {capabilities}")
            
            text = " | ".join(text_parts)
            return await self.generate_embedding(text)
            
        except Exception as e:
            logger.error("Failed to generate model embedding", error=str(e))
            return None


# Global embedding generator
embedding_generator = EmbeddingGenerator()
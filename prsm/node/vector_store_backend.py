"""
Vector Store Backend for Content Economy
========================================

Provides semantic search capabilities for PRSM content.
Supports multiple backends:
- pgvector (PostgreSQL extension)
- Milvus
- Qdrant
- In-memory (for testing)

Usage:
    from prsm.node.vector_store_backend import VectorStoreBackend, VectorStoreConfig
    
    config = VectorStoreConfig(backend="pgvector")
    store = VectorStoreBackend(config)
    await store.initialize()
    
    # Index content
    await store.upsert("QmCID...", embedding, metadata)
    
    # Search
    results = await store.search(query_embedding, limit=10)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────────────

class VectorBackend(str, Enum):
    """Supported vector store backends."""
    PGVECTOR = "pgvector"
    MILVUS = "milvus"
    QDRANT = "qdrant"
    MEMORY = "memory"  # For testing
    CHROMA = "chroma"  # Lightweight option


@dataclass
class VectorStoreConfig:
    """Configuration for vector store backend."""
    backend: str = "memory"
    
    # PostgreSQL/pgvector
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "prsm"
    postgres_user: str = "prsm"
    postgres_password: str = ""
    
    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "prsm_content"
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "prsm_content"
    
    # Chroma
    chroma_persist_dir: str = "~/.prsm/chroma"
    
    # General
    embedding_dimension: int = 1536  # OpenAI ada-002 default
    similarity_metric: str = "cosine"  # cosine, euclidean, dot
    max_results: int = 100
    
    # Content Economy integration
    default_max_royalty_rate: float = 0.10


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class ContentEmbedding:
    """Represents a content embedding with metadata."""
    content_cid: str
    embedding: np.ndarray
    creator_id: str
    royalty_rate: float
    content_type: str
    filename: str
    indexed_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_cid": self.content_cid,
            "embedding": self.embedding.tolist(),
            "creator_id": self.creator_id,
            "royalty_rate": self.royalty_rate,
            "content_type": self.content_type,
            "filename": self.filename,
            "indexed_at": self.indexed_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """A search result with similarity score."""
    content_cid: str
    similarity_score: float
    creator_id: str
    royalty_rate: float
    content_type: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        # Ensure similarity is normalized to [0, 1]
        self.similarity_score = max(0.0, min(1.0, self.similarity_score))


# ── Abstract Backend Interface ─────────────────────────────────────────────

class VectorBackendBase(ABC):
    """Abstract base class for vector store backends."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the backend (create tables/collections)."""
        pass
    
    @abstractmethod
    async def upsert(
        self,
        content_cid: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
    ) -> bool:
        """Insert or update an embedding."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar embeddings."""
        pass
    
    @abstractmethod
    async def delete(self, content_cid: str) -> bool:
        """Delete an embedding by CID."""
        pass
    
    @abstractmethod
    async def get(self, content_cid: str) -> Optional[ContentEmbedding]:
        """Get embedding by CID."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Count total embeddings."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close connections."""
        pass


# ── In-Memory Backend (Testing) ────────────────────────────────────────────

class InMemoryBackend(VectorBackendBase):
    """Simple in-memory vector store for testing."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._store: Dict[str, ContentEmbedding] = {}
        self._initialized = False
    
    async def initialize(self) -> bool:
        self._initialized = True
        return True
    
    async def upsert(
        self,
        content_cid: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
    ) -> bool:
        self._store[content_cid] = ContentEmbedding(
            content_cid=content_cid,
            embedding=np.array(embedding),
            creator_id=metadata.get("creator_id", ""),
            royalty_rate=metadata.get("royalty_rate", 0.01),
            content_type=metadata.get("content_type", "text"),
            filename=metadata.get("filename", ""),
            indexed_at=datetime.utcnow(),
            metadata=metadata,
        )
        return True
    
    async def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        query = np.array(query_embedding)
        results = []
        
        for cid, emb in self._store.items():
            # Apply filters
            if filters:
                if "max_royalty_rate" in filters:
                    if emb.royalty_rate > filters["max_royalty_rate"]:
                        continue
                if "content_type" in filters:
                    if emb.content_type != filters["content_type"]:
                        continue
            
            # Calculate cosine similarity
            similarity = np.dot(query, emb.embedding) / (
                np.linalg.norm(query) * np.linalg.norm(emb.embedding)
            )
            
            # Apply minimum similarity filter
            if filters and "min_similarity" in filters:
                if similarity < filters["min_similarity"]:
                    continue
            
            results.append(SearchResult(
                content_cid=cid,
                similarity_score=float(similarity),
                creator_id=emb.creator_id,
                royalty_rate=emb.royalty_rate,
                content_type=emb.content_type,
                metadata=emb.metadata,
            ))
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:limit]
    
    async def delete(self, content_cid: str) -> bool:
        if content_cid in self._store:
            del self._store[content_cid]
            return True
        return False
    
    async def get(self, content_cid: str) -> Optional[ContentEmbedding]:
        return self._store.get(content_cid)
    
    async def count(self) -> int:
        return len(self._store)
    
    async def close(self) -> None:
        self._store.clear()


# ── pgvector Backend ────────────────────────────────────────────────────────

class PgvectorBackend(VectorBackendBase):
    """PostgreSQL with pgvector extension backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._pool = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize PostgreSQL connection and create tables."""
        try:
            import asyncpg
            
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                host=self.config.postgres_host,
                port=self.config.postgres_port,
                database=self.config.postgres_database,
                user=self.config.postgres_user,
                password=self.config.postgres_password,
            )
            
            # Create extension and table
            async with self._pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create content embeddings table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS content_embeddings (
                        content_cid VARCHAR(128) PRIMARY KEY,
                        embedding vector({self.config.embedding_dimension}),
                        creator_id VARCHAR(256),
                        royalty_rate REAL,
                        content_type VARCHAR(64),
                        filename VARCHAR(512),
                        indexed_at TIMESTAMP DEFAULT NOW(),
                        metadata JSONB
                    )
                """)
                
                # Create similarity search index
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS content_embedding_idx 
                    ON content_embeddings 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
                
                # Create index on creator_id for filtering
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS content_creator_idx 
                    ON content_embeddings (creator_id)
                """)
                
                # Create index on royalty_rate for filtering
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS content_royalty_idx 
                    ON content_embeddings (royalty_rate)
                """)
            
            self._initialized = True
            logger.info(f"pgvector backend initialized: {self.config.postgres_database}")
            return True
            
        except ImportError:
            logger.warning("asyncpg not installed, pgvector backend unavailable")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize pgvector: {e}")
            return False
    
    async def upsert(
        self,
        content_cid: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
    ) -> bool:
        if not self._pool:
            return False
        
        try:
            async with self._pool.acquire() as conn:
                # Convert embedding to pgvector format
                emb_str = "[" + ",".join(str(x) for x in embedding) + "]"
                
                await conn.execute("""
                    INSERT INTO content_embeddings 
                    (content_cid, embedding, creator_id, royalty_rate, content_type, filename, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (content_cid) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        creator_id = EXCLUDED.creator_id,
                        royalty_rate = EXCLUDED.royalty_rate,
                        content_type = EXCLUDED.content_type,
                        filename = EXCLUDED.filename,
                        metadata = EXCLUDED.metadata,
                        indexed_at = NOW()
                """,
                    content_cid,
                    emb_str,
                    metadata.get("creator_id", ""),
                    metadata.get("royalty_rate", 0.01),
                    metadata.get("content_type", "text"),
                    metadata.get("filename", ""),
                    metadata,
                )
                return True
                
        except Exception as e:
            logger.error(f"Failed to upsert embedding: {e}")
            return False
    
    async def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if not self._pool:
            return []
        
        try:
            async with self._pool.acquire() as conn:
                # Convert query to pgvector format
                query_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
                
                # Build query with filters
                where_clauses = []
                params = [query_str]
                param_idx = 2
                
                if filters:
                    if "max_royalty_rate" in filters:
                        where_clauses.append(f"royalty_rate <= ${param_idx}")
                        params.append(filters["max_royalty_rate"])
                        param_idx += 1
                    
                    if "content_type" in filters:
                        where_clauses.append(f"content_type = ${param_idx}")
                        params.append(filters["content_type"])
                        param_idx += 1
                    
                    if "creator_id" in filters:
                        where_clauses.append(f"creator_id = ${param_idx}")
                        params.append(filters["creator_id"])
                        param_idx += 1
                
                where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
                
                # Add limit parameter
                params.append(limit)
                
                # Cosine similarity search
                query = f"""
                    SELECT 
                        content_cid,
                        1 - (embedding <=> $1) as similarity,
                        creator_id,
                        royalty_rate,
                        content_type,
                        metadata
                    FROM content_embeddings
                    WHERE {where_sql}
                    ORDER BY embedding <=> $1
                    LIMIT ${param_idx}
                """
                
                rows = await conn.fetch(query, *params)
                
                results = []
                for row in rows:
                    similarity = float(row["similarity"])
                    
                    # Apply minimum similarity filter
                    if filters and "min_similarity" in filters:
                        if similarity < filters["min_similarity"]:
                            continue
                    
                    results.append(SearchResult(
                        content_cid=row["content_cid"],
                        similarity_score=similarity,
                        creator_id=row["creator_id"],
                        royalty_rate=float(row["royalty_rate"]),
                        content_type=row["content_type"],
                        metadata=row["metadata"] or {},
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search embeddings: {e}")
            return []
    
    async def delete(self, content_cid: str) -> bool:
        if not self._pool:
            return False
        
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM content_embeddings WHERE content_cid = $1",
                    content_cid,
                )
                return "DELETE" in result
        except Exception as e:
            logger.error(f"Failed to delete embedding: {e}")
            return False
    
    async def get(self, content_cid: str) -> Optional[ContentEmbedding]:
        if not self._pool:
            return None
        
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM content_embeddings WHERE content_cid = $1",
                    content_cid,
                )
                
                if not row:
                    return None
                
                # Parse embedding from string
                emb_str = row["embedding"]
                embedding = np.array([float(x) for x in emb_str.strip("[]").split(",")])
                
                return ContentEmbedding(
                    content_cid=row["content_cid"],
                    embedding=embedding,
                    creator_id=row["creator_id"],
                    royalty_rate=float(row["royalty_rate"]),
                    content_type=row["content_type"],
                    filename=row["filename"],
                    indexed_at=row["indexed_at"],
                    metadata=row["metadata"] or {},
                )
                
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    async def count(self) -> int:
        if not self._pool:
            return 0
        
        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT COUNT(*) FROM content_embeddings")
                return result or 0
        except Exception as e:
            logger.error(f"Failed to count embeddings: {e}")
            return 0
    
    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None


# ── Qdrant Backend ──────────────────────────────────────────────────────────

class QdrantBackend(VectorBackendBase):
    """Qdrant vector database backend."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._client = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize Qdrant client and create collection."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            from qdrant_client.http.models import Distance, VectorParams
            
            self._client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
            )
            
            # Create collection if not exists
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.config.qdrant_collection not in collection_names:
                self._client.create_collection(
                    collection_name=self.config.qdrant_collection,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )
            
            self._initialized = True
            logger.info(f"Qdrant backend initialized: {self.config.qdrant_collection}")
            return True
            
        except ImportError:
            logger.warning("qdrant-client not installed, Qdrant backend unavailable")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            return False
    
    async def upsert(
        self,
        content_cid: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
    ) -> bool:
        if not self._client:
            return False
        
        try:
            from qdrant_client.http.models import PointStruct
            
            self._client.upsert(
                collection_name=self.config.qdrant_collection,
                points=[
                    PointStruct(
                        id=content_cid,
                        vector=embedding.tolist(),
                        payload={
                            "creator_id": metadata.get("creator_id", ""),
                            "royalty_rate": metadata.get("royalty_rate", 0.01),
                            "content_type": metadata.get("content_type", "text"),
                            "filename": metadata.get("filename", ""),
                            **metadata,
                        },
                    )
                ],
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert to Qdrant: {e}")
            return False
    
    async def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if not self._client:
            return []
        
        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range
            
            # Build filter
            query_filter = None
            if filters:
                conditions = []
                
                if "max_royalty_rate" in filters:
                    conditions.append(
                        FieldCondition(
                            key="royalty_rate",
                            range=Range(lte=filters["max_royalty_rate"]),
                        )
                    )
                
                if "content_type" in filters:
                    conditions.append(
                        FieldCondition(
                            key="content_type",
                            match=MatchValue(value=filters["content_type"]),
                        )
                    )
                
                if "creator_id" in filters:
                    conditions.append(
                        FieldCondition(
                            key="creator_id",
                            match=MatchValue(value=filters["creator_id"]),
                        )
                    )
                
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Search
            results = self._client.search(
                collection_name=self.config.qdrant_collection,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=query_filter,
                score_threshold=filters.get("min_similarity", 0.0) if filters else 0.0,
            )
            
            return [
                SearchResult(
                    content_cid=r.id,
                    similarity_score=r.score,
                    creator_id=r.payload.get("creator_id", ""),
                    royalty_rate=r.payload.get("royalty_rate", 0.01),
                    content_type=r.payload.get("content_type", "text"),
                    metadata=r.payload,
                )
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to search Qdrant: {e}")
            return []
    
    async def delete(self, content_cid: str) -> bool:
        if not self._client:
            return False
        
        try:
            self._client.delete(
                collection_name=self.config.qdrant_collection,
                points_selector=[content_cid],
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Qdrant: {e}")
            return False
    
    async def get(self, content_cid: str) -> Optional[ContentEmbedding]:
        if not self._client:
            return None
        
        try:
            results = self._client.retrieve(
                collection_name=self.config.qdrant_collection,
                ids=[content_cid],
                with_vectors=True,
            )
            
            if not results:
                return None
            
            r = results[0]
            return ContentEmbedding(
                content_cid=r.id,
                embedding=np.array(r.vector),
                creator_id=r.payload.get("creator_id", ""),
                royalty_rate=r.payload.get("royalty_rate", 0.01),
                content_type=r.payload.get("content_type", "text"),
                filename=r.payload.get("filename", ""),
                indexed_at=datetime.utcnow(),  # Qdrant doesn't store this directly
                metadata=r.payload,
            )
            
        except Exception as e:
            logger.error(f"Failed to get from Qdrant: {e}")
            return None
    
    async def count(self) -> int:
        if not self._client:
            return 0
        
        try:
            result = self._client.get_collection(self.config.qdrant_collection)
            return result.points_count or 0
        except Exception as e:
            logger.error(f"Failed to count Qdrant: {e}")
            return 0
    
    async def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None


# ── Main Vector Store Backend ──────────────────────────────────────────────

class VectorStoreBackend:
    """Main vector store interface that wraps backend implementations."""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self._backend: Optional[VectorBackendBase] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the configured backend."""
        if self._initialized:
            return True
        
        # Create backend based on configuration
        backend_type = self.config.backend.lower()
        
        if backend_type == VectorBackend.MEMORY.value:
            self._backend = InMemoryBackend(self.config)
        elif backend_type == VectorBackend.PGVECTOR.value:
            self._backend = PgvectorBackend(self.config)
        elif backend_type == VectorBackend.QDRANT.value:
            self._backend = QdrantBackend(self.config)
        else:
            logger.warning(f"Unknown backend '{backend_type}', using in-memory")
            self._backend = InMemoryBackend(self.config)
        
        success = await self._backend.initialize()
        self._initialized = success
        return success
    
    async def upsert(
        self,
        content_cid: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Insert or update a content embedding."""
        if not self._initialized or not self._backend:
            return False
        
        return await self._backend.upsert(content_cid, embedding, metadata or {})
    
    async def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar content embeddings."""
        if not self._initialized or not self._backend:
            return []
        
        return await self._backend.search(query_embedding, limit, filters)
    
    async def delete(self, content_cid: str) -> bool:
        """Delete a content embedding."""
        if not self._initialized or not self._backend:
            return False
        
        return await self._backend.delete(content_cid)
    
    async def get(self, content_cid: str) -> Optional[ContentEmbedding]:
        """Get a content embedding by CID."""
        if not self._initialized or not self._backend:
            return None
        
        return await self._backend.get(content_cid)
    
    async def count(self) -> int:
        """Count total embeddings."""
        if not self._initialized or not self._backend:
            return 0
        
        return await self._backend.count()
    
    async def close(self) -> None:
        """Close the backend connection."""
        if self._backend:
            await self._backend.close()
            self._backend = None
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def backend_name(self) -> str:
        return self.config.backend


# ── Factory Function ────────────────────────────────────────────────────────

def create_vector_store(
    backend: str = "memory",
    **kwargs,
) -> VectorStoreBackend:
    """Factory function to create a vector store with specified backend.
    
    Args:
        backend: Backend type ("memory", "pgvector", "qdrant", "milvus")
        **kwargs: Additional configuration options
        
    Returns:
        VectorStoreBackend instance
    """
    config = VectorStoreConfig(backend=backend, **kwargs)
    return VectorStoreBackend(config)

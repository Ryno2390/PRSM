"""
Factory function for creating vector store instances

Provides a unified way to create different vector store implementations
based on configuration, with graceful handling of missing dependencies.
"""

from typing import Type
from ..base import PRSMVectorStore, VectorStoreConfig, VectorStoreType


def create_vector_store(config: VectorStoreConfig) -> PRSMVectorStore:
    """
    Create a vector store instance based on configuration
    
    Args:
        config: Vector store configuration specifying type and connection details
        
    Returns:
        Initialized vector store instance
        
    Raises:
        ValueError: If store type is not supported
        ImportError: If required dependencies are not installed
    """
    
    if config.store_type == VectorStoreType.PGVECTOR:
        from .pgvector_store import PgVectorStore
        return PgVectorStore(config)
    
    elif config.store_type == VectorStoreType.MILVUS:
        try:
            from .milvus_store import MilvusVectorStore
            return MilvusVectorStore(config)
        except ImportError:
            raise ImportError(
                "Milvus support requires additional dependencies. Install with:\n"
                "pip install pymilvus"
            )
    
    elif config.store_type == VectorStoreType.QDRANT:
        try:
            from .qdrant_store import QdrantVectorStore
            return QdrantVectorStore(config)
        except ImportError:
            raise ImportError(
                "Qdrant support requires additional dependencies. Install with:\n"
                "pip install qdrant-client"
            )
    
    elif config.store_type == VectorStoreType.PINECONE:
        try:
            from .pinecone_store import PineconeVectorStore
            return PineconeVectorStore(config)
        except ImportError:
            raise ImportError(
                "Pinecone support requires additional dependencies. Install with:\n"
                "pip install pinecone-client"
            )
    
    else:
        raise ValueError(f"Unsupported vector store type: {config.store_type}")


def get_available_store_types() -> list[VectorStoreType]:
    """
    Get list of vector store types that can be created (have dependencies installed)
    
    Returns:
        List of available VectorStoreType values
    """
    available = []
    
    # pgvector is always available (only requires asyncpg/psycopg2)
    available.append(VectorStoreType.PGVECTOR)
    
    # Check optional dependencies
    try:
        import pymilvus
        available.append(VectorStoreType.MILVUS)
    except ImportError:
        pass
    
    try:
        import qdrant_client
        available.append(VectorStoreType.QDRANT)
    except ImportError:
        pass
    
    try:
        import pinecone
        available.append(VectorStoreType.PINECONE)
    except ImportError:
        pass
    
    return available


def create_development_store(database_url: str = None) -> PRSMVectorStore:
    """
    Create a vector store optimized for development
    
    Automatically selects the best available option:
    1. pgvector (if PostgreSQL available)
    2. Falls back to in-memory options if available
    
    Args:
        database_url: Optional PostgreSQL connection string
        
    Returns:
        Ready-to-use vector store for development
    """
    
    # Try pgvector first (best for development)
    if VectorStoreType.PGVECTOR in get_available_store_types():
        if database_url:
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            
            config = VectorStoreConfig(
                store_type=VectorStoreType.PGVECTOR,
                host=parsed.hostname or "localhost",
                port=parsed.port or 5432,
                database=parsed.path[1:] if parsed.path else "prsm_dev",
                username=parsed.username or "postgres",
                password=parsed.password or "postgres",
                collection_name="dev_vectors"
            )
        else:
            # Default development configuration
            config = VectorStoreConfig(
                store_type=VectorStoreType.PGVECTOR,
                host="localhost",
                port=5432,
                database="prsm_dev",
                username="postgres",
                password="postgres",
                collection_name="dev_vectors"
            )
        
        return create_vector_store(config)
    
    else:
        raise RuntimeError(
            "No suitable vector store available for development.\n"
            "Please install PostgreSQL with pgvector extension, or install:\n"
            "pip install asyncpg psycopg2-binary"
        )
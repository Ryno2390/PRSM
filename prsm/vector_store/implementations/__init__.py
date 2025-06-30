"""
Vector Store Implementations

Concrete implementations of the PRSM vector store interface for different
database backends.
"""

from .pgvector_store import PgVectorStore
from .factory import create_vector_store

# Stub classes for missing implementations (will be implemented in Phase 1B)
class MilvusVectorStore:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Milvus implementation coming in Phase 1B")

class QdrantVectorStore:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Qdrant implementation coming in Phase 1B")

class PineconeVectorStore:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Pinecone implementation coming in Phase 1A")

# Lazy imports for optional dependencies
def get_milvus_store():
    """Lazy import for Milvus (requires pymilvus)"""
    return MilvusVectorStore

def get_qdrant_store():
    """Lazy import for Qdrant (requires qdrant-client)"""
    return QdrantVectorStore

def get_pinecone_store():
    """Lazy import for Pinecone (requires pinecone-client)"""
    return PineconeVectorStore

__all__ = [
    "PgVectorStore",
    "MilvusVectorStore",
    "QdrantVectorStore",
    "PineconeVectorStore",
    "create_vector_store",
    "get_milvus_store",
    "get_qdrant_store", 
    "get_pinecone_store"
]
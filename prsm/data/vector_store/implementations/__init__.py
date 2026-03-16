"""
Vector Store Implementations

Concrete implementations of the PRSM vector store interface.
Each backend is lazily imported so missing optional dependencies
don't break the import of other backends.
"""

from .pgvector_store import PgVectorStore
from .factory import create_vector_store, get_available_store_types


def _get_milvus():
    from .milvus_store import MilvusVectorStore
    return MilvusVectorStore


def _get_qdrant():
    from .qdrant_store import QdrantVectorStore
    return QdrantVectorStore


def _get_pinecone():
    from .pinecone_store import PineconeVectorStore
    return PineconeVectorStore


# Convenience aliases — these trigger the lazy import at access time
class MilvusVectorStore:
    """Proxy that loads the real implementation only when instantiated."""
    def __new__(cls, *args, **kwargs):
        return _get_milvus()(*args, **kwargs)


class QdrantVectorStore:
    """Proxy that loads the real implementation only when instantiated."""
    def __new__(cls, *args, **kwargs):
        return _get_qdrant()(*args, **kwargs)


class PineconeVectorStore:
    """Proxy that loads the real implementation only when instantiated."""
    def __new__(cls, *args, **kwargs):
        return _get_pinecone()(*args, **kwargs)


# Legacy helper functions (kept for backwards compatibility)
def get_milvus_store():
    return _get_milvus()


def get_qdrant_store():
    return _get_qdrant()


def get_pinecone_store():
    return _get_pinecone()


__all__ = [
    "PgVectorStore",
    "MilvusVectorStore",
    "QdrantVectorStore",
    "PineconeVectorStore",
    "create_vector_store",
    "get_available_store_types",
    "get_milvus_store",
    "get_qdrant_store",
    "get_pinecone_store",
]

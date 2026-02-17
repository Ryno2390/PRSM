"""
PRSM Vector Store Module

Provides unified interface for vector database operations with built-in
provenance tracking and FTNS royalty integration.

Supports multiple vector database backends:
- pgvector (PostgreSQL extension) - Development and initial production
- Milvus - Large-scale distributed deployment  
- Qdrant - High-performance edge nodes

Key Features:
- Seamless migration between vector database backends
- Automatic provenance tracking for creator royalties
- IPFS content addressing integration
- Multi-modal embedding support (text, image, audio)
- Performance monitoring and optimization
"""

from .base import PRSMVectorStore, ContentMatch, VectorStoreConfig, VectorStoreType
from .coordinator import VectorStoreCoordinator
from .implementations import (
    PgVectorStore,
    MilvusVectorStore, 
    QdrantVectorStore
)
from .provenance import ProvenanceVectorIntegration
from .migration import VectorStoreMigrator

__all__ = [
    "PRSMVectorStore",
    "ContentMatch",
    "VectorStoreConfig",
    "VectorStoreType",
    "VectorStoreCoordinator",
    "PgVectorStore",
    "MilvusVectorStore",
    "QdrantVectorStore",
    "ProvenanceVectorIntegration",
    "VectorStoreMigrator"
]
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

from .base import (
    PRSMVectorStore, ContentMatch, VectorStoreConfig, VectorStoreType,
    ContentType, SearchFilters, VectorStoreBenchmark
)
from .coordinator import VectorStoreCoordinator, MigrationPhase
from .implementations import (
    PgVectorStore,
    MilvusVectorStore,
    QdrantVectorStore
)
from .implementations.pgvector_store import (
    create_development_pgvector_store,
    get_docker_connection_info
)
from .provenance import ProvenanceVectorIntegration
from .migration import VectorStoreMigrator

__all__ = [
    "PRSMVectorStore",
    "ContentMatch",
    "VectorStoreConfig",
    "VectorStoreType",
    "ContentType",
    "SearchFilters",
    "VectorStoreBenchmark",
    "VectorStoreCoordinator",
    "MigrationPhase",
    "PgVectorStore",
    "MilvusVectorStore",
    "QdrantVectorStore",
    "create_development_pgvector_store",
    "get_docker_connection_info",
    "ProvenanceVectorIntegration",
    "VectorStoreMigrator"
]
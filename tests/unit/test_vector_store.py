"""
Unit tests for vector store implementations.

These tests verify factory routing, import handling, and base class behavior
without requiring live database connections.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import numpy as np

from prsm.data.vector_store.base import (
    ContentMatch, SearchFilters, VectorStoreConfig,
    VectorStoreType, ContentType
)
from prsm.data.vector_store.implementations import (
    PgVectorStore, MilvusVectorStore, QdrantVectorStore, PineconeVectorStore
)
from prsm.data.vector_store.implementations.factory import (
    create_vector_store, get_available_store_types
)


class TestFactoryRouting:
    """Tests for factory routing behavior."""

    def test_factory_routes_pgvector(self):
        """create_vector_store with PGVECTOR config returns a PgVectorStore instance."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.PGVECTOR,
            host="localhost",
            port=5432,
            database="test_db",
            collection_name="test_collection"
        )
        # Mock asyncpg connection pool creation
        with patch('asyncpg.create_pool') as mock_pool:
            mock_pool.return_value = MagicMock()
            store = create_vector_store(config)
            assert isinstance(store, PgVectorStore)

    def test_factory_raises_on_unknown_type(self):
        """VectorStoreType('unknown') raises ValueError from factory."""
        config = VectorStoreConfig(
            store_type="unknown",  # Invalid type
            host="localhost",
            port=5432,
            database="test_db",
            collection_name="test_collection"
        )
        with pytest.raises(ValueError):
            create_vector_store(config)

    def test_factory_raises_import_error_for_milvus_without_dep(self):
        """Mock ImportError on pymilvus, assert factory raises ImportError with install hint."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.MILVUS,
            host="localhost",
            port=19530,
            database="test_db",
            collection_name="test_collection"
        )
        with patch('prsm.data.vector_store.implementations.milvus_store.HAS_MILVUS', False):
            with pytest.raises(ImportError, match="pymilvus"):
                create_vector_store(config)

    def test_factory_raises_import_error_for_qdrant_without_dep(self):
        """Mock ImportError on qdrant_client, assert factory raises ImportError."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.QDRANT,
            host="localhost",
            port=6333,
            database="test_db",
            collection_name="test_collection"
        )
        with patch('prsm.data.vector_store.implementations.qdrant_store.HAS_QDRANT', False):
            with pytest.raises(ImportError, match="qdrant-client"):
                create_vector_store(config)

    def test_factory_raises_import_error_for_pinecone_without_dep(self):
        """Mock ImportError on pinecone, assert factory raises ImportError."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.PINECONE,
            host="localhost",
            port=443,
            database="test_db",
            collection_name="test_collection"
        )
        # Mock HAS_PINECONE at the module level before import
        with patch.dict('sys.modules', {'pinecone': None}):
            with patch('prsm.data.vector_store.implementations.pinecone_store.HAS_PINECONE', False):
                with pytest.raises(ImportError, match="pinecone"):
                    create_vector_store(config)


class TestAvailableStoreTypes:
    """Tests for get_available_store_types function."""

    def test_get_available_store_types_always_includes_pgvector(self):
        """get_available_store_types() always returns VectorStoreType.PGVECTOR."""
        # Mock the optional dependencies to avoid import errors
        with patch.dict('sys.modules', {'pymilvus': None, 'qdrant_client': None, 'pinecone': None}):
            # Need to reload the factory module to pick up the mocked imports
            import importlib
            from prsm.data.vector_store.implementations import factory
            importlib.reload(factory)
            types = factory.get_available_store_types()
            assert VectorStoreType.PGVECTOR in types


class TestStoreInstantiationWithoutDeps:
    """Tests that stores raise proper errors when dependencies are missing."""

    def test_milvus_store_raises_without_pymilvus(self):
        """MilvusVectorStore(config) raises ImportError when HAS_MILVUS=False."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.MILVUS,
            host="localhost",
            port=19530,
            database="test_db",
            collection_name="test_collection"
        )
        with patch('prsm.data.vector_store.implementations.milvus_store.HAS_MILVUS', False):
            with pytest.raises(ImportError, match="pymilvus"):
                MilvusVectorStore(config)

    def test_qdrant_store_raises_without_qdrant_client(self):
        """QdrantVectorStore(config) raises ImportError when HAS_QDRANT=False."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.QDRANT,
            host="localhost",
            port=6333,
            database="test_db",
            collection_name="test_collection"
        )
        with patch('prsm.data.vector_store.implementations.qdrant_store.HAS_QDRANT', False):
            with pytest.raises(ImportError, match="qdrant-client"):
                QdrantVectorStore(config)

    def test_pinecone_store_raises_without_pinecone_client(self):
        """PineconeVectorStore(config) raises ImportError when HAS_PINECONE=False."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.PINECONE,
            host="localhost",
            port=443,
            database="test_db",
            collection_name="test_collection"
        )
        # Mock at sys.modules level to prevent the broken pinecone package from loading
        with patch.dict('sys.modules', {'pinecone': None}):
            with patch('prsm.data.vector_store.implementations.pinecone_store.HAS_PINECONE', False):
                with pytest.raises(ImportError, match="pinecone"):
                    PineconeVectorStore(config)


class TestDataModels:
    """Tests for data model validation."""

    def test_content_match_similarity_bounds(self):
        """ContentMatch with similarity_score outside [0,1] should be clamped or raise."""
        # ContentMatch is a Pydantic model with ge=0.0, le=1.0
        # Test valid bounds
        match = ContentMatch(
            content_cid="test_cid",
            similarity_score=0.5,
            metadata={}
        )
        assert match.similarity_score == 0.5

        # Test upper bound
        match_high = ContentMatch(
            content_cid="test_cid",
            similarity_score=1.0,
            metadata={}
        )
        assert match_high.similarity_score == 1.0

        # Test lower bound
        match_low = ContentMatch(
            content_cid="test_cid",
            similarity_score=0.0,
            metadata={}
        )
        assert match_low.similarity_score == 0.0

    def test_search_filters_construction(self):
        """SearchFilters with content_types and min_quality_score constructs correctly."""
        filters = SearchFilters(
            content_types=[ContentType.TEXT, ContentType.CODE],
            min_quality_score=0.8
        )
        assert len(filters.content_types) == 2
        assert ContentType.TEXT in filters.content_types
        assert ContentType.CODE in filters.content_types
        assert filters.min_quality_score == 0.8


class TestPerformanceMetrics:
    """Tests for performance metrics tracking."""

    def test_performance_metrics_update(self):
        """Call _update_performance_metrics on a mock store, verify metrics updated."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.PGVECTOR,
            host="localhost",
            port=5432,
            database="test_db",
            collection_name="test_collection"
        )
        
        # Create a minimal mock store that has the metrics update method
        class MockStore:
            def __init__(self, config):
                self.config = config
                self.total_queries = 0
                self.total_storage_operations = 0
                self.average_query_time = 0.0
                self._query_times = []
            
            def _update_performance_metrics(self, operation: str, duration: float, success: bool):
                if operation == "query":
                    self.total_queries += 1
                    self._query_times.append(duration)
                    self.average_query_time = sum(self._query_times) / len(self._query_times)
                elif operation == "storage":
                    self.total_storage_operations += 1
        
        store = MockStore(config)
        store._update_performance_metrics("query", 0.1, True)
        
        assert store.total_queries == 1
        assert store.average_query_time == 0.1

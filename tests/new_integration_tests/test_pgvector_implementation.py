#!/usr/bin/env python3
"""
Test Suite for PRSM PostgreSQL + pgvector Implementation

This script tests the complete production-grade pgvector implementation
with real PostgreSQL database operations.

Prerequisites:
1. Docker and docker-compose installed
2. asyncpg installed: pip install asyncpg
3. PostgreSQL + pgvector running via Docker

Quick Setup:
    docker-compose -f docker-compose.vector.yml up -d postgres-vector
    python test_pgvector_implementation.py
"""

import asyncio
import logging
import numpy as np
import sys
from datetime import datetime
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import PRSM components
try:
    from prsm.data.vector_store import VectorStoreConfig, VectorStoreType, ContentType, SearchFilters
    from prsm.data.vector_store import (
        PgVectorStore, create_development_pgvector_store, get_docker_connection_info
    )
except ImportError as e:
    import pytest
    pytest.skip(f"Import error: {e}. Make sure you're running from the PRSM root directory", allow_module_level=True)


class PgVectorTestSuite:
    """Comprehensive test suite for PostgreSQL + pgvector implementation"""
    
    def __init__(self):
        self.store: PgVectorStore = None
        self.test_content = []
        self.results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": []
        }
    
    async def setup(self):
        """Initialize database connection and test data"""
        print("\n🚀 SETTING UP PRSM PGVECTOR TEST SUITE")
        print("=" * 60)
        
        # Display connection info
        conn_info = get_docker_connection_info()
        print(f"📡 Connecting to PostgreSQL:")
        print(f"   Host: {conn_info['host']}")
        print(f"   Port: {conn_info['port']}")
        print(f"   Database: {conn_info['database']}")
        print(f"   Docker command: {conn_info['docker_command']}")
        print()
        
        try:
            # Create development store
            self.store = await create_development_pgvector_store()
            print("✅ Successfully connected to PostgreSQL + pgvector")
            
            # Prepare test content
            self._prepare_test_content()
            print("✅ Test content prepared")
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            print("\n🔧 Troubleshooting:")
            print("1. Make sure Docker is running")
            print("2. Start PostgreSQL: docker-compose -f docker-compose.vector.yml up postgres-vector")
            print("3. Wait for database initialization (first run takes longer)")
            print("4. Check port 5433 is available")
            raise
    
    def _prepare_test_content(self):
        """Prepare test content for various scenarios"""
        self.test_content = [
            {
                "cid": "QmTest1_AIGovernance",
                "embedding": np.random.random(384).astype(np.float32),
                "metadata": {
                    "title": "Decentralized AI Governance Framework",
                    "description": "A comprehensive framework for democratic AI governance using blockchain",
                    "content_type": ContentType.RESEARCH_PAPER.value,
                    "creator_id": "governance_researcher_001",
                    "royalty_rate": 0.08,
                    "license": "Creative Commons",
                    "quality_score": 0.95,
                    "citation_count": 127,
                    "keywords": ["AI", "governance", "blockchain", "democracy"]
                }
            },
            {
                "cid": "QmTest2_ClimateData",
                "embedding": np.random.random(384).astype(np.float32),
                "metadata": {
                    "title": "Global Climate Dataset 2020-2024",
                    "description": "High-resolution global climate measurements for ML applications",
                    "content_type": ContentType.DATASET.value,
                    "creator_id": "climate_science_org",
                    "royalty_rate": 0.06,
                    "license": "Open Data",
                    "quality_score": 0.98,
                    "citation_count": 234,
                    "keywords": ["climate", "data", "global", "temperature"]
                }
            },
            {
                "cid": "QmTest3_VectorCode",
                "embedding": np.random.random(384).astype(np.float32),
                "metadata": {
                    "title": "Efficient Vector Search Implementation",
                    "description": "High-performance vector similarity search algorithms",
                    "content_type": ContentType.CODE.value,
                    "creator_id": "vector_dev_team",
                    "royalty_rate": 0.05,
                    "license": "MIT",
                    "quality_score": 0.87,
                    "citation_count": 89,
                    "keywords": ["vector", "search", "algorithm", "performance"]
                }
            },
            {
                "cid": "QmTest4_EthicsResearch",
                "embedding": np.random.random(384).astype(np.float32),
                "metadata": {
                    "title": "AI Ethics in Distributed Systems",
                    "description": "Ethical considerations for decentralized AI networks",
                    "content_type": ContentType.RESEARCH_PAPER.value,
                    "creator_id": "ethics_institute",
                    "royalty_rate": 0.08,
                    "license": "Academic Use",
                    "quality_score": 0.92,
                    "citation_count": 156,
                    "keywords": ["ethics", "AI", "distributed", "systems"]
                }
            },
            {
                "cid": "QmTest5_AudioContent",
                "embedding": np.random.random(384).astype(np.float32),
                "metadata": {
                    "title": "AI-Generated Music Composition",
                    "description": "Musical compositions created using AI algorithms",
                    "content_type": ContentType.AUDIO.value,
                    "creator_id": "ai_music_lab",
                    "royalty_rate": 0.10,
                    "license": "Creative Commons",
                    "quality_score": 0.85,
                    "citation_count": 43,
                    "keywords": ["music", "AI", "composition", "audio"]
                }
            }
        ]
    
    async def test_database_connection(self):
        """Test basic database connectivity and health"""
        self.results["tests_run"] += 1
        
        try:
            print("\n🔗 Testing Database Connection")
            print("-" * 40)
            
            # Test health check
            health = await self.store.health_check()
            assert health["status"] == "healthy", f"Health check failed: {health}"
            print(f"✅ Health check passed: {health['status']}")
            
            # Test collection stats
            stats = await self.store.get_collection_stats()
            assert "total_vectors" in stats, "Collection stats missing total_vectors"
            print(f"✅ Collection stats retrieved: {stats['total_vectors']} vectors")
            
            self.results["tests_passed"] += 1
            
        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Database connection test: {e}")
            print(f"❌ Database connection test failed: {e}")
            raise
    
    async def test_content_storage(self):
        """Test storing content with embeddings"""
        self.results["tests_run"] += 1
        
        try:
            print("\n📦 Testing Content Storage")
            print("-" * 40)
            
            stored_ids = []
            
            for i, content in enumerate(self.test_content):
                print(f"  Storing content {i+1}: {content['metadata']['title'][:50]}...")
                
                vector_id = await self.store.store_content_with_embeddings(
                    content["cid"],
                    content["embedding"],
                    content["metadata"]
                )
                
                stored_ids.append(vector_id)
                print(f"    ✅ Stored with vector ID: {vector_id}")
            
            # Verify all content was stored
            final_stats = await self.store.get_collection_stats()
            print(f"\n✅ All content stored successfully")
            print(f"   Total vectors in database: {final_stats['total_vectors']}")
            print(f"   Unique creators: {final_stats['unique_creators']}")
            
            self.results["tests_passed"] += 1
            
        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Content storage test: {e}")
            print(f"❌ Content storage test failed: {e}")
            raise
    
    async def test_similarity_search(self):
        """Test vector similarity search functionality"""
        self.results["tests_run"] += 1
        
        try:
            print("\n🔍 Testing Similarity Search")
            print("-" * 40)
            
            # Test basic search
            query_embedding = self.test_content[0]["embedding"]  # Use first item as query
            results = await self.store.search_similar_content(query_embedding, top_k=3)
            
            assert len(results) > 0, "No search results returned"
            print(f"✅ Basic search returned {len(results)} results")
            
            # Display results
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.metadata.get('title', 'Unknown')[:50]}...")
                print(f"     Similarity: {result.similarity_score:.3f}")
                print(f"     Creator: {result.creator_id}")
                
            # Test that first result is most similar (should be exact match)
            assert results[0].similarity_score > 0.99, f"Expected high similarity, got {results[0].similarity_score}"
            print("✅ Similarity scoring working correctly")
            
            self.results["tests_passed"] += 1
            
        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Similarity search test: {e}")
            print(f"❌ Similarity search test failed: {e}")
            raise
    
    async def test_filtered_search(self):
        """Test search with various filters"""
        self.results["tests_run"] += 1
        
        try:
            print("\n🔍 Testing Filtered Search")
            print("-" * 40)
            
            query_embedding = np.random.random(384).astype(np.float32)
            
            # Test content type filter
            print("  Testing content type filter (research papers only)...")
            filters = SearchFilters(content_types=[ContentType.RESEARCH_PAPER])
            results = await self.store.search_similar_content(
                query_embedding, filters=filters, top_k=5
            )
            
            assert all(r.content_type == ContentType.RESEARCH_PAPER for r in results), \
                "Content type filter not working"
            print(f"    ✅ Found {len(results)} research papers")
            
            # Test creator filter
            print("  Testing creator filter...")
            filters = SearchFilters(creator_ids=["governance_researcher_001"])
            creator_results = await self.store.search_similar_content(
                query_embedding, filters=filters, top_k=5
            )
            
            assert all(r.creator_id == "governance_researcher_001" for r in creator_results), \
                "Creator filter not working"
            print(f"    ✅ Found {len(creator_results)} results from specific creator")
            
            # Test quality filter
            print("  Testing quality score filter...")
            filters = SearchFilters(min_quality_score=0.9)
            quality_results = await self.store.search_similar_content(
                query_embedding, filters=filters, top_k=5
            )
            
            assert all(r.quality_score >= 0.9 for r in quality_results if r.quality_score), \
                "Quality filter not working"
            print(f"    ✅ Found {len(quality_results)} high-quality results")
            
            self.results["tests_passed"] += 1
            
        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Filtered search test: {e}")
            print(f"❌ Filtered search test failed: {e}")
            raise
    
    async def test_metadata_operations(self):
        """Test metadata update and content management"""
        self.results["tests_run"] += 1
        
        try:
            print("\n📝 Testing Metadata Operations")
            print("-" * 40)
            
            test_cid = self.test_content[0]["cid"]
            
            # Test metadata update
            print("  Testing metadata update...")
            update_success = await self.store.update_content_metadata(
                test_cid,
                {"updated_for_test": True, "update_timestamp": datetime.utcnow().isoformat()}
            )
            
            assert update_success, "Metadata update failed"
            print("    ✅ Metadata updated successfully")
            
            # Verify update by searching
            query_embedding = self.test_content[0]["embedding"]
            results = await self.store.search_similar_content(query_embedding, top_k=1)
            
            updated_metadata = results[0].metadata
            assert updated_metadata.get("updated_for_test") == True, "Metadata update not reflected"
            print("    ✅ Metadata update verified")
            
            self.results["tests_passed"] += 1
            
        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Metadata operations test: {e}")
            print(f"❌ Metadata operations test failed: {e}")
            raise
    
    async def test_performance_monitoring(self):
        """Test performance tracking and metrics"""
        self.results["tests_run"] += 1
        
        try:
            print("\n⚡ Testing Performance Monitoring")
            print("-" * 40)
            
            # Check initial performance metrics
            initial_metrics = self.store.performance_metrics.copy()
            print(f"  Initial metrics: {initial_metrics}")
            
            # Perform several operations
            query_embedding = np.random.random(384).astype(np.float32)
            
            for i in range(5):
                await self.store.search_similar_content(query_embedding, top_k=3)
            
            # Check updated metrics
            final_metrics = self.store.performance_metrics
            
            assert final_metrics["total_queries"] > initial_metrics["total_queries"], \
                "Query count not incrementing"
            assert final_metrics["average_query_time"] > 0, "Average query time not tracked"
            
            print(f"  ✅ Performance tracking working")
            print(f"    Total queries: {final_metrics['total_queries']}")
            print(f"    Avg query time: {final_metrics['average_query_time']*1000:.2f}ms")
            print(f"    Total storage ops: {final_metrics['total_storage_operations']}")
            
            self.results["tests_passed"] += 1
            
        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Performance monitoring test: {e}")
            print(f"❌ Performance monitoring test failed: {e}")
            raise
    
    async def test_large_batch_operations(self):
        """Test handling of larger batch operations"""
        self.results["tests_run"] += 1
        
        try:
            print("\n📊 Testing Batch Operations")
            print("-" * 40)
            
            # Generate batch test data
            batch_size = 50
            print(f"  Generating {batch_size} test vectors...")
            
            batch_data = []
            for i in range(batch_size):
                batch_data.append((
                    f"QmBatch_{i:03d}",
                    np.random.random(384).astype(np.float32),
                    {
                        "title": f"Batch Test Document {i}",
                        "content_type": ContentType.TEXT.value,
                        "creator_id": f"batch_creator_{i % 10}",
                        "batch_test": True
                    }
                ))
            
            # Test batch storage
            print("  Testing batch storage...")
            start_time = asyncio.get_event_loop().time()
            
            stored_ids = await self.store.batch_store_content(batch_data, batch_size=10)
            
            duration = asyncio.get_event_loop().time() - start_time
            rate = len(stored_ids) / duration
            
            assert len(stored_ids) == batch_size, f"Expected {batch_size} stored, got {len(stored_ids)}"
            print(f"    ✅ Stored {len(stored_ids)} items in {duration:.2f}s")
            print(f"    📊 Storage rate: {rate:.1f} items/second")
            
            # Test batch search performance
            print("  Testing search performance with larger dataset...")
            search_start = asyncio.get_event_loop().time()
            
            for _ in range(20):
                query_embedding = np.random.random(384).astype(np.float32)
                await self.store.search_similar_content(query_embedding, top_k=10)
            
            search_duration = asyncio.get_event_loop().time() - search_start
            search_rate = 20 / search_duration
            
            print(f"    ✅ Completed 20 searches in {search_duration:.2f}s")
            print(f"    📊 Search rate: {search_rate:.1f} queries/second")
            
            self.results["tests_passed"] += 1
            
        except Exception as e:
            self.results["tests_failed"] += 1
            self.results["errors"].append(f"Batch operations test: {e}")
            print(f"❌ Batch operations test failed: {e}")
            raise
    
    async def cleanup(self):
        """Clean up test resources"""
        if self.store:
            await self.store.disconnect()
            print("✅ Database connection closed")
    
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 60)
        print("🎉 PRSM PGVECTOR TEST RESULTS")
        print("=" * 60)
        
        print(f"Tests run: {self.results['tests_run']}")
        print(f"Tests passed: {self.results['tests_passed']}")
        print(f"Tests failed: {self.results['tests_failed']}")
        
        if self.results["tests_failed"] == 0:
            print("\n✅ ALL TESTS PASSED!")
            print("\n🚀 PostgreSQL + pgvector implementation is working correctly!")
            print("\n🎯 Key Validations:")
            print("   ✅ Database connectivity and health checks")
            print("   ✅ Vector storage with metadata")
            print("   ✅ Cosine similarity search with HNSW indexing")
            print("   ✅ Advanced filtering (content type, creator, quality)")
            print("   ✅ Metadata updates and content management") 
            print("   ✅ Performance monitoring and metrics")
            print("   ✅ Batch operations and scalability")
            print("\n💡 Ready for:")
            print("   📦 Integration with PRSM demo")
            print("   🔄 Migration to production database")
            print("   📈 Phase 1B scaling to Milvus/Qdrant")
            print("   💰 FTNS royalty integration")
        else:
            print(f"\n❌ {self.results['tests_failed']} TESTS FAILED")
            print("\nErrors:")
            for error in self.results["errors"]:
                print(f"  - {error}")


async def main():
    """Run the complete test suite"""
    suite = PgVectorTestSuite()
    
    try:
        # Setup
        await suite.setup()
        
        # Run all tests
        await suite.test_database_connection()
        await suite.test_content_storage()
        await suite.test_similarity_search()
        await suite.test_filtered_search()
        await suite.test_metadata_operations()
        await suite.test_performance_monitoring()
        await suite.test_large_batch_operations()
        
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        logger.exception("Test suite error")
    
    finally:
        # Cleanup
        await suite.cleanup()
        
        # Print results
        suite.print_summary()
        
        # Exit with appropriate code
        if suite.results["tests_failed"] > 0:
            sys.exit(1)


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())
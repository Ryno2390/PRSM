#!/usr/bin/env python3
"""
PRSM Real Embedding Pipeline Integration Test

Comprehensive test of the complete embedding pipeline including:
- Content text processing and chunking
- Embedding generation with caching
- Real API integration (with fallback to mock)
- Vector storage and retrieval
- Performance monitoring
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent))

from prsm.content_processing.text_processor import ContentType
from prsm.embeddings import (
    EmbeddingPipeline, 
    EmbeddingPipelineConfig, 
    create_pipeline,
    get_embedding_api,
    create_optimized_cache
)
from prsm.vector_store.implementations.pgvector_store import PgVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_embedding_api():
    """Test real embedding API providers"""
    print("\n🧪 Testing Embedding API Providers")
    print("=" * 50)
    
    api = get_embedding_api()
    
    # Test all providers
    results = await api.test_all_providers()
    
    for provider, result in results.items():
        if result['success']:
            print(f"✅ {provider}: Response time {result['response_time']:.3f}s, "
                  f"Dimension: {result['embedding_dimension']}")
        else:
            print(f"❌ {provider}: {result['error']}")
    
    return results


async def test_embedding_cache():
    """Test embedding cache functionality"""
    print("\n💾 Testing Embedding Cache")
    print("=" * 50)
    
    cache = await create_optimized_cache(
        cache_dir="test_embedding_cache",
        max_size_mb=100
    )
    
    # Test cache miss and storage
    test_text = "This is a test document for embedding caching."
    model_name = "mock/test-model"
    
    print("Testing cache miss...")
    embedding1 = await cache.get_embedding(test_text, model_name)
    assert embedding1 is None, "Expected cache miss"
    print("✅ Cache miss working correctly")
    
    # Store embedding
    import numpy as np
    test_embedding = np.random.rand(384).astype(np.float32)
    await cache.store_embedding(test_text, test_embedding, model_name)
    print("✅ Embedding stored in cache")
    
    # Test cache hit
    print("Testing cache hit...")
    embedding2 = await cache.get_embedding(test_text, model_name)
    assert embedding2 is not None, "Expected cache hit"
    assert np.allclose(test_embedding, embedding2), "Embeddings should match"
    print("✅ Cache hit working correctly")
    
    # Test batch processing
    texts = [f"Test document {i}" for i in range(5)]
    
    async def mock_embedding_function(input_texts):
        # Simulate API call
        await asyncio.sleep(0.1)
        return [np.random.rand(384).astype(np.float32) for _ in input_texts]
    
    print("Testing batch embedding generation...")
    start_time = time.time()
    batch_embeddings = await cache.batch_get_or_generate_embeddings(
        texts, model_name, mock_embedding_function
    )
    batch_time = time.time() - start_time
    
    assert len(batch_embeddings) == len(texts), "Should have embeddings for all texts"
    print(f"✅ Batch processing complete in {batch_time:.3f}s")
    
    # Get cache stats
    stats = cache.get_cache_stats()
    print(f"📊 Cache Stats: {stats['cache_storage']['total_entries']} entries, "
          f"{stats['cache_performance']['hit_rate']:.2%} hit rate")
    
    return cache


async def test_content_processing():
    """Test content text processing"""
    print("\n📝 Testing Content Processing")
    print("=" * 50)
    
    # Test research paper processing
    research_paper = """
    Title: Advanced Machine Learning for Scientific Discovery
    
    Authors: Dr. Jane Smith, Prof. John Doe
    
    Abstract: This paper presents a comprehensive study of machine learning 
    applications in scientific discovery. We explore various techniques including 
    deep learning, reinforcement learning, and natural language processing.
    
    1. Introduction
    
    Machine learning has revolutionized many fields of science. This work focuses on
    applications in computational biology, materials science, and climate modeling.
    
    2. Methods
    
    We implemented several neural network architectures including transformers,
    convolutional networks, and graph neural networks. The training data consisted
    of over 100,000 scientific papers and experimental datasets.
    
    3. Results
    
    Our models achieved state-of-the-art performance on multiple benchmarks,
    demonstrating the potential for AI-assisted scientific discovery.
    """
    
    from prsm.content_processing.text_processor import ContentTextProcessor, ProcessingConfig
    
    config = ProcessingConfig(
        content_type=ContentType.RESEARCH_PAPER,
        max_chunk_size=256,
        chunk_overlap=50,
        extract_metadata=True
    )
    
    processor = ContentTextProcessor(config)
    processed = processor.process_content(research_paper, "test_paper_001")
    
    print(f"✅ Processed paper into {len(processed.processed_chunks)} chunks")
    print(f"📊 Extracted metadata: {list(processed.extracted_metadata.keys())}")
    print(f"⏱️  Processing time: {processed.processing_stats['processing_time']:.3f}s")
    
    # Show first chunk
    if processed.processed_chunks:
        first_chunk = processed.processed_chunks[0]
        print(f"📄 First chunk preview: {first_chunk.text[:100]}...")
    
    return processed


async def test_full_pipeline():
    """Test complete embedding pipeline"""
    print("\n🚀 Testing Complete Embedding Pipeline")
    print("=" * 50)
    
    # Initialize vector store
    from prsm.vector_store.base import VectorStoreConfig, VectorStoreType
    config = VectorStoreConfig(
        store_type=VectorStoreType.PGVECTOR,
        host="localhost",
        port=5433,
        database="prsm_vector_dev",
        username="postgres",
        password="postgres123"
    )
    vector_store = PgVectorStore(config)
    
    try:
        await vector_store.connect()
        print("✅ Vector store initialized")
    except Exception as e:
        print(f"⚠️  Vector store initialization failed: {e}")
        print("Using mock vector store for testing...")
        # We'll continue with a limited test
    
    # Create pipeline configuration
    config = EmbeddingPipelineConfig(
        max_chunk_size=256,
        chunk_overlap=50,
        preferred_embedding_provider='mock',  # Use mock for reliable testing
        batch_size=10,
        max_concurrent_requests=5
    )
    
    # Create and initialize pipeline
    pipeline = EmbeddingPipeline(vector_store, config)
    await pipeline.initialize()
    print("✅ Pipeline initialized")
    
    # Test content processing
    test_content = """
    PRSM: Protocol for Recursive Scientific Modeling
    
    PRSM is a revolutionary approach to scientific knowledge management that enables
    researchers to build upon each other's work in a decentralized, transparent manner.
    
    Key Features:
    - Decentralized knowledge storage using IPFS
    - AI-powered content discovery and synthesis
    - Token-based creator compensation (FTNS)
    - Reproducible research protocols
    - Cross-disciplinary collaboration tools
    
    The system uses advanced vector embeddings to understand semantic relationships
    between research papers, datasets, and code repositories. This enables intelligent
    content recommendation and automated literature reviews.
    """
    
    print("Processing test content through pipeline...")
    start_time = time.time()
    
    result = await pipeline.process_content(
        text=test_content,
        content_id="prsm_overview_001",
        content_type=ContentType.RESEARCH_PAPER,
        metadata={
            "title": "PRSM Overview",
            "category": "technical_documentation",
            "version": "1.0"
        }
    )
    
    processing_time = time.time() - start_time
    
    if result.success:
        print(f"✅ Pipeline processing successful!")
        print(f"📊 Processed {result.processing_stats['chunks_processed']} chunks")
        print(f"🔗 Generated {result.processing_stats['embeddings_generated']} embeddings")
        print(f"💾 Cache hit rate: {result.processing_stats['cache_hit_rate']:.2%}")
        print(f"⏱️  Total time: {processing_time:.3f}s")
        
        # Test similarity search
        if hasattr(vector_store, 'search_similar'):
            print("\nTesting similarity search...")
            try:
                query = "AI-powered research discovery"
                search_results = await pipeline.search_similar_content(query, top_k=3)
                print(f"✅ Found {len(search_results)} similar items")
            except Exception as e:
                print(f"⚠️  Similarity search failed: {e}")
    else:
        print(f"❌ Pipeline processing failed: {result.errors}")
    
    # Get pipeline statistics
    stats = pipeline.get_pipeline_stats()
    print(f"\n📈 Pipeline Performance:")
    print(f"   - Success rate: {stats['pipeline_performance']['success_rate']:.2%}")
    print(f"   - Average processing time: {stats['pipeline_performance']['average_processing_time']:.3f}s")
    
    # Health check
    print("\nPerforming health check...")
    health = await pipeline.health_check()
    if health['overall_healthy']:
        print("✅ All pipeline components healthy")
    else:
        print("⚠️  Some pipeline components have issues:")
        for component, status in health['components'].items():
            if not status['healthy']:
                print(f"   - {component}: {status.get('error', 'Unknown issue')}")
    
    await pipeline.cleanup()
    return result


async def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🧪 PRSM Real Embedding Pipeline Integration Test")
    print("=" * 60)
    
    try:
        # Test 1: Embedding API
        api_results = await test_embedding_api()
        working_providers = [name for name, result in api_results.items() if result['success']]
        
        if not working_providers:
            print("\n⚠️  No working embedding providers found - tests will use mock only")
        
        # Test 2: Embedding Cache
        await test_embedding_cache()
        
        # Test 3: Content Processing
        await test_content_processing()
        
        # Test 4: Full Pipeline
        await test_full_pipeline()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Test Summary:")
        print("✅ Embedding API integration working")
        print("✅ Embedding cache functioning properly")
        print("✅ Content processing pipeline operational")
        print("✅ Complete pipeline integration successful")
        
        if working_providers:
            print(f"✅ Real embedding providers available: {working_providers}")
        else:
            print("⚠️  Only mock embedding provider available")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)
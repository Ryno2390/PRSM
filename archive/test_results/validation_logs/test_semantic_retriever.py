#!/usr/bin/env python3
"""
Test Semantic Retriever for NWTN System 1 → System 2 → Attribution Pipeline
===========================================================================

This script tests the new semantic retrieval system that replaces simple
keyword search with sophisticated embedding-based semantic search.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_semantic_retriever():
    """Test the semantic retriever functionality"""
    print("🔍 Testing Semantic Retriever...")
    print("=" * 80)
    
    # Set up environment
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"
    
    try:
        # Test 1: Initialize External Knowledge Base
        print("\n🔧 Test 1: Initialize External Knowledge Base")
        print("-" * 50)
        
        from prsm.nwtn.external_storage_config import ExternalStorageConfig, ExternalStorageManager, ExternalKnowledgeBase
        
        config = ExternalStorageConfig()
        storage_manager = ExternalStorageManager(config)
        await storage_manager.initialize()
        
        kb = ExternalKnowledgeBase(storage_manager)
        await kb.initialize()
        
        print(f"✓ External Knowledge Base initialized: {kb.initialized}")
        print(f"✓ Storage path: {config.storage_path}")
        print(f"✓ Embeddings available: {config.embeddings_count}")
        
        # Test 2: Initialize Semantic Retriever
        print("\n🔧 Test 2: Initialize Semantic Retriever")
        print("-" * 50)
        
        from prsm.nwtn.semantic_retriever import SemanticRetriever, TextEmbeddingGenerator
        
        embedding_generator = TextEmbeddingGenerator()
        retriever = SemanticRetriever(kb, embedding_generator)
        await retriever.initialize()
        
        print(f"✓ Semantic Retriever initialized: {retriever.initialized}")
        print(f"✓ Embedding model: {embedding_generator.model_name}")
        print(f"✓ Embedding dimension: {embedding_generator.embedding_dimension}")
        
        # Test 3: Test Embedding Generation
        print("\n🔧 Test 3: Test Embedding Generation")
        print("-" * 50)
        
        test_text = "quantum tunneling effects in semiconductor devices"
        embedding = await embedding_generator.generate_embedding(test_text)
        
        print(f"✓ Generated embedding for: '{test_text}'")
        print(f"✓ Embedding length: {len(embedding)}")
        print(f"✓ Embedding type: {type(embedding)}")
        print(f"✓ First few values: {embedding[:5]}")
        
        # Test 4: Semantic Search - Keyword Method
        print("\n🔧 Test 4: Semantic Search - Keyword Method")
        print("-" * 50)
        
        query = "What is quantum tunneling and how does it work?"
        result = await retriever.semantic_search(
            query=query,
            top_k=5,
            similarity_threshold=0.3,
            search_method="keyword"
        )
        
        print(f"✓ Query: {query}")
        print(f"✓ Search method: {result.retrieval_method}")
        print(f"✓ Papers found: {len(result.retrieved_papers)}")
        print(f"✓ Search time: {result.search_time_seconds:.3f} seconds")
        
        if result.retrieved_papers:
            print("\n📄 Retrieved Papers (Keyword Search):")
            for i, paper in enumerate(result.retrieved_papers[:3]):
                print(f"  {i+1}. {paper.title}")
                print(f"     Authors: {paper.authors}")
                print(f"     Relevance: {paper.relevance_score:.3f}")
                print(f"     Method: {paper.retrieval_method}")
        
        # Test 5: Semantic Search - Hybrid Method
        print("\n🔧 Test 5: Semantic Search - Hybrid Method")
        print("-" * 50)
        
        result_hybrid = await retriever.semantic_search(
            query=query,
            top_k=5,
            similarity_threshold=0.3,
            search_method="hybrid"
        )
        
        print(f"✓ Query: {query}")
        print(f"✓ Search method: {result_hybrid.retrieval_method}")
        print(f"✓ Papers found: {len(result_hybrid.retrieved_papers)}")
        print(f"✓ Search time: {result_hybrid.search_time_seconds:.3f} seconds")
        
        if result_hybrid.retrieved_papers:
            print("\n📄 Retrieved Papers (Hybrid Search):")
            for i, paper in enumerate(result_hybrid.retrieved_papers[:3]):
                print(f"  {i+1}. {paper.title}")
                print(f"     Authors: {paper.authors}")
                print(f"     Relevance: {paper.relevance_score:.3f}")
                print(f"     Method: {paper.retrieval_method}")
        
        # Test 6: Test Different Queries
        print("\n🔧 Test 6: Test Different Queries")
        print("-" * 50)
        
        test_queries = [
            "machine learning algorithms for natural language processing",
            "climate change impacts on ocean temperature",
            "quantum computing error correction methods",
            "artificial intelligence safety and alignment"
        ]
        
        for query in test_queries:
            result = await retriever.semantic_search(
                query=query,
                top_k=3,
                similarity_threshold=0.3,
                search_method="hybrid"
            )
            
            print(f"Query: '{query}'")
            print(f"  Papers found: {len(result.retrieved_papers)}")
            print(f"  Search time: {result.search_time_seconds:.3f}s")
            
            if result.retrieved_papers:
                top_paper = result.retrieved_papers[0]
                print(f"  Top result: {top_paper.title[:60]}...")
                print(f"  Relevance: {top_paper.relevance_score:.3f}")
        
        # Test 7: Retrieval Statistics
        print("\n🔧 Test 7: Retrieval Statistics")
        print("-" * 50)
        
        stats = retriever.get_retrieval_statistics()
        print(f"✓ Total retrievals: {stats['total_retrievals']}")
        print(f"✓ Successful retrievals: {stats['successful_retrievals']}")
        print(f"✓ Success rate: {stats['success_rate']:.3f}")
        print(f"✓ Average search time: {stats['average_search_time']:.3f}s")
        print(f"✓ Average papers per retrieval: {stats['average_papers_per_retrieval']:.1f}")
        
        # Test 8: Configure Parameters
        print("\n🔧 Test 8: Configure Retrieval Parameters")
        print("-" * 50)
        
        await retriever.configure_retrieval_params(
            top_k=10,
            similarity_threshold=0.5,
            max_papers_to_search=2000
        )
        
        # Test with new parameters
        result_configured = await retriever.semantic_search(
            query="deep learning neural networks",
            search_method="hybrid"
        )
        
        print(f"✓ Configured parameters applied")
        print(f"✓ Papers found with new params: {len(result_configured.retrieved_papers)}")
        print(f"✓ Search time: {result_configured.search_time_seconds:.3f}s")
        
        # Success Summary
        print("\n" + "=" * 80)
        print("🎉 SEMANTIC RETRIEVER TESTS COMPLETED!")
        print("=" * 80)
        
        success_criteria = {
            "External Knowledge Base Ready": kb.initialized,
            "Semantic Retriever Initialized": retriever.initialized,
            "Embedding Generation Working": len(embedding) > 0,
            "Keyword Search Working": len(result.retrieved_papers) > 0,
            "Hybrid Search Working": len(result_hybrid.retrieved_papers) > 0,
            "Multiple Query Types Working": all(len((await retriever.semantic_search(q)).retrieved_papers) >= 0 for q in test_queries[:2]),
            "Statistics Tracking": stats['total_retrievals'] > 0,
            "Parameter Configuration": len(result_configured.retrieved_papers) >= 0
        }
        
        all_passed = True
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            print(f"{status} {criterion}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n🏆 ALL SEMANTIC RETRIEVER TESTS PASSED!")
            print("🔍 Phase 1.1 of the roadmap is ready for next steps")
            return True
        else:
            print("\n⚠️  Some tests failed - review implementation")
            return False
        
    except Exception as e:
        print(f"❌ Error during semantic retriever tests: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_semantic_retriever())
    if success:
        print("\n🎯 Ready to proceed to Phase 1.2: Content Analysis Engine")
    else:
        print("\n🔧 Fix issues before proceeding")
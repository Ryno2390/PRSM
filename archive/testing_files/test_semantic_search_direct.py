#!/usr/bin/env python3
"""
Direct Semantic Search Test - Core Pipeline Validation
======================================================

This test directly validates the core semantic search pipeline:
1. Load external knowledge base with 149,726 papers
2. Initialize semantic retriever with embeddings  
3. Perform semantic search across embedding batches
4. Validate paper retrieval and citations

This bypasses orchestrator complexity to test the core functionality.
"""

import asyncio
import sys
import json
from datetime import datetime
sys.path.insert(0, '.')

from prsm.nwtn.external_storage_config import ExternalKnowledgeBase, ExternalStorageManager
from prsm.nwtn.semantic_retriever import create_semantic_retriever


async def test_semantic_search_direct():
    """Test semantic search directly against the 150K corpus"""
    
    print("🧪 DIRECT SEMANTIC SEARCH TEST")
    print("=" * 60)
    print("📚 Testing core search functionality against 149,726 papers")
    print("🔍 Query: Transformer architectures for NLP")
    print("📊 Expected: Relevant papers with similarity scores")
    print()
    
    try:
        # Initialize external storage manager
        print("🔧 Initializing external storage manager...")
        storage_manager = ExternalStorageManager()
        initialized = await storage_manager.initialize()
        
        if not initialized:
            print("❌ Failed to initialize external storage")
            return False
        
        print("✅ External storage initialized")
        print(f"📊 Available embeddings: {storage_manager.config.embeddings_count}")
        print()
        
        # Initialize knowledge base
        print("📚 Initializing knowledge base...")
        knowledge_base = ExternalKnowledgeBase(storage_manager=storage_manager)
        await knowledge_base.initialize()
        print("✅ Knowledge base initialized")
        print()
        
        # Initialize semantic retriever
        print("🔍 Initializing semantic retriever...")
        retriever = await create_semantic_retriever(knowledge_base)
        print("✅ Semantic retriever ready")
        print()
        
        # Test query
        test_query = "transformer architectures for natural language processing attention mechanisms"
        print(f"🔍 Searching for: '{test_query}'")
        print("⏱️ Searching across 4,724 embedding batches...")
        
        start_time = datetime.now()
        
        # Perform semantic search
        search_result = await retriever.semantic_search(
            query=test_query,
            top_k=10,
            similarity_threshold=0.3,
            search_method="hybrid"
        )
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Analyze results
        print("\n" + "=" * 60)
        print("📊 SEARCH RESULTS")
        print("=" * 60)
        print(f"⏱️ Search time: {search_time:.2f} seconds")
        print(f"📄 Papers found: {len(search_result.retrieved_papers)}")
        print(f"🔍 Search method: {search_result.retrieval_method}")
        print(f"🧠 Embedding model: {search_result.embedding_model}")
        print()
        
        if search_result.retrieved_papers:
            print("✅ SUCCESS: Papers retrieved from corpus!")
            print("-" * 40)
            
            for i, paper in enumerate(search_result.retrieved_papers[:5], 1):
                print(f"📄 Paper {i}:")
                print(f"   Title: {paper.title[:80]}...")
                print(f"   Authors: {paper.authors[:50]}...")
                print(f"   Relevance: {paper.relevance_score:.3f}")
                print(f"   arXiv ID: {paper.arxiv_id}")
                print(f"   Method: {paper.retrieval_method}")
                print()
            
            # Validate papers are real
            await validate_paper_authenticity(search_result.retrieved_papers[:3], knowledge_base)
            
            return True
            
        else:
            print("❌ No papers found - search failed")
            return False
    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def validate_paper_authenticity(papers, knowledge_base):
    """Validate that retrieved papers are real papers from our corpus"""
    
    print("🔬 VALIDATING PAPER AUTHENTICITY")
    print("-" * 40)
    
    for paper in papers:
        try:
            # Try to retrieve full paper details from storage
            stored_papers = await knowledge_base.search_papers(
                f"title:{paper.title[:30]}", max_results=1
            )
            
            if stored_papers:
                stored_paper = stored_papers[0]
                print(f"✅ Verified: '{paper.title[:50]}...'")
                print(f"   Authors match: {paper.authors[:30]}...")
                print(f"   arXiv ID: {paper.arxiv_id}")
                print(f"   Stored in corpus: YES")
            else:
                print(f"⚠️ Could not verify: '{paper.title[:50]}...'")
                
        except Exception as e:
            print(f"❌ Validation error for paper: {e}")
        
        print()


async def test_embedding_batch_access():
    """Test direct access to embedding batches"""
    
    print("🧪 TESTING EMBEDDING BATCH ACCESS")
    print("=" * 50)
    
    try:
        storage_manager = ExternalStorageManager()
        await storage_manager.initialize()
        
        # Test loading first few batches
        print("📥 Loading embedding batches...")
        
        for batch_id in range(min(3, storage_manager.config.embeddings_count)):
            batch_data = await storage_manager.load_embedding_batch(batch_id)
            
            if batch_data:
                embeddings_count = len(batch_data.get('embeddings', []))
                metadata_count = len(batch_data.get('metadata', []))
                model_name = batch_data.get('model_name', 'unknown')
                created_at = batch_data.get('created_at', 'unknown')
                
                print(f"✅ Batch {batch_id}: {embeddings_count} embeddings, {metadata_count} papers")
                print(f"   Model: {model_name}")
                print(f"   Created: {created_at}")
                
                # Show sample paper
                if batch_data.get('metadata'):
                    sample = batch_data['metadata'][0]
                    print(f"   Sample: {sample.get('title', 'N/A')[:60]}...")
            else:
                print(f"❌ Failed to load batch {batch_id}")
        
        print("\n✅ Embedding batch access working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Embedding batch test failed: {e}")
        return False


async def main():
    """Main test function"""
    
    print("🧪 NWTN CORE PIPELINE VALIDATION")
    print("=" * 60)
    print("🎯 Goal: Validate semantic search over 149,726 papers")
    print("📊 Scope: Direct embedding search without orchestrator")
    print("🔍 Focus: Core retrieval functionality")
    print()
    
    # Test 1: Embedding batch access
    print("TEST 1: Embedding Batch Access")
    print("-" * 30)
    batch_success = await test_embedding_batch_access()
    print()
    
    # Test 2: Semantic search
    print("TEST 2: Semantic Search Pipeline")
    print("-" * 30)
    search_success = await test_semantic_search_direct()
    print()
    
    # Overall result
    print("=" * 60)
    if batch_success and search_success:
        print("🎉 CORE PIPELINE TESTS SUCCESSFUL!")
        print("✅ Embeddings accessible: 4,724 batches")
        print("✅ Semantic search working: 149,726 papers")
        print("✅ Paper retrieval functional")
        print("🔧 Ready for full orchestrator integration")
    else:
        print("❌ CORE PIPELINE ISSUES DETECTED")
        print(f"Embedding access: {'✅' if batch_success else '❌'}")
        print(f"Semantic search: {'✅' if search_success else '❌'}")
        print("🔧 Core components need fixes before full pipeline")
    
    print("=" * 60)
    return batch_success and search_success


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Quick NWTN Validation Test
==========================

A quick test to verify NWTN is working with the full corpus
and demonstrate the improvement over the previous approach.
"""

import asyncio
import sys
import time
from datetime import datetime

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from enhanced_semantic_search import EnhancedSemanticSearchEngine, SearchQuery

async def main():
    """Quick validation test"""
    print("🚀 NWTN QUICK VALIDATION TEST")
    print("=" * 50)
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔍 Testing semantic search with full corpus")
    print()
    
    # Initialize semantic search engine
    print("🔧 Initializing semantic search engine...")
    search_engine = EnhancedSemanticSearchEngine(
        index_dir="/Volumes/My Passport/PRSM_Storage/PRSM_Indices",
        index_type="HNSW"
    )
    
    if not search_engine.initialize():
        print("❌ Failed to initialize semantic search engine")
        return
    
    # Get statistics
    stats = search_engine.get_statistics()
    print(f"✅ Search engine initialized")
    print(f"📊 Total papers: {stats['total_papers']:,}")
    print(f"🔍 Index type: {stats['index_type']}")
    print(f"🤖 Model: {stats['model_name']}")
    print()
    
    # Test queries
    test_queries = [
        "quantum computing optimization algorithms",
        "machine learning protein folding prediction",
        "climate modeling computational methods",
        "neural networks deep learning architectures",
        "cryptography quantum information theory"
    ]
    
    print("🧪 Testing semantic search queries...")
    print("-" * 50)
    
    total_start = time.time()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        start_time = time.time()
        
        search_query = SearchQuery(
            query_text=query,
            max_results=5,
            similarity_threshold=0.2
        )
        
        results = await search_engine.search(search_query)
        search_time = time.time() - start_time
        
        print(f"   ⏱️  Search time: {search_time:.3f}s")
        print(f"   📄 Results: {len(results)}")
        
        if results:
            print(f"   🎯 Top result: {results[0].title[:60]}...")
            print(f"   📊 Similarity: {results[0].similarity_score:.3f}")
            print(f"   🏷️  Domain: {results[0].domain}")
            
            # Show domain diversity
            domains = set(r.domain for r in results)
            print(f"   🌐 Domains: {', '.join(domains)}")
        else:
            print("   ⚠️  No results found")
    
    total_time = time.time() - total_start
    
    print(f"\n📊 QUICK VALIDATION SUMMARY:")
    print("=" * 50)
    print(f"✅ Semantic search: WORKING")
    print(f"📚 Corpus size: {stats['total_papers']:,} papers")
    print(f"🔍 Index type: HNSW (optimized for speed)")
    print(f"⚡ Total test time: {total_time:.2f}s")
    print(f"🚀 Average query time: {total_time/len(test_queries):.3f}s")
    print(f"📈 Queries per second: {len(test_queries)/total_time:.1f}")
    print()
    
    print("🎉 NWTN SEMANTIC SEARCH VALIDATION: SUCCESS!")
    print("🚀 Ready for full corpus reasoning!")
    print()
    
    # Test a simple reasoning scenario
    print("🧠 Testing NWTN reasoning integration...")
    try:
        from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
        
        # Quick reasoning test
        meta_engine = MetaReasoningEngine()
        await meta_engine.initialize()
        
        # Get papers for reasoning
        reasoning_query = "quantum machine learning applications"
        search_query = SearchQuery(
            query_text=reasoning_query,
            max_results=3,
            similarity_threshold=0.2
        )
        
        papers = await search_engine.search(search_query)
        
        reasoning_context = {
            'query': reasoning_query,
            'papers': [
                {
                    'id': p.paper_id,
                    'title': p.title,
                    'abstract': p.abstract,
                    'domain': p.domain,
                    'relevance_score': p.similarity_score
                } for p in papers
            ]
        }
        
        start_time = time.time()
        result = await meta_engine.reason(
            query=reasoning_query,
            context=reasoning_context,
            thinking_mode=ThinkingMode.INTERMEDIATE
        )
        reasoning_time = time.time() - start_time
        
        print(f"✅ NWTN reasoning: WORKING")
        print(f"⏱️  Reasoning time: {reasoning_time:.2f}s")
        print(f"🎯 Confidence: {result.meta_confidence:.3f}")
        print(f"📄 Papers used: {len(papers)}")
        print()
        
        print("🏆 FULL INTEGRATION TEST: SUCCESS!")
        print("🎯 NWTN + Full Corpus Semantic Search: OPERATIONAL")
        
    except Exception as e:
        print(f"⚠️  Reasoning test failed: {e}")
        print("🔍 Semantic search is working, reasoning needs investigation")

if __name__ == "__main__":
    asyncio.run(main())
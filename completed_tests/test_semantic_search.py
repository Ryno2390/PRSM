#!/usr/bin/env python3
"""
Test Enhanced Semantic Search Engine with test index
"""

import asyncio
import sys
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from enhanced_semantic_search import EnhancedSemanticSearchEngine, SearchQuery

async def main():
    # Use test index
    engine = EnhancedSemanticSearchEngine(
        index_dir="/Volumes/My Passport/PRSM_Storage/PRSM_Indices_Test",
        index_type="Flat"
    )
    
    if not engine.initialize():
        print("Failed to initialize search engine")
        return
    
    # Test queries
    test_queries = [
        "quantum mechanics and relativity theory",
        "machine learning neural networks",
        "computer vision object detection",
        "natural language processing transformers",
        "reinforcement learning algorithms"
    ]
    
    print("🔍 Testing Enhanced Semantic Search Engine")
    print("=" * 50)
    
    for query_text in test_queries:
        query = SearchQuery(
            query_text=query_text,
            max_results=3,
            similarity_threshold=0.2  # Lower threshold for test
        )
        
        results = await engine.search(query)
        
        print(f"\n🔍 Query: '{query_text}'")
        print(f"📊 Results: {len(results)}")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. ({result.similarity_score:.3f}) {result.title}")
            print(f"     Domain: {result.domain} | Authors: {', '.join(result.authors[:2])}")
            print(f"     Abstract: {result.abstract[:100]}...")
    
    # Print statistics
    stats = engine.get_statistics()
    print(f"\n📊 Search Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Verify NWTN Interface Fix
========================

Quick verification that the NWTN interface fix worked.
"""

import asyncio
import sys
import time
from datetime import datetime

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
from enhanced_semantic_search import EnhancedSemanticSearchEngine, SearchQuery

async def main():
    print("🔧 NWTN Interface Fix Verification")
    print("=" * 50)
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test simple query
    test_query = "quantum computing drug discovery integration"
    
    print(f"🧪 Testing query: '{test_query}'")
    print()
    
    # Initialize semantic search
    print("🔍 Initializing semantic search...")
    search_engine = EnhancedSemanticSearchEngine(
        index_dir="/Volumes/My Passport/PRSM_Storage/PRSM_Indices",
        index_type="HNSW"
    )
    
    if not search_engine.initialize():
        print("❌ Failed to initialize semantic search")
        return
    
    # Get papers
    search_query = SearchQuery(
        query_text=test_query,
        max_results=5,
        similarity_threshold=0.2
    )
    
    papers = await search_engine.search(search_query)
    print(f"📄 Retrieved {len(papers)} papers")
    
    # Test NWTN with corrected interface
    print("🧠 Testing NWTN meta_reason method...")
    
    meta_engine = MetaReasoningEngine()
    
    reasoning_context = {
        'query': test_query,
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
    
    try:
        start_time = time.time()
        
        # Use the CORRECTED method name: meta_reason
        result = await meta_engine.meta_reason(
            query=test_query,
            context=reasoning_context,
            thinking_mode=ThinkingMode.INTERMEDIATE
        )
        
        reasoning_time = time.time() - start_time
        
        print(f"✅ NWTN meta_reason: SUCCESS")
        print(f"⏱️  Reasoning time: {reasoning_time:.2f}s")
        print(f"🎯 Meta-confidence: {result.meta_confidence:.3f}")
        print(f"📄 Papers processed: {len(papers)}")
        
        if hasattr(result, 'reasoning_engines_used'):
            engines_used = list(result.reasoning_engines_used.keys())
            print(f"🧠 Reasoning engines used: {len(engines_used)}")
        
        print()
        print("🎉 NWTN INTERFACE FIX: CONFIRMED!")
        print("🚀 meta_reason method working correctly")
        print("✅ Ready for complex R&D strategy prompts")
        
    except Exception as e:
        print(f"❌ NWTN meta_reason failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Check available methods
        methods = [m for m in dir(meta_engine) if not m.startswith('_')]
        print(f"Available methods: {[m for m in methods if 'reason' in m]}")

if __name__ == "__main__":
    asyncio.run(main())
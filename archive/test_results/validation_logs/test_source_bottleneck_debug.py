#!/usr/bin/env python3
"""
Source Bottleneck Debug Test
============================

Trace the exact path of source retrieval to find where we're getting 
limited to 1 source instead of 3+ sources.
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.external_storage_config import ExternalStorageConfig

async def debug_source_bottleneck():
    """Debug exactly where source count gets limited"""
    print("🔍 Source Bottleneck Debug Test")
    print("=" * 50)
    
    # Initialize system integrator
    external_storage = ExternalStorageConfig()
    integrator = SystemIntegrator(external_storage_config=external_storage)
    await integrator.initialize()
    
    # Simple query for debugging
    test_query = 'machine learning algorithms'
    print(f"📝 Test Query: '{test_query}'")
    
    # Step 1: Test semantic retriever directly
    print(f"\n🔍 Step 1: Direct Semantic Retrieval...")
    semantic_results = await integrator.external_storage_config.search_papers(test_query, max_results=25)
    print(f"   Semantic Results: {len(semantic_results)} papers")
    
    # Step 2: Test meta reasoning engine external search
    print(f"\n🔍 Step 2: Meta Reasoning External Search...")
    try:
        meta_papers = await integrator.meta_reasoning_engine.external_knowledge_base.search_papers(test_query, max_results=20)
        print(f"   Meta Reasoning Results: {len(meta_papers)} papers")
    except Exception as e:
        print(f"   Meta Reasoning Error: {e}")
    
    # Step 3: Test full pipeline and trace citations
    print(f"\n🔍 Step 3: Full Pipeline Test...")
    start_time = time.time()
    result = await integrator.process_complete_query(
        query=test_query,
        user_id='debug_test',
        query_cost=2.0
    )
    end_time = time.time()
    
    print(f"   Pipeline Success: {result.success}")
    print(f"   Final Citations: {len(result.citations)}")
    print(f"   Processing Time: {end_time - start_time:.1f}s")
    
    if result.success and len(result.citations) > 0:
        print(f"\n📚 Citations Found:")
        for i, citation in enumerate(result.citations, 1):
            # Handle both string and dict citation formats
            if isinstance(citation, dict):
                title = citation.get('title', f'Citation {i}')
            else:
                title = str(citation)
            
            display_title = title[:60] + '...' if len(title) > 60 else title
            print(f"   {i}. {display_title}")
    
    # Step 4: Test citation filter directly if we can access it
    print(f"\n🔍 Step 4: Citation Filter Analysis...")
    if hasattr(integrator, 'citation_filter'):
        print(f"   Citation Filter Max: {integrator.citation_filter.max_citations}")
        print(f"   Relevance Threshold: {integrator.citation_filter.default_relevance_threshold}")
        print(f"   Contribution Threshold: {integrator.citation_filter.default_contribution_threshold}")
    
    # Diagnosis
    print(f"\n🎯 Bottleneck Analysis:")
    if len(semantic_results) < 3:
        print(f"   ❌ BOTTLENECK: Semantic search only finding {len(semantic_results)} papers")
    elif len(result.citations) < 3:
        print(f"   ❌ BOTTLENECK: Pipeline reducing {len(semantic_results)} semantic results to {len(result.citations)} citations")
        print(f"   🔧 Likely bottleneck: Citation filtering or meta-reasoning synthesis")
    else:
        print(f"   ✅ SUCCESS: Found {len(result.citations)} sources as expected")
    
    return len(result.citations) >= 3

if __name__ == "__main__":
    success = asyncio.run(debug_source_bottleneck())
    
    if success:
        print(f"\n✅ Source retrieval working as expected")
    else:
        print(f"\n🔧 Source retrieval bottleneck identified - needs fixing")
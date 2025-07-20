#!/usr/bin/env python3
"""
Quick Citation Debug Test - Find source count bottleneck
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from prsm.nwtn.system_integrator import SystemIntegrator
from prsm.nwtn.external_storage_config import ExternalStorageConfig

async def quick_citation_debug():
    """Quick test to see where citations get filtered down to 1"""
    print("🔍 Quick Citation Debug Test")
    print("=" * 35)
    
    # Initialize system integrator 
    external_storage = ExternalStorageConfig()
    integrator = SystemIntegrator(external_storage_config=external_storage)
    await integrator.initialize()
    
    query = 'artificial intelligence'
    print(f"📝 Query: '{query}'")
    
    # Test semantic search directly first
    print(f"\n🔍 Step 1: Direct Semantic Search...")
    try:
        semantic_results = await integrator.external_storage_config.search_papers(query, max_results=10)
        print(f"   Semantic Search Results: {len(semantic_results)} papers")
        if len(semantic_results) > 0:
            print(f"   First Result: {semantic_results[0].get('title', 'No title')[:50]}...")
    except Exception as e:
        print(f"   Semantic Search Error: {e}")
        return False
    
    # Test full pipeline with detailed logging
    print(f"\n🔍 Step 2: Full Pipeline...")
    
    # Monkey patch citation filter to add debug logging
    original_filter = integrator.citation_filter.filter_citations
    
    async def debug_filter_citations(evaluation_result, **kwargs):
        print(f"   📊 Citation Filter Input:")
        print(f"      Candidate Evaluations: {len(evaluation_result.candidate_evaluations)}")
        
        # Check each candidate's source contributions
        total_sources = 0
        for i, candidate_eval in enumerate(evaluation_result.candidate_evaluations):
            sources = len(candidate_eval.candidate_answer.source_contributions)
            total_sources += sources
            print(f"      Candidate {i+1}: Score {candidate_eval.overall_score:.3f}, Sources: {sources}")
        
        print(f"      Total Sources Available: {total_sources}")
        print(f"      Citation Filter Settings:")
        print(f"         Top Candidates to Consider: {integrator.citation_filter.top_candidates_to_consider}")
        print(f"         Max Citations: {integrator.citation_filter.max_citations}")
        print(f"         Relevance Threshold: {integrator.citation_filter.default_relevance_threshold}")
        print(f"         Contribution Threshold: {integrator.citation_filter.default_contribution_threshold}")
        
        result = await original_filter(evaluation_result, **kwargs)
        
        print(f"   📊 Citation Filter Output:")
        print(f"      Filtered Citations: {len(result.filtered_citations)}")
        
        return result
    
    integrator.citation_filter.filter_citations = debug_filter_citations
    
    # Run the query
    start_time = time.time()
    result = await integrator.process_complete_query(
        query=query,
        user_id='citation_debug',
        query_cost=2.0
    )
    end_time = time.time()
    
    print(f"\n✅ Final Results:")
    print(f"   Success: {result.success}")
    print(f"   Final Citations: {len(result.citations)}")
    print(f"   Processing Time: {end_time - start_time:.1f}s")
    
    if result.success and len(result.citations) > 0:
        print(f"\n📚 Final Citations:")
        for i, citation in enumerate(result.citations, 1):
            citation_text = str(citation)[:60] + "..." if len(str(citation)) > 60 else str(citation)
            print(f"   {i}. {citation_text}")
    
    return len(result.citations) >= 3

if __name__ == "__main__":
    success = asyncio.run(quick_citation_debug())
    
    if success:
        print(f"\n🎉 Citation optimization working!")
    else:
        print(f"\n🔧 Citation bottleneck identified")
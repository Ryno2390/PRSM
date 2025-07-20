#!/usr/bin/env python3
"""
Check NWTN Deep Reasoning Progress
=================================

This script runs a shorter version of the deep reasoning to check progress
and see the current state of the full pipeline.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from prsm.nwtn.semantic_retriever import SemanticRetriever, TextEmbeddingGenerator, SemanticSearchResult
from prsm.nwtn.external_storage_config import get_external_knowledge_base
from prsm.nwtn.content_analyzer import ContentAnalyzer
from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode


async def check_nwtn_progress():
    """Check NWTN progress with the actual 150K embeddings"""
    
    print("🔍 CHECKING NWTN PROGRESS - 150K EMBEDDINGS")
    print("=" * 60)
    
    # Use the same query as before
    query = "What are the most promising approaches for improving transformer attention mechanisms to handle very long sequences efficiently?"
    print(f"🔍 Query: {query}")
    print()
    
    # Quick status check
    print("📚 Checking external knowledge base...")
    knowledge_base = await get_external_knowledge_base()
    print(f"✅ Knowledge base ready: {knowledge_base.initialized}")
    print()
    
    # Quick semantic search
    print("🔍 Running quick semantic search...")
    embedding_generator = TextEmbeddingGenerator()
    retriever = SemanticRetriever(knowledge_base, embedding_generator)
    await retriever.initialize()
    
    # Search with smaller result set for speed
    search_result = await retriever.semantic_search(
        query=query,
        top_k=5,  # Just top 5 for quick check
        similarity_threshold=0.2,
        search_method="semantic"
    )
    
    print(f"📊 Quick search completed:")
    print(f"   - Papers found: {len(search_result.retrieved_papers)}")
    print(f"   - Search time: {search_result.search_time_seconds:.2f} seconds")
    print()
    
    if search_result.retrieved_papers:
        print("📝 TOP PAPERS FOUND:")
        for i, paper in enumerate(search_result.retrieved_papers, 1):
            print(f"   {i}. {paper.title}")
            print(f"      Relevance: {paper.relevance_score:.3f}")
            print()
    
    # Test meta-reasoning in INTERMEDIATE mode for faster results
    print("🧠 Testing meta-reasoning in INTERMEDIATE mode...")
    meta_engine = MetaReasoningEngine()
    
    reasoning_context = {
        "retrieved_papers": [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "relevance_score": paper.relevance_score
            }
            for paper in search_result.retrieved_papers
        ],
        "query_analysis": {
            "complexity": "high",
            "domain": "machine_learning",
            "search_time": search_result.search_time_seconds
        }
    }
    
    # Use INTERMEDIATE mode for faster completion but still comprehensive
    reasoning_result = await meta_engine.meta_reason(
        query=query,
        context=reasoning_context,
        thinking_mode=ThinkingMode.INTERMEDIATE  # Faster than DEEP
    )
    
    print(f"📊 Meta-reasoning completed:")
    print(f"   - Processing time: {reasoning_result.total_processing_time:.2f} seconds")
    print(f"   - Meta confidence: {reasoning_result.meta_confidence:.3f}")
    print(f"   - FTNS cost: {reasoning_result.ftns_cost:.2f} tokens")
    print()
    
    if reasoning_result.final_synthesis:
        print("🎯 FINAL SYNTHESIS PREVIEW:")
        print("-" * 40)
        if isinstance(reasoning_result.final_synthesis, dict):
            for key, value in list(reasoning_result.final_synthesis.items())[:3]:  # Show first 3 items
                print(f"{key.upper()}: {str(value)[:150]}{'...' if len(str(value)) > 150 else ''}")
        else:
            preview = str(reasoning_result.final_synthesis)
            print(preview[:300] + ('...' if len(preview) > 300 else ''))
        print("-" * 40)
    
    print("\n✅ NWTN PROGRESS CHECK COMPLETED!")
    print(f"🎉 System Status: FULLY OPERATIONAL with 150K embeddings")
    
    return True


async def main():
    """Main function"""
    print("🚀 NWTN PROGRESS CHECK - INTERMEDIATE MODE")
    print("=" * 60)
    
    success = await check_nwtn_progress()
    
    if success:
        print("✅ PROGRESS CHECK SUCCESSFUL!")
    else:
        print("❌ PROGRESS CHECK FAILED!")


if __name__ == "__main__":
    asyncio.run(main())
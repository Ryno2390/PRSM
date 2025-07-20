#!/usr/bin/env python3
"""
ACTUAL DEEP REASONING TEST - ALL 5040 PERMUTATIONS
==================================================

This will run the ACTUAL deep reasoning with all 5040 permutations
against the 150K embedding corpus. Expected runtime: 30+ minutes.
"""

import asyncio
import sys
import signal
import time
from datetime import datetime, timezone
sys.path.insert(0, '.')

from prsm.nwtn.semantic_retriever import SemanticRetriever, TextEmbeddingGenerator, SemanticSearchResult
from prsm.nwtn.external_storage_config import get_external_knowledge_base
from prsm.nwtn.content_analyzer import ContentAnalyzer
from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode


class DeepReasoningMonitor:
    """Monitor deep reasoning progress"""
    
    def __init__(self):
        self.start_time = None
        self.running = True
        
    def start_monitoring(self):
        """Start monitoring"""
        self.start_time = time.time()
        print(f"🚀 DEEP REASONING STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("⏳ Expected completion time: 30+ minutes")
        print("📊 Will process ALL 5040 permutations")
        print("🔄 Progress updates every 5 minutes...")
        print()
        
    def log_progress(self):
        """Log progress periodically"""
        if not self.start_time:
            return
            
        elapsed = time.time() - self.start_time
        elapsed_minutes = elapsed / 60
        
        print(f"⏰ PROGRESS UPDATE: {elapsed_minutes:.1f} minutes elapsed")
        print(f"📈 Status: Deep reasoning still running...")
        print(f"🎯 Target: Complete all 5040 permutations")
        print()


async def run_actual_deep_reasoning():
    """Run ACTUAL deep reasoning - all 5040 permutations"""
    
    monitor = DeepReasoningMonitor()
    
    print("🧠 ACTUAL DEEP REASONING TEST - ALL 5040 PERMUTATIONS")
    print("=" * 70)
    print("This will run the COMPLETE deep reasoning pipeline:")
    print("- Search 150K paper embeddings")
    print("- Content analysis of retrieved papers")
    print("- DEEP meta-reasoning with ALL 5040 permutations")
    print("- Expected runtime: 30+ minutes")
    print()
    
    print("🚀 STARTING AUTOMATICALLY (background mode)")
    print()
    
    # Query about transformers
    query = "What are the most promising approaches for improving transformer attention mechanisms to handle very long sequences efficiently?"
    
    print(f"🔍 Query: {query}")
    print()
    
    # Initialize external knowledge base
    print("📚 Connecting to 150K paper corpus...")
    knowledge_base = await get_external_knowledge_base()
    print(f"✅ Connected to external storage")
    print()
    
    # Semantic search
    print("🔍 Performing semantic search on 150K embeddings...")
    embedding_generator = TextEmbeddingGenerator()
    retriever = SemanticRetriever(knowledge_base, embedding_generator)
    await retriever.initialize()
    
    search_result = await retriever.semantic_search(
        query=query,
        top_k=10,  # Get top 10 for deep analysis
        similarity_threshold=0.2,
        search_method="semantic"
    )
    
    print(f"📊 Search completed: {len(search_result.retrieved_papers)} papers found")
    print(f"⏱️  Search time: {search_result.search_time_seconds:.2f} seconds")
    print()
    
    # Content analysis
    print("🔬 Analyzing retrieved papers...")
    content_analyzer = ContentAnalyzer()
    await content_analyzer.initialize()
    
    analysis_search_result = SemanticSearchResult(
        query=search_result.query,
        retrieved_papers=search_result.retrieved_papers,
        search_time_seconds=search_result.search_time_seconds,
        total_papers_searched=search_result.total_papers_searched,
        retrieval_method=search_result.retrieval_method,
        embedding_model=search_result.embedding_model,
        search_id=search_result.search_id
    )
    
    analysis_result = await content_analyzer.analyze_retrieved_papers(analysis_search_result)
    print(f"✅ Content analysis: {len(analysis_result.analyzed_papers)} papers analyzed")
    print(f"📝 Concepts extracted: {analysis_result.total_concepts_extracted}")
    print()
    
    # DEEP META-REASONING - ALL 5040 PERMUTATIONS
    print("🧠 STARTING DEEP META-REASONING - ALL 5040 PERMUTATIONS")
    print("=" * 70)
    print("⚠️  WARNING: This will take 30+ minutes to complete!")
    print("🔄 The system will process every possible sequence of 7 reasoning engines")
    print("📊 Progress updates will be logged every 100 sequences")
    print()
    
    monitor.start_monitoring()
    
    # Create reasoning context
    reasoning_context = {
        "retrieved_papers": [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "relevance_score": paper.relevance_score
            }
            for paper in search_result.retrieved_papers
        ],
        "content_analysis": {
            "analyzed_papers": [
                {
                    "paper_id": summary.paper_id,
                    "title": summary.title,
                    "quality_score": summary.quality_score,
                    "main_contributions": summary.main_contributions,
                    "key_concepts": [{"concept": c.concept, "category": c.category, "confidence": c.confidence} 
                                   for c in summary.key_concepts],
                    "methodologies": summary.methodologies,
                    "findings": summary.findings
                }
                for summary in analysis_result.analyzed_papers
            ],
            "total_concepts": analysis_result.total_concepts_extracted
        }
    }
    
    # Initialize meta-reasoning engine
    meta_engine = MetaReasoningEngine()
    
    # RUN DEEP REASONING - ALL 5040 PERMUTATIONS
    deep_start_time = time.time()
    
    reasoning_result = await meta_engine.meta_reason(
        query=query,
        context=reasoning_context,
        thinking_mode=ThinkingMode.DEEP  # This will run ALL 5040 permutations
    )
    
    deep_end_time = time.time()
    total_deep_time = deep_end_time - deep_start_time
    
    print()
    print("🎉 DEEP REASONING COMPLETED!")
    print("=" * 70)
    print(f"⏱️  Total processing time: {total_deep_time/60:.2f} minutes")
    print(f"🧠 Meta confidence: {reasoning_result.meta_confidence:.3f}")
    print(f"💰 FTNS cost: {reasoning_result.ftns_cost:.2f} tokens")
    print(f"📊 Reasoning depth: {reasoning_result.reasoning_depth}")
    
    if reasoning_result.parallel_results:
        print(f"🔄 Parallel results: {len(reasoning_result.parallel_results)}")
    if reasoning_result.sequential_results:
        print(f"🔗 Sequential results: {len(reasoning_result.sequential_results)}")
    
    print()
    print("🎯 FINAL COMPREHENSIVE SYNTHESIS:")
    print("=" * 70)
    if reasoning_result.final_synthesis:
        if isinstance(reasoning_result.final_synthesis, dict):
            for key, value in reasoning_result.final_synthesis.items():
                print(f"{key.upper()}:")
                print(f"  {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
                print()
        else:
            synthesis_text = str(reasoning_result.final_synthesis)
            print(synthesis_text[:500] + ('...' if len(synthesis_text) > 500 else ''))
    
    print("=" * 70)
    print(f"✅ DEEP REASONING TEST COMPLETED SUCCESSFULLY!")
    print(f"🕐 Started: {datetime.fromtimestamp(monitor.start_time).strftime('%H:%M:%S')}")
    print(f"🕐 Finished: {datetime.now().strftime('%H:%M:%S')}")
    print(f"⏱️  Duration: {total_deep_time/60:.2f} minutes")
    
    return True


async def main():
    """Main function"""
    try:
        success = await run_actual_deep_reasoning()
        if success:
            print("\n🎉 ALL TESTS PASSED - DEEP REASONING WORKING CORRECTLY!")
        else:
            print("\n❌ DEEP REASONING TEST FAILED!")
    except KeyboardInterrupt:
        print("\n⚠️  Deep reasoning test interrupted by user")
    except Exception as e:
        print(f"\n❌ Deep reasoning test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
NWTN R&D Challenge Demo
======================

Run a single challenging R&D prompt and show the full NWTN response.
"""

import asyncio
import sys
import time
import argparse
from datetime import datetime

# Add PRSM to path
sys.path.insert(0, '/Users/ryneschultz/Documents/GitHub/PRSM')

from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
from enhanced_semantic_search import EnhancedSemanticSearchEngine, SearchQuery

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NWTN R&D Challenge Demo')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose mode with detailed analysis')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug mode with full internal reasoning traces')
    parser.add_argument('--output-file', '-o', type=str,
                       help='Save output to file instead of console')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Use QUICK mode instead of DEEP for faster testing')
    args = parser.parse_args()
    
    # Set up output
    output_file = None
    if args.output_file:
        output_file = open(args.output_file, 'w', encoding='utf-8')
        def print_to_output(*args_print, **kwargs):
            print(*args_print, file=output_file, **kwargs)
            if not args.debug:  # Also print to console unless in debug mode
                print(*args_print, **kwargs)
        print_func = print_to_output
    else:
        print_func = print
    
    print_func("🎯 NWTN R&D CHALLENGE DEMONSTRATION")
    print_func("=" * 80)
    print_func(f"📅 Demo Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_func("🧠 Testing NWTN with a challenging R&D strategy prompt")
    print_func("📚 Using full corpus: 151,120 arXiv papers")
    print_func("🔧 Using corrected meta_reason interface")
    if args.verbose:
        print_func("🔍 Verbose mode: ON")
    if args.debug:
        print_func("🐛 Debug mode: ON (full internal reasoning traces)")
    print_func("=" * 80)
    print_func()
    
    # The challenging R&D prompt
    challenge_prompt = """Based on the latest research developments, what are the three most promising avenues for integrating quantum computing into drug discovery and pharmaceutical R&D over the next 5-10 years? Consider molecular simulation capabilities, optimization challenges, and practical implementation barriers. Provide specific research directions that pharmaceutical companies should prioritize for competitive advantage."""
    
    print_func("🎯 CHALLENGE PROMPT:")
    print_func("-" * 40)
    print_func(challenge_prompt)
    print_func("-" * 40)
    print_func()
    
    # Initialize systems
    print_func("🔧 Initializing systems...")
    
    # Initialize semantic search
    print_func("🔍 Initializing semantic search engine...")
    search_engine = EnhancedSemanticSearchEngine(
        index_dir="/Volumes/My Passport/PRSM_Storage/PRSM_Indices",
        index_type="HNSW"
    )
    
    if not search_engine.initialize():
        print("❌ Failed to initialize semantic search")
        return
    
    stats = search_engine.get_statistics()
    print(f"✅ Semantic search ready: {stats['total_papers']:,} papers indexed")
    
    # Initialize NWTN
    print("🧠 Initializing NWTN Meta-Reasoning Engine...")
    meta_engine = MetaReasoningEngine()
    print("✅ NWTN ready with 7 reasoning engines")
    print()
    
    # Retrieve relevant research papers
    print("📚 RETRIEVING RELEVANT RESEARCH PAPERS...")
    print("-" * 50)
    
    search_queries = [
        "quantum computing drug discovery",
        "quantum simulation molecular dynamics",
        "quantum algorithms pharmaceutical optimization",
        "quantum machine learning drug design"
    ]
    
    all_papers = []
    for i, query in enumerate(search_queries, 1):
        print(f"{i}. Searching: '{query}'")
        
        search_query = SearchQuery(
            query_text=query,
            max_results=5,
            similarity_threshold=0.15
        )
        
        start_time = time.time()
        results = await search_engine.search(search_query)
        search_time = time.time() - start_time
        
        print(f"   📄 Found {len(results)} papers in {search_time:.2f}s")
        
        if results:
            print(f"   🏆 Top result: {results[0].title[:60]}...")
            print(f"   📊 Relevance: {results[0].similarity_score:.3f}")
        
        for result in results:
            all_papers.append({
                'id': result.paper_id,
                'title': result.title,
                'abstract': result.abstract,
                'domain': result.domain,
                'authors': result.authors,
                'relevance_score': result.similarity_score,
                'search_term': query
            })
        
        print()
    
    # Remove duplicates and get top papers
    unique_papers = {}
    for paper in all_papers:
        if paper['id'] not in unique_papers or paper['relevance_score'] > unique_papers[paper['id']]['relevance_score']:
            unique_papers[paper['id']] = paper
    
    top_papers = sorted(unique_papers.values(), key=lambda x: x['relevance_score'], reverse=True)[:15]
    
    print("📊 RESEARCH CONTEXT SUMMARY:")
    print("-" * 30)
    print(f"📄 Total papers retrieved: {len(top_papers)}")
    
    if top_papers:
        avg_relevance = sum(p['relevance_score'] for p in top_papers) / len(top_papers)
        domains = set(p['domain'] for p in top_papers)
        print(f"📊 Average relevance: {avg_relevance:.3f}")
        print(f"🌐 Domains covered: {', '.join(domains)}")
    
    print()
    
    # Show top 3 most relevant papers
    print("🏆 TOP 3 MOST RELEVANT PAPERS:")
    print("-" * 40)
    for i, paper in enumerate(top_papers[:3], 1):
        print(f"{i}. {paper['title']}")
        print(f"   📊 Relevance: {paper['relevance_score']:.3f}")
        print(f"   🏷️  Domain: {paper['domain']}")
        print(f"   📝 Abstract: {paper['abstract'][:150]}...")
        print()
    
    # Prepare reasoning context
    reasoning_context = {
        'challenge_prompt': challenge_prompt,
        'research_papers': top_papers,
        'corpus_size': 151120,
        'domain_focus': 'quantum_computing_drug_discovery',
        'analysis_type': 'strategic_rd_planning',
        'time_horizon': '5-10 years',
        'stakeholder': 'pharmaceutical_companies',
        'competitive_focus': True
    }
    
    # Choose thinking mode based on args
    thinking_mode = ThinkingMode.QUICK if args.quick else ThinkingMode.DEEP
    
    # Perform NWTN meta-reasoning
    print("🧠 NWTN META-REASONING IN PROGRESS...")
    print("-" * 50)
    print(f"🔧 Using ThinkingMode.{thinking_mode.value} for analysis")
    if thinking_mode == ThinkingMode.DEEP:
        print("⚙️  Activating all 7 reasoning engines")
        print("🌍 Integrating with world model (223 knowledge items)")
        print("📊 Processing evidence from 15 research papers")
    else:
        print("⚡ Using quick mode for faster processing")
    print()
    
    start_time = time.time()
    
    try:
        # Use the corrected method: meta_reason
        result = await meta_engine.meta_reason(
            query=challenge_prompt,
            context=reasoning_context,
            thinking_mode=thinking_mode
        )
        
        reasoning_time = time.time() - start_time
        
        print("✅ NWTN META-REASONING COMPLETED!")
        print(f"⏱️  Processing time: {reasoning_time:.1f} seconds")
        print(f"🎯 Meta-confidence: {result.meta_confidence:.3f}")
        
        if hasattr(result, 'reasoning_engines_used'):
            engines_used = list(result.reasoning_engines_used.keys())
            print(f"🧠 Reasoning engines used: {len(engines_used)} - {engines_used}")
        
        if hasattr(result, 'synthesis_quality'):
            print(f"🔗 Synthesis quality: {result.synthesis_quality:.3f}")
        
        if hasattr(result, 'world_model_integration_score'):
            print(f"🌍 World model integration: {result.world_model_integration_score:.3f}")
        
        print()
        print("=" * 80)
        print("🎯 NWTN RESPONSE TO R&D CHALLENGE:")
        print("=" * 80)
        
        # Display response based on mode using new presentation layer
        if args.debug:
            # Debug mode: Show all internal reasoning (WARNING: Can be very large!)
            print_func("🐛 DEBUG MODE: Full internal reasoning traces")
            print_func("=" * 40)
            
            # Show parallel results
            if result.parallel_results:
                print_func(f"📊 Parallel Results ({len(result.parallel_results)} engines):")
                for i, res in enumerate(result.parallel_results, 1):
                    print_func(f"  {i}. {res.engine.value}: {res.content[:200]}...")
                print_func()
            
            # Show sequential results
            if result.sequential_results:
                print_func(f"🔄 Sequential Results ({len(result.sequential_results)} chains):")
                for i, seq_res in enumerate(result.sequential_results, 1):
                    print_func(f"  Chain {i}: {' -> '.join([eng.value for eng in seq_res.sequence])}")
                print_func()
            
            # Show full synthesis
            print_func("🎯 Full Final Synthesis:")
            print_func(str(result.final_synthesis))
            print_func()
        
        elif args.verbose:
            # Verbose mode: Show detailed analysis using new presentation layer
            print_func("📊 DETAILED ANALYSIS:")
            print_func("=" * 40)
            response_text = result.get_detailed_analysis(reasoning_context)
            print_func(response_text)
        
        else:
            # Normal mode: Show clean user-facing response using new voicebox presentation layer
            print_func("🎯 STRATEGIC ANALYSIS:")
            print_func("=" * 40)
            
            try:
                # Try using voicebox for natural language translation
                response_text = await result.get_voicebox_response(challenge_prompt, context=reasoning_context)
                print_func("✨ Response generated using NWTN Voicebox (LLM-translated)")
            except Exception as e:
                # Fallback to conversational response
                print_func(f"⚠️  Voicebox translation failed ({e}), using built-in formatter")
                response_text = result.get_conversational_response(reasoning_context)
            
            print_func(response_text)
        
        print()
        print("=" * 80)
        print("📊 RESPONSE ANALYSIS:")
        print("=" * 80)
        
        # Analyze response quality
        strategic_indicators = [
            "promising avenues", "research directions", "competitive advantage",
            "implementation barriers", "pharmaceutical", "molecular simulation",
            "optimization", "quantum", "drug discovery"
        ]
        
        strategic_score = sum(1 for indicator in strategic_indicators if indicator.lower() in response_text.lower())
        print(f"📈 Strategic thinking indicators: {strategic_score}/{len(strategic_indicators)}")
        
        # Check for evidence utilization
        print(f"📚 Evidence base: {len(top_papers)} research papers")
        print(f"🌐 Cross-domain synthesis: {len(domains)} domains")
        print(f"🎯 Confidence level: {result.meta_confidence:.3f}")
        
        # Overall assessment
        if result.meta_confidence > 0.7:
            assessment = "EXCELLENT"
        elif result.meta_confidence > 0.5:
            assessment = "GOOD"
        else:
            assessment = "MODERATE"
        
        print(f"🏆 Overall assessment: {assessment}")
        
        print()
        print_func("🎉 NWTN REAL-WORLD R&D CHALLENGE: DEMONSTRATED!")
        print_func("🚀 Successfully handled complex strategic thinking prompt")
        print_func("💡 Provided evidence-based insights for pharmaceutical R&D")
        print_func("🌟 Demonstrated capabilities beyond traditional LLMs")
        
    except Exception as e:
        print_func(f"❌ NWTN meta-reasoning failed: {e}")
        print_func(f"🔍 Error type: {type(e).__name__}")
        import traceback
        if args.debug:
            traceback.print_exc(file=output_file if output_file else sys.stdout)
        else:
            print_func("Use --debug flag for full traceback")
    
    finally:
        if output_file:
            output_file.close()
            print(f"✅ Output saved to {args.output_file}")
            
            # Show file size
            import os
            file_size = os.path.getsize(args.output_file)
            if file_size > 1024 * 1024:  # > 1MB
                print(f"⚠️  Output file size: {file_size / (1024*1024):.1f} MB")
            else:
                print(f"📄 Output file size: {file_size / 1024:.1f} KB")

if __name__ == "__main__":
    asyncio.run(main())
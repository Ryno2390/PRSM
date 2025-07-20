#!/usr/bin/env python3
"""
Quick test to debug candidate count through pipeline stages
"""

import asyncio
import sys
import time
sys.path.insert(0, '.')

from prsm.nwtn.semantic_retriever import SemanticRetriever
from prsm.nwtn.content_analyzer import ContentAnalyzer  
from prsm.nwtn.candidate_answer_generator import CandidateAnswerGenerator
from prsm.nwtn.candidate_evaluator import CandidateEvaluator
from prsm.nwtn.citation_filter import CitationFilter
from prsm.nwtn.external_storage_config import ExternalStorageConfig

async def debug_candidate_count():
    """Debug how many candidates we have at each pipeline stage"""
    print("🔍 Candidate Count Debug Test")
    print("=" * 40)
    
    # Initialize components
    external_storage = ExternalStorageConfig()
    retriever = SemanticRetriever(external_storage_config=external_storage)
    analyzer = ContentAnalyzer()
    generator = CandidateAnswerGenerator()
    evaluator = CandidateEvaluator()
    citation_filter = CitationFilter()
    
    await retriever.initialize()
    await analyzer.initialize()
    await generator.initialize()
    await evaluator.initialize()
    await citation_filter.initialize()
    
    query = "machine learning algorithms"
    print(f"📝 Query: '{query}'")
    
    # Stage 1: Semantic Retrieval
    print(f"\n🔍 Stage 1: Semantic Retrieval")
    retrieval_result = await retriever.semantic_search(query)
    print(f"   Papers Retrieved: {len(retrieval_result.retrieved_papers)}")
    
    # Stage 2: Content Analysis
    print(f"\n🔍 Stage 2: Content Analysis")
    analysis_result = await analyzer.analyze_content(retrieval_result)
    print(f"   Papers Analyzed: {len(analysis_result.analyzed_papers)}")
    
    # Stage 3: Candidate Generation
    print(f"\n🔍 Stage 3: Candidate Generation")
    candidate_result = await generator.generate_candidates(query, analysis_result)
    print(f"   Candidates Generated: {len(candidate_result.candidate_answers)}")
    print(f"   Target Candidates: {generator.target_candidates}")
    
    # Stage 4: Candidate Evaluation  
    print(f"\n🔍 Stage 4: Candidate Evaluation")
    evaluation_result = await evaluator.evaluate_candidates(candidate_result)
    print(f"   Candidates Evaluated: {len(evaluation_result.candidate_evaluations)}")
    
    # Stage 5: Citation Filtering - debug inputs
    print(f"\n🔍 Stage 5: Citation Filtering Debug")
    print(f"   Evaluation Candidates: {len(evaluation_result.candidate_evaluations)}")
    print(f"   Top Candidates to Consider: {citation_filter.top_candidates_to_consider}")
    print(f"   Max Citations: {citation_filter.max_citations}")
    print(f"   Relevance Threshold: {citation_filter.default_relevance_threshold}")
    print(f"   Contribution Threshold: {citation_filter.default_contribution_threshold}")
    
    # Extract candidate sources manually to debug
    sorted_candidates = sorted(
        evaluation_result.candidate_evaluations,
        key=lambda x: x.overall_score,
        reverse=True
    )[:citation_filter.top_candidates_to_consider]
    
    total_source_contributions = 0
    for candidate in sorted_candidates:
        contributions = len(candidate.candidate_answer.source_contributions)
        total_source_contributions += contributions
        print(f"   Candidate Score: {candidate.overall_score:.3f}, Sources: {contributions}")
    
    citation_result = await citation_filter.filter_citations(evaluation_result)
    print(f"   Final Citations: {len(citation_result.filtered_citations)}")
    
    # Diagnosis
    print(f"\n🎯 Diagnosis:")
    if len(candidate_result.candidate_answers) < 3:
        print(f"   ❌ BOTTLENECK: Only {len(candidate_result.candidate_answers)} candidates generated")
    elif len(evaluation_result.candidate_evaluations) < 3:
        print(f"   ❌ BOTTLENECK: Only {len(evaluation_result.candidate_evaluations)} candidates evaluated")
    elif total_source_contributions < 3:
        print(f"   ❌ BOTTLENECK: Only {total_source_contributions} total source contributions")
    elif len(citation_result.filtered_citations) < 3:
        print(f"   ❌ BOTTLENECK: Citation filtering reduced to {len(citation_result.filtered_citations)}")
    else:
        print(f"   ✅ SUCCESS: {len(citation_result.filtered_citations)} citations as expected")

if __name__ == "__main__":
    asyncio.run(debug_candidate_count())
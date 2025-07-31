#!/usr/bin/env python3
"""
Test Citations Functionality Only
================================
Tests just the paper citation formatting with mock data
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from prsm.nwtn.enhanced_orchestrator import EnhancedNWTNOrchestrator
from prsm.nwtn.semantic_retriever import RetrievedPaper, SemanticSearchResult
from datetime import datetime, timezone

print("🧪 TESTING CITATIONS FUNCTIONALITY")
print("=" * 40)

async def test_citation_formatting():
    """Test the citation formatting with mock papers"""
    
    # Create mock retrieved papers
    mock_papers = [
        RetrievedPaper(
            paper_id="paper_1",
            title="Machine Learning Applications in Climate Modeling",
            authors="Smith, J., Johnson, A., Williams, K.",
            abstract="This paper explores various machine learning techniques for climate prediction...",
            arxiv_id="2024.12345",
            publish_date="2024-01-15",
            relevance_score=0.95,
            similarity_score=0.87
        ),
        RetrievedPaper(
            paper_id="paper_2", 
            title="Deep Neural Networks for Weather Pattern Recognition",
            authors="Brown, M., Davis, R., Miller, S.",
            abstract="We present a comprehensive study of deep learning approaches to weather forecasting...",
            arxiv_id="2024.67890",
            publish_date="2024-03-22",
            relevance_score=0.88,
            similarity_score=0.82
        ),
        RetrievedPaper(
            paper_id="paper_3",
            title="Time Series Analysis of Global Temperature Trends Using AI",
            authors="Anderson, P., Wilson, T., Garcia, L.",
            abstract="This research investigates AI-based time series methods for climate analysis...",
            arxiv_id="2024.11111",
            publish_date="2024-02-10",
            relevance_score=0.92,
            similarity_score=0.85
        )
    ]
    
    # Create mock retrieval result
    mock_retrieval_result = SemanticSearchResult(
        query="machine learning climate change prediction",
        retrieved_papers=mock_papers,
        search_time_seconds=1.5,
        total_papers_searched=100000,
        retrieval_method="semantic_embedding",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("✅ Mock retrieval result created")
    print(f"   • Papers found: {len(mock_papers)}")
    print(f"   • Search time: {mock_retrieval_result.search_time_seconds}s")
    print()
    
    # Initialize orchestrator and test citation formatting
    orchestrator = EnhancedNWTNOrchestrator()
    print("✅ Enhanced orchestrator initialized")
    
    # Test the citation formatting function
    print("🔄 Testing citation formatting...")
    citations = orchestrator._format_paper_citations(mock_retrieval_result)
    
    print("📋 FORMATTED CITATIONS:")
    print("-" * 25)
    print(citations)
    print("-" * 50)
    print()
    
    # Validate citation content
    validation_checks = [
        ("Contains References section", "## References" in citations),
        ("Contains Works Cited section", "## Works Cited" in citations),
        ("Contains paper titles", all(paper.title in citations for paper in mock_papers)),
        ("Contains author names", all(paper.authors in citations for paper in mock_papers)),
        ("Contains arXiv IDs", all(paper.arxiv_id in citations for paper in mock_papers)),
        ("Contains publication dates", all(paper.publish_date in citations for paper in mock_papers)),
        ("Contains relevance scores", "Relevance Score:" in citations),
        ("Citations not empty", len(citations.strip()) > 50)
    ]
    
    print("🎯 CITATION VALIDATION:")
    print("-" * 25)
    all_passed = True
    for check_name, passed in validation_checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    print()
    print(f"📊 VALIDATION SUMMARY:")
    print(f"   • Total checks: {len(validation_checks)}")
    print(f"   • Passed: {sum(1 for _, passed in validation_checks if passed)}")
    print(f"   • Failed: {sum(1 for _, passed in validation_checks if not passed)}")
    print(f"   • Citation length: {len(citations)} characters")
    print()
    
    if all_passed:
        print("🎉 SUCCESS: All citation formatting tests passed!")
        return True
    else:
        print("❌ FAILURE: Some citation formatting tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_citation_formatting())
    if success:
        print("\n🎯 CITATION FUNCTIONALITY TEST: PASSED ✅")
    else:
        print("\n💥 CITATION FUNCTIONALITY TEST: FAILED ❌")
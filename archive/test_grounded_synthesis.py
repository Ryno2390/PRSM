#!/usr/bin/env python3
"""
Test Content Grounding System
==============================

This script tests the integrated content grounding system to ensure Claude responses
are properly grounded in actual paper content rather than hallucinated knowledge.

Usage:
    python test_grounded_synthesis.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from prsm.nwtn.content_grounding_synthesizer import ContentGroundingSynthesizer, GroundedPaperContent
from prsm.nwtn.voicebox import NWTNVoicebox
from prsm.nwtn.meta_reasoning_engine import MetaReasoningResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockExternalKnowledgeBase:
    """Mock external knowledge base for testing"""
    
    def __init__(self):
        self.initialized = True
        self.storage_manager = self
        self.storage_db = self
        
        # Mock paper database
        self.papers = [
            {
                'title': 'Atomically Precise Manufacturing: A Revolutionary Approach',
                'abstract': 'This paper explores the potential of atomically precise manufacturing (APM) to revolutionize production processes. We demonstrate novel techniques for assembling materials at the atomic level with unprecedented precision and control.',
                'authors': 'Smith, J., Johnson, M., Lee, K.',
                'arxiv_id': '2301.12345',
                'publish_date': '2023-01-15',
                'categories': ['nanotechnology', 'materials-science'],
                'domain': 'nanotechnology',
                'journal_ref': 'Nature Nanotechnology 15, 234-245 (2023)'
            },
            {
                'title': 'Enzymatic Catalysis in Molecular Manufacturing Systems',
                'abstract': 'We present a comprehensive study of enzymatic catalysis mechanisms that could be applied to molecular manufacturing systems. Our findings suggest significant potential for bio-inspired manufacturing processes.',
                'authors': 'Chen, L., Rodriguez, A., Patel, S.',
                'arxiv_id': '2302.67890',
                'publish_date': '2023-02-20',
                'categories': ['biotechnology', 'materials-science'],
                'domain': 'biotechnology',
                'journal_ref': 'Journal of Molecular Biology 425, 1234-1250 (2023)'
            }
        ]
    
    def cursor(self):
        return self
    
    def execute(self, query, params=None):
        # Mock database query execution
        if "WHERE arxiv_id = ?" in query and params:
            arxiv_id = params[0]
            for paper in self.papers:
                if paper['arxiv_id'] == arxiv_id:
                    self.result = [(
                        paper['title'],
                        paper['abstract'],
                        paper['authors'],
                        paper['arxiv_id'],
                        paper['publish_date'],
                        ','.join(paper['categories']),
                        paper['domain'],
                        paper['journal_ref']
                    )]
                    return
        elif "categories LIKE" in query and params:
            # Return papers matching category
            self.result = []
            for paper in self.papers:
                row = (
                    paper['title'],
                    paper['abstract'],
                    paper['authors'],
                    paper['arxiv_id'],
                    paper['publish_date'],
                    ','.join(paper['categories']),
                    paper['domain'],
                    paper['journal_ref']
                )
                self.result.append(row)
            return
        
        self.result = []
    
    def fetchone(self):
        return self.result[0] if hasattr(self, 'result') and self.result else None
    
    def fetchall(self):
        return getattr(self, 'result', [])

class MockMetaReasoningResult:
    """Mock reasoning result for testing"""
    
    def __init__(self):
        self.meta_confidence = 0.85
        self.reasoning_path = ['analogical', 'causal', 'network_validation']
        self.integrated_conclusion = 'Multi-modal reasoning analysis suggests significant potential for breakthrough applications'
        self.multi_modal_evidence = ['Evidence 1', 'Evidence 2', 'Evidence 3']
        self.identified_uncertainties = ['Manufacturing scalability challenges', 'Economic feasibility concerns']
        self.reasoning_results = [
            {'engine': 'analogical', 'confidence': 0.9},
            {'engine': 'causal', 'confidence': 0.8}
        ]
        self.summary = 'NWTN completed comprehensive analysis using analogical and causal reasoning'

async def test_content_grounding_synthesizer():
    """Test the ContentGroundingSynthesizer with mock data"""
    
    logger.info("🧪 Testing ContentGroundingSynthesizer...")
    
    # Create mock external knowledge base
    mock_kb = MockExternalKnowledgeBase()
    
    # Create content grounding synthesizer
    synthesizer = ContentGroundingSynthesizer(mock_kb)
    
    # Create mock reasoning result
    reasoning_result = MockMetaReasoningResult()
    
    # Mock retrieved papers
    retrieved_papers = [
        {'arxiv_id': '2301.12345', 'score': 0.9},
        {'arxiv_id': '2302.67890', 'score': 0.8}
    ]
    
    # Test grounded synthesis preparation
    verbosity_levels = ["BRIEF", "STANDARD", "DETAILED", "COMPREHENSIVE", "ACADEMIC"]
    
    for verbosity_level in verbosity_levels:
        logger.info(f"Testing {verbosity_level} verbosity level...")
        
        # Get target tokens for verbosity level
        target_tokens = {
            "BRIEF": 500,
            "STANDARD": 1000,
            "DETAILED": 2000,
            "COMPREHENSIVE": 3500,
            "ACADEMIC": 4000
        }[verbosity_level]
        
        # Prepare grounded synthesis
        grounding_result = await synthesizer.prepare_grounded_synthesis(
            reasoning_result=reasoning_result,
            target_tokens=target_tokens,
            retrieved_papers=retrieved_papers,
            verbosity_level=verbosity_level
        )
        
        logger.info(f"✅ {verbosity_level} grounding completed",
                   extra={
                       'source_papers': len(grounding_result.source_papers),
                       'content_tokens': grounding_result.content_tokens_estimate,
                       'grounding_quality': grounding_result.grounding_quality,
                       'expansion_available': grounding_result.available_expansion_content
                   })
        
        # Verify grounding result
        assert grounding_result.source_papers, f"No source papers for {verbosity_level}"
        assert grounding_result.grounded_content, f"No grounded content for {verbosity_level}"
        assert grounding_result.grounding_quality > 0, f"Zero grounding quality for {verbosity_level}"
        
        # Check that grounded content includes actual paper abstracts
        grounded_content = grounding_result.grounded_content
        assert "Atomically Precise Manufacturing" in grounded_content, "Paper 1 title not found"
        assert "Enzymatic Catalysis" in grounded_content, "Paper 2 title not found"
        assert "revolutionize production processes" in grounded_content, "Paper 1 abstract not found"
        assert "enzymatic catalysis mechanisms" in grounded_content, "Paper 2 abstract not found"
        
        logger.info(f"📋 Sample grounded content for {verbosity_level}:")
        logger.info(grounded_content[:500] + "...")
    
    logger.info("✅ ContentGroundingSynthesizer tests passed!")

async def test_integrated_voicebox_grounding():
    """Test the integrated VoiceBox with grounding"""
    
    logger.info("🧪 Testing integrated VoiceBox grounding...")
    
    try:
        # Create VoiceBox instance (but don't initialize all dependencies for test)
        voicebox = NWTNVoicebox()
        
        # Test helper methods
        target_tokens = voicebox._get_target_tokens_for_verbosity("COMPREHENSIVE")
        assert target_tokens == 3500, f"Expected 3500 tokens, got {target_tokens}"
        
        # Test paper extraction
        mock_result = MockMetaReasoningResult()
        mock_result.content_sources = ["Paper 1 by Author A", "Paper 2 by Author B"]
        
        extracted_papers = voicebox._extract_retrieved_papers(mock_result)
        assert len(extracted_papers) == 2, f"Expected 2 papers, got {len(extracted_papers)}"
        assert extracted_papers[0]['title'] == "Paper 1", f"Wrong title: {extracted_papers[0]['title']}"
        assert extracted_papers[0]['authors'] == "Author A", f"Wrong authors: {extracted_papers[0]['authors']}"
        
        # Test works cited generation
        grounded_papers = [
            type('Paper', (), {
                'authors': 'Smith, J.',
                'title': 'Test Paper',
                'arxiv_id': '2301.12345',
                'publish_date': '2023-01-15'
            })()
        ]
        
        works_cited = voicebox._generate_works_cited_from_grounded_papers(grounded_papers)
        assert "Smith, J. (2023). Test Paper. arXiv:2301.12345." in works_cited, f"Wrong citation format: {works_cited}"
        
        logger.info("✅ Integrated VoiceBox grounding tests passed!")
        
    except ImportError as e:
        logger.warning(f"⚠️  Skipping VoiceBox integration test due to dependency: {e}")

async def main():
    """Run all content grounding tests"""
    logger.info("🚀 Starting Content Grounding System Tests")
    
    try:
        # Test content grounding synthesizer
        await test_content_grounding_synthesizer()
        
        # Test integrated voicebox grounding
        await test_integrated_voicebox_grounding()
        
        logger.info("🎉 All content grounding tests passed!")
        logger.info("📝 Content grounding system is ready to prevent hallucinations")
        
    except Exception as e:
        logger.error(f"❌ Content grounding tests failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
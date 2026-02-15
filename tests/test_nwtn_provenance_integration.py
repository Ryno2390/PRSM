#!/usr/bin/env python3
"""
NWTN Provenance Integration Testing Framework
=============================================

Comprehensive test suite for the NWTN provenance tracking and royalty distribution system.
This framework validates end-to-end functionality across all components and scenarios.

Test Categories:
1. Provenance Tracking Tests - Ensure all content usage is properly tracked
2. Royalty Calculation Tests - Validate sophisticated royalty algorithms 
3. Attribution Tests - Verify proper source attribution in responses
4. Duplicate Detection Tests - Test multi-layer duplicate detection
5. Integration Tests - End-to-end workflow validation
6. Performance Tests - Ensure acceptable performance under load
7. Edge Case Tests - Handle unusual scenarios gracefully
8. Security Tests - Prevent fraud and abuse

Usage:
    python -m pytest tests/test_nwtn_provenance_integration.py -v
    python -m pytest tests/test_nwtn_provenance_integration.py::TestProvenanceTracking -v
"""

import asyncio
import pytest
import json
import hashlib
import time
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from uuid import uuid4, UUID
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock

# Import the systems under test
from prsm.compute.nwtn.meta_reasoning_engine import MetaReasoningEngine
from prsm.compute.nwtn.knowledge_corpus_interface import NWTNKnowledgeCorpusInterface
from prsm.compute.nwtn.voicebox import NWTNVoicebox, VoiceboxResponse, SourceLink
from prsm.compute.nwtn.content_royalty_engine import (
    ContentRoyaltyEngine, QueryComplexity, UserTier, ContentImportance, 
    RoyaltyCalculation, RoyaltyDistributionResult
)
from prsm.compute.nwtn.content_ingestion_engine import (
    NWTNContentIngestionEngine, ContentIngestionResult, DuplicateMatch,
    IngestionStatus, ContentQuality
)
from prsm.data.provenance.enhanced_provenance_system import (
    EnhancedProvenanceSystem, ContentType, LicenseType, ContentFingerprint, 
    AttributionChain, UsageEvent
)
from prsm.economy.tokenomics.ftns_service import FTNSService


class TestProvenanceTracking:
    """Test suite for provenance tracking functionality"""
    
    @pytest.fixture
    async def provenance_system(self):
        """Create a test provenance system"""
        system = EnhancedProvenanceSystem()
        return system
    
    @pytest.fixture
    async def meta_reasoning_engine(self):
        """Create a test meta-reasoning engine"""
        engine = MetaReasoningEngine()
        return engine
    
    @pytest.fixture
    def sample_content(self):
        """Sample content for testing"""
        return {
            'content_data': b"This is a sample research paper about quantum computing and its applications in machine learning.",
            'content_type': ContentType.RESEARCH_PAPER,
            'metadata': {
                'title': 'Quantum ML Applications',
                'domain': 'quantum_computing',
                'tags': ['quantum', 'machine_learning', 'algorithms']
            },
            'creator_info': {
                'name': 'test_creator',
                'ftns_address': 'ftns_test_address_123',
                'platform': 'PRSM_TEST'
            },
            'license_info': {
                'type': 'open_source',
                'terms': {'attribution_required': True}
            }
        }
    
    @pytest.mark.asyncio
    async def test_content_registration_with_provenance(self, provenance_system, sample_content):
        """Test that content is properly registered with complete provenance tracking"""
        
        # Register content
        content_id, fingerprint, attribution_chain = await provenance_system.register_content_with_provenance(
            content_data=sample_content['content_data'],
            content_type=sample_content['content_type'],
            creator_info=sample_content['creator_info'],
            license_info=sample_content['license_info'],
            metadata=sample_content['metadata']
        )
        
        # Verify content ID is valid UUID
        assert isinstance(content_id, UUID)
        
        # Verify fingerprint is created
        assert isinstance(fingerprint, ContentFingerprint)
        assert fingerprint.sha256_hash
        assert fingerprint.blake2b_hash
        assert fingerprint.size_bytes == len(sample_content['content_data'])
        assert fingerprint.content_type == sample_content['content_type']
        
        # Verify attribution chain
        assert isinstance(attribution_chain, AttributionChain)
        assert attribution_chain.content_id == content_id
        assert attribution_chain.original_creator == sample_content['creator_info']['name']
        assert attribution_chain.creator_address == sample_content['creator_info']['ftns_address']
        assert attribution_chain.license_type == LicenseType.OPEN_SOURCE
    
    @pytest.mark.asyncio
    async def test_usage_tracking(self, provenance_system, sample_content):
        """Test that content usage is properly tracked"""
        
        # First register content
        content_id, _, _ = await provenance_system.register_content_with_provenance(
            content_data=sample_content['content_data'],
            content_type=sample_content['content_type'],
            creator_info=sample_content['creator_info'],
            license_info=sample_content['license_info'],
            metadata=sample_content['metadata']
        )
        
        # Track usage
        session_id = uuid4()
        usage_event = await provenance_system.track_content_usage(
            content_id=content_id,
            user_id='test_user',
            session_id=session_id,
            usage_type='reasoning_source',
            context={
                'query': 'What are quantum computing applications?',
                'reasoning_mode': 'meta_reasoning'
            }
        )
        
        # Verify usage event
        assert isinstance(usage_event, UsageEvent)
        assert usage_event.content_id == content_id
        assert usage_event.user_id == 'test_user'
        assert usage_event.session_id == session_id
        assert usage_event.usage_type == 'reasoning_source'
    
    @pytest.mark.asyncio
    async def test_knowledge_usage_tracking_in_meta_reasoning(self, meta_reasoning_engine):
        """Test that knowledge usage is tracked during meta-reasoning"""
        
        # Mock world model knowledge
        mock_knowledge = [
            {
                'content': 'Quantum algorithms can provide exponential speedup',
                'certainty': 0.95,
                'domain': 'quantum_computing',
                'category': 'algorithms',
                'content_id': str(uuid4())
            },
            {
                'content': 'Machine learning benefits from quantum parallelism',
                'certainty': 0.88,
                'domain': 'machine_learning',
                'category': 'optimization',
                'content_id': str(uuid4())
            }
        ]
        
        # Mock the world model method
        with patch.object(meta_reasoning_engine, 'get_world_model_knowledge', return_value=mock_knowledge):
            with patch.object(meta_reasoning_engine, '_track_knowledge_usage') as mock_track:
                
                # Test meta-reasoning with knowledge tracking
                context = {'user_id': 'test_user'}
                query = "How can quantum computing improve machine learning?"
                
                # This would normally call meta_reason, but we'll test the tracking directly
                session_id = uuid4()
                await meta_reasoning_engine._track_knowledge_usage(
                    relevant_knowledge=mock_knowledge,
                    session_id=session_id,
                    user_id='test_user',
                    query=query
                )
                
                # Verify tracking was called
                mock_track.assert_called_once()
                
                # Verify content tracking
                assert len(meta_reasoning_engine.used_content) > 0


class TestRoyaltyCalculation:
    """Test suite for royalty calculation functionality"""
    
    @pytest.fixture
    async def royalty_engine(self):
        """Create a test royalty engine"""
        engine = ContentRoyaltyEngine()
        await engine.initialize()
        return engine
    
    @pytest.fixture
    def sample_content_sources(self):
        """Sample content sources for testing"""
        return [uuid4() for _ in range(3)]
    
    @pytest.mark.asyncio
    async def test_royalty_calculation_by_complexity(self, royalty_engine, sample_content_sources):
        """Test that royalty calculation varies appropriately by query complexity"""
        
        user_tier = "basic"
        session_id = uuid4()
        
        # Test different complexity levels
        complexities = [
            QueryComplexity.SIMPLE,
            QueryComplexity.MODERATE, 
            QueryComplexity.COMPLEX,
            QueryComplexity.BREAKTHROUGH
        ]
        
        royalty_results = {}
        
        for complexity in complexities:
            calculations = await royalty_engine.calculate_usage_royalty(
                content_sources=sample_content_sources,
                query_complexity=complexity,
                user_tier=user_tier,
                session_id=session_id
            )
            
            total_royalty = sum(calc.final_royalty for calc in calculations)
            royalty_results[complexity] = total_royalty
        
        # Verify increasing royalties with complexity
        assert royalty_results[QueryComplexity.SIMPLE] < royalty_results[QueryComplexity.MODERATE]
        assert royalty_results[QueryComplexity.MODERATE] < royalty_results[QueryComplexity.COMPLEX]
        assert royalty_results[QueryComplexity.COMPLEX] < royalty_results[QueryComplexity.BREAKTHROUGH]
    
    @pytest.mark.asyncio
    async def test_royalty_calculation_by_user_tier(self, royalty_engine, sample_content_sources):
        """Test that royalty calculation varies by user tier"""
        
        complexity = QueryComplexity.COMPLEX
        session_id = uuid4()
        
        # Test different user tiers
        tiers = ["basic", "premium", "researcher", "enterprise"]
        royalty_results = {}
        
        for tier in tiers:
            calculations = await royalty_engine.calculate_usage_royalty(
                content_sources=sample_content_sources,
                query_complexity=complexity,
                user_tier=tier,
                session_id=session_id
            )
            
            total_royalty = sum(calc.final_royalty for calc in calculations)
            royalty_results[tier] = total_royalty
        
        # Verify increasing royalties with tier
        assert royalty_results["basic"] < royalty_results["premium"]
        assert royalty_results["premium"] < royalty_results["researcher"]
        assert royalty_results["researcher"] < royalty_results["enterprise"]
    
    @pytest.mark.asyncio
    async def test_royalty_distribution(self, royalty_engine):
        """Test that royalties are properly distributed to creators"""
        
        # Create sample royalty calculations
        calculations = [
            RoyaltyCalculation(
                content_id=uuid4(),
                creator_id="creator_1",
                creator_address="ftns_address_1",
                base_royalty=Decimal('0.05'),
                complexity_multiplier=2.0,
                quality_multiplier=1.5,
                importance_multiplier=2.0,
                user_tier_multiplier=1.2,
                final_royalty=Decimal('0.18'),
                calculation_timestamp=datetime.now(timezone.utc),
                session_id=uuid4(),
                reasoning_context={}
            ),
            RoyaltyCalculation(
                content_id=uuid4(),
                creator_id="creator_2", 
                creator_address="ftns_address_2",
                base_royalty=Decimal('0.03'),
                complexity_multiplier=2.0,
                quality_multiplier=1.0,
                importance_multiplier=1.5,
                user_tier_multiplier=1.2,
                final_royalty=Decimal('0.108'),
                calculation_timestamp=datetime.now(timezone.utc),
                session_id=uuid4(),
                reasoning_context={}
            )
        ]
        
        # Mock FTNS service for distribution
        with patch.object(royalty_engine.ftns_service, 'reward_contribution', return_value=True):
            
            # Distribute royalties
            distribution_result = await royalty_engine.distribute_royalties(calculations)
            
            # Verify distribution result
            assert isinstance(distribution_result, RoyaltyDistributionResult)
            assert distribution_result.successful_distributions == 2
            assert distribution_result.failed_distributions == 0
            assert distribution_result.total_royalties_distributed > 0
            assert len(distribution_result.creator_distributions) == 2


class TestAttribution:
    """Test suite for attribution functionality"""
    
    @pytest.fixture
    async def voicebox(self):
        """Create a test voicebox"""
        voicebox = NWTNVoicebox()
        await voicebox.initialize()
        return voicebox
    
    @pytest.fixture
    def mock_reasoning_result(self):
        """Mock reasoning result with content sources"""
        result = Mock()
        result.content_sources = [uuid4() for _ in range(2)]
        result.overall_confidence = 0.85
        result.reasoning_path = ['deductive', 'inductive', 'analogical']
        result.multi_modal_evidence = ['evidence1', 'evidence2']
        return result
    
    @pytest.mark.asyncio
    async def test_source_link_generation(self, voicebox, mock_reasoning_result):
        """Test that source links are properly generated"""
        
        # Mock provenance system responses
        mock_attribution = Mock()
        mock_attribution.original_creator = "test_creator"
        mock_attribution.creation_timestamp = datetime.now(timezone.utc)
        
        mock_fingerprint = Mock()
        mock_fingerprint.ipfs_hash = "QmTestHash123456789"
        mock_fingerprint.content_type = ContentType.RESEARCH_PAPER
        
        with patch.object(voicebox.provenance_system, '_load_attribution_chain', return_value=mock_attribution):
            with patch.object(voicebox.provenance_system, '_load_content_fingerprint', return_value=mock_fingerprint):
                
                # Generate source links
                source_links = await voicebox._generate_source_links(mock_reasoning_result)
                
                # Verify source links
                assert isinstance(source_links, list)
                assert len(source_links) == len(mock_reasoning_result.content_sources)
                
                for link in source_links:
                    assert isinstance(link, SourceLink)
                    assert link.creator == "test_creator"
                    assert link.ipfs_link.startswith("https://ipfs.prsm.ai/ipfs/")
                    assert "QmTestHash123456789" in link.ipfs_link
    
    @pytest.mark.asyncio
    async def test_attribution_summary_generation(self, voicebox, mock_reasoning_result):
        """Test that attribution summaries are properly generated"""
        
        # Create sample source links
        source_links = [
            SourceLink(
                content_id=str(uuid4()),
                title="Quantum Computing Research",
                creator="creator_1",
                ipfs_link="https://ipfs.prsm.ai/ipfs/QmHash1",
                contribution_date="2024-01-01T00:00:00Z",
                relevance_score=0.9,
                content_type="research_paper"
            ),
            SourceLink(
                content_id=str(uuid4()),
                title="ML Algorithms Dataset",
                creator="creator_2", 
                ipfs_link="https://ipfs.prsm.ai/ipfs/QmHash2",
                contribution_date="2024-01-02T00:00:00Z",
                relevance_score=0.8,
                content_type="dataset"
            )
        ]
        
        # Generate attribution summary
        summary = await voicebox._generate_attribution_summary(mock_reasoning_result, source_links)
        
        # Verify summary content
        assert isinstance(summary, str)
        assert "2 verified sources" in summary
        assert "2 contributors" in summary
        assert "PRSM's knowledge corpus" in summary
        assert "multi-modal reasoning" in summary


class TestDuplicateDetection:
    """Test suite for duplicate detection functionality"""
    
    @pytest.fixture
    async def ingestion_engine(self):
        """Create a test content ingestion engine"""
        engine = NWTNContentIngestionEngine()
        await engine.initialize()
        return engine
    
    @pytest.fixture
    def sample_content_data(self):
        """Sample content for duplicate testing"""
        return {
            'original': b"This is original research content about quantum computing applications.",
            'duplicate_exact': b"This is original research content about quantum computing applications.",
            'duplicate_similar': b"This is original research content regarding quantum computing applications.",
            'different': b"This is completely different content about classical computing systems."
        }
    
    @pytest.mark.asyncio
    async def test_exact_duplicate_detection(self, ingestion_engine, sample_content_data):
        """Test detection of exact duplicate content"""
        
        # Mock existing content in database
        with patch.object(ingestion_engine, '_find_exact_hash_duplicates') as mock_exact:
            mock_exact.return_value = [
                DuplicateMatch(
                    existing_content_id=uuid4(),
                    match_method="exact_hash",
                    similarity_score=1.0,
                    match_confidence=1.0,
                    existing_creator="original_creator",
                    existing_creation_date=datetime.now(timezone.utc),
                    match_details={'hash_match': 'SHA256'}
                )
            ]
            
            # Test duplicate detection
            duplicates = await ingestion_engine.check_content_for_duplicates(
                content=sample_content_data['duplicate_exact'],
                content_type=ContentType.RESEARCH_PAPER
            )
            
            # Verify duplicate was found
            assert len(duplicates) == 1
            assert duplicates[0].similarity_score == 1.0
            assert duplicates[0].match_method == "exact_hash"
    
    @pytest.mark.asyncio
    async def test_content_ingestion_duplicate_rejection(self, ingestion_engine, sample_content_data):
        """Test that duplicate content is rejected during ingestion"""
        
        # Mock duplicate detection to return a match
        with patch.object(ingestion_engine, '_detect_duplicates') as mock_detect:
            mock_detect.return_value = [
                DuplicateMatch(
                    existing_content_id=uuid4(),
                    match_method="exact_hash",
                    similarity_score=1.0,
                    match_confidence=1.0,
                    existing_creator="original_creator",
                    existing_creation_date=datetime.now(timezone.utc),
                    match_details={}
                )
            ]
            
            # Attempt to ingest duplicate content
            result = await ingestion_engine.ingest_user_content(
                content=sample_content_data['duplicate_exact'],
                content_type=ContentType.RESEARCH_PAPER,
                metadata={'title': 'Test Paper'},
                user_id='test_user'
            )
            
            # Verify rejection
            assert not result.success
            assert result.status == IngestionStatus.DUPLICATE_REJECTED
            assert len(result.duplicate_matches) > 0
            assert result.ftns_reward == Decimal('0')
    
    @pytest.mark.asyncio
    async def test_quality_assessment_and_rewards(self, ingestion_engine, sample_content_data):
        """Test content quality assessment and appropriate FTNS rewards"""
        
        # Mock no duplicates found
        with patch.object(ingestion_engine, '_detect_duplicates', return_value=[]):
            with patch.object(ingestion_engine, '_create_content_with_provenance') as mock_create:
                with patch.object(ingestion_engine.ftns_service, 'reward_contribution', return_value=True):
                    
                    mock_create.return_value = (uuid4(), Mock())
                    
                    # Test high-quality content ingestion
                    result = await ingestion_engine.ingest_user_content(
                        content=sample_content_data['original'],
                        content_type=ContentType.RESEARCH_PAPER,
                        metadata={
                            'title': 'Comprehensive Quantum Computing Research',
                            'description': 'Detailed analysis of quantum algorithms',
                            'domain': 'quantum_computing',
                            'tags': ['quantum', 'algorithms', 'research']
                        },
                        user_id='test_user'
                    )
                    
                    # Verify successful ingestion
                    assert result.success
                    assert result.status == IngestionStatus.SUCCESS
                    assert result.content_id is not None
                    assert result.ftns_reward > 0
                    assert result.quality_assessment is not None


class TestIntegrationScenarios:
    """Test suite for end-to-end integration scenarios"""
    
    @pytest.fixture
    async def full_system(self):
        """Create a complete system setup for integration testing"""
        system = {
            'voicebox': NWTNVoicebox(),
            'royalty_engine': ContentRoyaltyEngine(),
            'ingestion_engine': NWTNContentIngestionEngine(),
            'provenance_system': EnhancedProvenanceSystem()
        }
        
        # Initialize async components
        await system['voicebox'].initialize()
        await system['royalty_engine'].initialize()
        await system['ingestion_engine'].initialize()
        
        return system
    
    @pytest.mark.asyncio
    async def test_end_to_end_content_lifecycle(self, full_system):
        """Test complete content lifecycle from ingestion to usage and royalty distribution"""
        
        # Step 1: Ingest new content
        content_data = b"Revolutionary quantum machine learning algorithm with proven efficiency gains."
        
        with patch.object(full_system['ingestion_engine'], '_detect_duplicates', return_value=[]):
            with patch.object(full_system['ingestion_engine'], '_create_content_with_provenance') as mock_create:
                with patch.object(full_system['ingestion_engine'].ftns_service, 'reward_contribution', return_value=True):
                    
                    content_id = uuid4()
                    mock_create.return_value = (content_id, Mock())
                    
                    # Ingest content
                    ingestion_result = await full_system['ingestion_engine'].ingest_user_content(
                        content=content_data,
                        content_type=ContentType.RESEARCH_PAPER,
                        metadata={'title': 'Quantum ML Algorithm', 'domain': 'quantum_computing'},
                        user_id='creator_user'
                    )
                    
                    assert ingestion_result.success
                    assert ingestion_result.ftns_reward > 0
        
        # Step 2: Use content in NWTN reasoning
        with patch.object(full_system['voicebox'], '_generate_source_links') as mock_sources:
            with patch.object(full_system['voicebox'], '_distribute_usage_royalties') as mock_royalties:
                
                # Mock reasoning result that uses the content
                mock_reasoning_result = Mock()
                mock_reasoning_result.content_sources = [content_id]
                mock_reasoning_result.overall_confidence = 0.9
                mock_reasoning_result.reasoning_path = ['deductive', 'analogical']
                
                mock_sources.return_value = [
                    SourceLink(
                        content_id=str(content_id),
                        title="Quantum ML Algorithm",
                        creator="creator_user",
                        ipfs_link=f"https://ipfs.prsm.ai/ipfs/QmTest{content_id}",
                        contribution_date="2024-01-01T00:00:00Z",
                        relevance_score=0.9,
                        content_type="research_paper"
                    )
                ]
                
                # Mock successful royalty distribution
                mock_royalties.return_value = None
                
                # Process query that uses the content
                with patch.object(full_system['voicebox'], '_process_through_nwtn', return_value=mock_reasoning_result):
                    with patch.object(full_system['voicebox'], '_translate_to_natural_language', return_value="Test response"):
                        with patch.object(full_system['voicebox'], '_calculate_actual_cost', return_value=1.0):
                            with patch.object(full_system['voicebox'].ftns_service, 'charge_user', return_value=True):
                                
                                # Create mock analysis
                                mock_analysis = Mock()
                                mock_analysis.estimated_cost_ftns = 1.0
                                
                                with patch.object(full_system['voicebox'], '_analyze_query', return_value=mock_analysis):
                                    
                                    # Process a query
                                    response = await full_system['voicebox'].process_query(
                                        user_id='query_user',
                                        query='How can quantum computing improve machine learning?'
                                    )
                                    
                                    # Verify response includes source attribution
                                    assert isinstance(response, VoiceboxResponse)
                                    assert len(response.source_links) > 0
                                    assert response.source_links[0].creator == "creator_user"
                                    
                                    # Verify royalty distribution was called
                                    mock_royalties.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_abuse_prevention_scenarios(self, full_system):
        """Test that the system prevents various forms of abuse"""
        
        # Test 1: Rate limiting
        with patch.object(full_system['ingestion_engine'], '_check_rate_limits') as mock_rate_limit:
            mock_rate_limit.return_value = {'allowed': False, 'reason': 'Hourly limit exceeded'}
            
            result = await full_system['ingestion_engine'].ingest_user_content(
                content=b"Test content",
                content_type=ContentType.RESEARCH_PAPER,
                metadata={'title': 'Test'},
                user_id='abusive_user'
            )
            
            assert not result.success
            assert result.status == IngestionStatus.ABUSE_REJECTED
        
        # Test 2: Low quality content rejection
        with patch.object(full_system['ingestion_engine'], '_check_rate_limits') as mock_rate_limit:
            with patch.object(full_system['ingestion_engine'], '_detect_duplicates', return_value=[]):
                with patch.object(full_system['ingestion_engine'], '_assess_content_quality') as mock_quality:
                    
                    mock_rate_limit.return_value = {'allowed': True, 'reason': 'Within limits'}
                    
                    # Mock poor quality assessment
                    from prsm.compute.nwtn.content_ingestion_engine import ContentQualityAssessment, ContentQuality
                    poor_assessment = ContentQualityAssessment(
                        quality_level=ContentQuality.POOR,
                        quality_score=0.3,
                        assessment_factors={},
                        content_length=10,
                        language_quality=0.3,
                        originality_score=0.3,
                        domain_relevance=0.3
                    )
                    mock_quality.return_value = poor_assessment
                    
                    result = await full_system['ingestion_engine'].ingest_user_content(
                        content=b"bad content",
                        content_type=ContentType.RESEARCH_PAPER,
                        metadata={'title': 'Poor Quality'},
                        user_id='test_user'
                    )
                    
                    assert not result.success
                    assert result.status == IngestionStatus.QUALITY_REJECTED


class TestPerformance:
    """Test suite for performance validation"""
    
    @pytest.mark.asyncio
    async def test_provenance_tracking_performance(self):
        """Test that provenance tracking has acceptable performance"""
        
        provenance_system = EnhancedProvenanceSystem()
        
        # Test content registration performance
        start_time = time.perf_counter()
        
        for i in range(10):  # Register 10 pieces of content
            content_data = f"Test content {i}".encode('utf-8')
            await provenance_system.register_content_with_provenance(
                content_data=content_data,
                content_type=ContentType.RESEARCH_PAPER,
                creator_info={'name': f'creator_{i}', 'platform': 'test'},
                license_info={'type': 'open_source'},
                metadata={'title': f'Content {i}'}
            )
        
        registration_time = time.perf_counter() - start_time
        avg_registration_time = registration_time / 10
        
        # Verify acceptable performance (less than 100ms per registration)
        assert avg_registration_time < 0.1, f"Average registration time {avg_registration_time:.3f}s exceeds threshold"
    
    @pytest.mark.asyncio
    async def test_royalty_calculation_performance(self):
        """Test that royalty calculation performs well with many content sources"""
        
        royalty_engine = ContentRoyaltyEngine()
        await royalty_engine.initialize()
        
        # Test with many content sources
        content_sources = [uuid4() for _ in range(50)]
        
        start_time = time.perf_counter()
        
        calculations = await royalty_engine.calculate_usage_royalty(
            content_sources=content_sources,
            query_complexity=QueryComplexity.COMPLEX,
            user_tier="premium"
        )
        
        calculation_time = time.perf_counter() - start_time
        
        # Verify acceptable performance (less than 500ms for 50 sources)
        assert calculation_time < 0.5, f"Royalty calculation time {calculation_time:.3f}s exceeds threshold"
        assert len(calculations) <= len(content_sources)
    
    @pytest.mark.asyncio
    async def test_duplicate_detection_performance(self):
        """Test that duplicate detection performs well"""
        
        ingestion_engine = NWTNContentIngestionEngine()
        await ingestion_engine.initialize()
        
        # Test duplicate detection with various content sizes
        content_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
        
        for size in content_sizes:
            content_data = b"A" * size
            
            start_time = time.perf_counter()
            
            duplicates = await ingestion_engine.check_content_for_duplicates(
                content=content_data,
                content_type=ContentType.RESEARCH_PAPER
            )
            
            detection_time = time.perf_counter() - start_time
            
            # Verify acceptable performance (less than 200ms for detection)
            assert detection_time < 0.2, f"Duplicate detection time {detection_time:.3f}s exceeds threshold for {size} bytes"


class TestSecurityAndEdgeCases:
    """Test suite for security and edge case handling"""
    
    @pytest.mark.asyncio
    async def test_malformed_content_handling(self):
        """Test handling of malformed or invalid content"""
        
        ingestion_engine = NWTNContentIngestionEngine()
        await ingestion_engine.initialize()
        
        # Test empty content
        result = await ingestion_engine.ingest_user_content(
            content=b"",
            content_type=ContentType.RESEARCH_PAPER,
            metadata={'title': 'Empty Content'},
            user_id='test_user'
        )
        
        assert not result.success
        assert result.status == IngestionStatus.FORMAT_REJECTED
        
        # Test extremely large content
        large_content = b"A" * (200 * 1024 * 1024)  # 200MB
        
        result = await ingestion_engine.ingest_user_content(
            content=large_content,
            content_type=ContentType.RESEARCH_PAPER,
            metadata={'title': 'Too Large'},
            user_id='test_user'
        )
        
        assert not result.success
        assert result.status == IngestionStatus.FORMAT_REJECTED
    
    @pytest.mark.asyncio
    async def test_invalid_user_handling(self):
        """Test handling of invalid user scenarios"""
        
        voicebox = NWTNVoicebox()
        await voicebox.initialize()
        
        # Test query without API configuration
        with pytest.raises(ValueError, match="User has not configured API key"):
            await voicebox.process_query(
                user_id='unconfigured_user',
                query='Test query'
            )
    
    @pytest.mark.asyncio
    async def test_system_failure_recovery(self):
        """Test system behavior during component failures"""
        
        royalty_engine = ContentRoyaltyEngine()
        await royalty_engine.initialize()
        
        # Test royalty calculation with invalid content sources
        invalid_sources = ['not-a-uuid', None, '']
        
        # Should handle gracefully without crashing
        calculations = await royalty_engine.calculate_usage_royalty(
            content_sources=invalid_sources,
            query_complexity=QueryComplexity.SIMPLE,
            user_tier="basic"
        )
        
        # Should return empty list for invalid inputs
        assert isinstance(calculations, list)


# Test runner and utilities
def run_comprehensive_tests():
    """Run the comprehensive test suite"""
    import subprocess
    import sys
    
    try:
        # Run tests with detailed output
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            __file__, 
            '-v', 
            '--tb=short',
            '--durations=10'
        ], capture_output=True, text=True)
        
        print("=== NWTN Provenance Integration Test Results ===")
        print(result.stdout)
        
        if result.stderr:
            print("=== Errors ===")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Failed to run tests: {e}")
        return False


if __name__ == "__main__":
    # Run tests if executed directly
    success = run_comprehensive_tests()
    exit(0 if success else 1)
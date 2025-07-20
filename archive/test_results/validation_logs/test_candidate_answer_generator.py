#!/usr/bin/env python3
"""
Comprehensive Test Suite for CandidateAnswerGenerator
===================================================

Tests the System 1 brainstorming component for generating diverse candidate answers
from analyzed research corpus with proper source tracking and confidence estimation.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any

from prsm.nwtn.candidate_answer_generator import (
    CandidateAnswerGenerator,
    ConceptSynthesizer,
    CandidateType,
    CandidateAnswer,
    SourceContribution,
    CandidateGenerationResult,
    create_candidate_generator
)
from prsm.nwtn.content_analyzer import (
    ContentAnalysisResult,
    ContentSummary,
    ExtractedConcept,
    ContentQuality
)


class TestConceptSynthesizer:
    """Test the ConceptSynthesizer component"""
    
    def setup_method(self):
        """Setup test environment"""
        self.synthesizer = ConceptSynthesizer()
        
        # Create test concepts
        self.test_concepts = [
            ExtractedConcept(
                concept="machine learning algorithm",
                category="methodology",
                confidence=0.8,
                context="Using machine learning algorithm for classification",
                paper_id="paper1"
            ),
            ExtractedConcept(
                concept="neural network approach",
                category="methodology",
                confidence=0.7,
                context="Neural network approach shows promising results",
                paper_id="paper2"
            ),
            ExtractedConcept(
                concept="significant improvement in accuracy",
                category="finding",
                confidence=0.9,
                context="Results show significant improvement in accuracy",
                paper_id="paper1"
            ),
            ExtractedConcept(
                concept="quantum computing theory",
                category="theory",
                confidence=0.6,
                context="Based on quantum computing theory principles",
                paper_id="paper3"
            ),
            ExtractedConcept(
                concept="real-world deployment",
                category="application",
                confidence=0.7,
                context="Suitable for real-world deployment scenarios",
                paper_id="paper2"
            )
        ]
    
    def test_synthesis_chains_creation(self):
        """Test creation of synthesis-based reasoning chains"""
        chains = self.synthesizer.synthesize_concepts(self.test_concepts, CandidateType.SYNTHESIS)
        
        assert len(chains) >= 1
        assert any("machine learning" in chain.lower() for chain in chains)
        print(f"✅ Synthesis chains created: {len(chains)}")
        for i, chain in enumerate(chains):
            print(f"   Chain {i+1}: {chain}")
    
    def test_methodological_chains_creation(self):
        """Test creation of methodology-focused reasoning chains"""
        chains = self.synthesizer.synthesize_concepts(self.test_concepts, CandidateType.METHODOLOGICAL)
        
        assert len(chains) >= 1
        assert any("approach" in chain.lower() for chain in chains)
        print(f"✅ Methodological chains created: {len(chains)}")
        for i, chain in enumerate(chains):
            print(f"   Chain {i+1}: {chain}")
    
    def test_theoretical_chains_creation(self):
        """Test creation of theory-based reasoning chains"""
        chains = self.synthesizer.synthesize_concepts(self.test_concepts, CandidateType.THEORETICAL)
        
        assert len(chains) >= 1
        assert any("theory" in chain.lower() for chain in chains)
        print(f"✅ Theoretical chains created: {len(chains)}")
        for i, chain in enumerate(chains):
            print(f"   Chain {i+1}: {chain}")
    
    def test_empirical_chains_creation(self):
        """Test creation of empirical-based reasoning chains"""
        chains = self.synthesizer.synthesize_concepts(self.test_concepts, CandidateType.EMPIRICAL)
        
        assert len(chains) >= 1
        assert any("evidence" in chain.lower() for chain in chains)
        print(f"✅ Empirical chains created: {len(chains)}")
        for i, chain in enumerate(chains):
            print(f"   Chain {i+1}: {chain}")
    
    def test_applied_chains_creation(self):
        """Test creation of application-focused reasoning chains"""
        chains = self.synthesizer.synthesize_concepts(self.test_concepts, CandidateType.APPLIED)
        
        assert len(chains) >= 1
        assert any("practical" in chain.lower() or "deployment" in chain.lower() for chain in chains)
        print(f"✅ Applied chains created: {len(chains)}")
        for i, chain in enumerate(chains):
            print(f"   Chain {i+1}: {chain}")
    
    def test_empty_concepts_handling(self):
        """Test handling of empty concept lists"""
        chains = self.synthesizer.synthesize_concepts([], CandidateType.SYNTHESIS)
        assert chains == []
        print("✅ Empty concepts handled correctly")


class TestCandidateAnswerGenerator:
    """Test the main CandidateAnswerGenerator component"""
    
    def setup_method(self):
        """Setup test environment"""
        self.generator = None
        
        # Create test content analysis result
        self.test_analysis = self._create_test_analysis()
    
    def _create_test_analysis(self) -> ContentAnalysisResult:
        """Create test ContentAnalysisResult with mock data"""
        # Create test papers with different quality levels
        papers = [
            ContentSummary(
                paper_id="paper1",
                title="Advanced Machine Learning for Classification",
                main_contributions=["Novel ML algorithm", "Improved accuracy"],
                key_concepts=[
                    ExtractedConcept("machine learning", "methodology", 0.9, "context", "paper1"),
                    ExtractedConcept("classification accuracy", "finding", 0.8, "context", "paper1"),
                    ExtractedConcept("supervised learning", "methodology", 0.7, "context", "paper1")
                ],
                methodologies=["supervised learning", "cross-validation"],
                findings=["95% accuracy achieved", "outperforms baseline"],
                applications=["image recognition", "text classification"],
                limitations=["requires large dataset"],
                quality_score=0.85,
                quality_level=ContentQuality.EXCELLENT
            ),
            ContentSummary(
                paper_id="paper2",
                title="Neural Network Architectures for Deep Learning",
                main_contributions=["New neural architecture", "Faster training"],
                key_concepts=[
                    ExtractedConcept("neural networks", "methodology", 0.8, "context", "paper2"),
                    ExtractedConcept("deep learning", "methodology", 0.9, "context", "paper2"),
                    ExtractedConcept("faster convergence", "finding", 0.7, "context", "paper2")
                ],
                methodologies=["deep learning", "backpropagation"],
                findings=["50% faster training", "better generalization"],
                applications=["computer vision", "natural language processing"],
                limitations=["high computational cost"],
                quality_score=0.75,
                quality_level=ContentQuality.GOOD
            ),
            ContentSummary(
                paper_id="paper3",
                title="Quantum Computing Applications in AI",
                main_contributions=["Quantum-classical hybrid", "Theoretical framework"],
                key_concepts=[
                    ExtractedConcept("quantum computing", "theory", 0.8, "context", "paper3"),
                    ExtractedConcept("quantum supremacy", "theory", 0.7, "context", "paper3"),
                    ExtractedConcept("hybrid algorithms", "methodology", 0.6, "context", "paper3")
                ],
                methodologies=["quantum algorithms", "hybrid computing"],
                findings=["theoretical advantages", "scalability challenges"],
                applications=["optimization problems", "cryptography"],
                limitations=["hardware limitations", "noise sensitivity"],
                quality_score=0.65,
                quality_level=ContentQuality.GOOD
            )
        ]
        
        return ContentAnalysisResult(
            query="How can machine learning improve classification accuracy?",
            analyzed_papers=papers,
            total_concepts_extracted=9,
            analysis_time_seconds=1.5,
            quality_distribution={
                ContentQuality.EXCELLENT: 1,
                ContentQuality.GOOD: 2,
                ContentQuality.AVERAGE: 0,
                ContentQuality.POOR: 0,
                ContentQuality.UNUSABLE: 0
            }
        )
    
    @pytest.mark.asyncio
    async def test_generator_initialization(self):
        """Test generator initialization"""
        self.generator = CandidateAnswerGenerator()
        success = await self.generator.initialize()
        
        assert success
        assert self.generator.initialized
        print("✅ Generator initialized successfully")
    
    @pytest.mark.asyncio
    async def test_factory_function(self):
        """Test factory function for creating generator"""
        generator = await create_candidate_generator()
        
        assert generator is not None
        assert generator.initialized
        print("✅ Factory function creates initialized generator")
    
    @pytest.mark.asyncio
    async def test_candidate_generation(self):
        """Test generation of candidate answers"""
        self.generator = await create_candidate_generator()
        
        result = await self.generator.generate_candidates(self.test_analysis)
        
        assert isinstance(result, CandidateGenerationResult)
        assert result.query == self.test_analysis.query
        assert len(result.candidate_answers) > 0
        assert len(result.candidate_answers) <= self.generator.target_candidates
        assert result.generation_time_seconds > 0
        assert result.total_sources_used > 0
        
        print(f"✅ Generated {len(result.candidate_answers)} candidate answers")
        print(f"   Query: {result.query}")
        print(f"   Generation time: {result.generation_time_seconds:.2f}s")
        print(f"   Sources used: {result.total_sources_used}")
        
        # Test individual candidates
        for i, candidate in enumerate(result.candidate_answers):
            assert candidate.candidate_id is not None
            assert candidate.query == self.test_analysis.query
            assert candidate.answer_text is not None
            assert isinstance(candidate.answer_type, CandidateType)
            assert 0.0 <= candidate.confidence_score <= 1.0
            assert len(candidate.source_contributions) > 0
            
            print(f"   Candidate {i+1}: {candidate.answer_type.value} (confidence: {candidate.confidence_score:.2f})")
            print(f"      Sources: {len(candidate.source_contributions)}")
            print(f"      Concepts: {len(candidate.key_concepts_used)}")
    
    @pytest.mark.asyncio
    async def test_answer_type_diversity(self):
        """Test that diverse answer types are generated"""
        self.generator = await create_candidate_generator()
        
        result = await self.generator.generate_candidates(self.test_analysis)
        
        answer_types = [c.answer_type for c in result.candidate_answers]
        unique_types = set(answer_types)
        
        assert len(unique_types) >= 2  # At least 2 different types
        print(f"✅ Generated {len(unique_types)} unique answer types: {[t.value for t in unique_types]}")
        
        # Check diversity metrics
        assert "type_diversity" in result.diversity_metrics
        assert result.diversity_metrics["type_diversity"] > 0.0
        print(f"   Type diversity: {result.diversity_metrics['type_diversity']:.2f}")
    
    @pytest.mark.asyncio
    async def test_source_contribution_tracking(self):
        """Test that source contributions are properly tracked"""
        self.generator = await create_candidate_generator()
        
        result = await self.generator.generate_candidates(self.test_analysis)
        
        for candidate in result.candidate_answers:
            assert len(candidate.source_contributions) > 0
            
            total_weight = sum(contrib.contribution_weight for contrib in candidate.source_contributions)
            assert 0.8 <= total_weight <= 1.2  # Allow some tolerance
            
            for contrib in candidate.source_contributions:
                assert contrib.paper_id in ["paper1", "paper2", "paper3"]
                assert contrib.contribution_type in ["primary", "supporting", "comparative", "methodological"]
                assert 0.0 <= contrib.contribution_weight <= 1.0
                assert 0.0 <= contrib.quality_score <= 1.0
                assert 0.0 <= contrib.confidence <= 1.0
                assert len(contrib.relevant_concepts) > 0
        
        print("✅ Source contributions tracked correctly")
        print(f"   Total unique sources used: {len(result.source_utilization)}")
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self):
        """Test confidence scoring mechanism"""
        self.generator = await create_candidate_generator()
        
        result = await self.generator.generate_candidates(self.test_analysis)
        
        confidence_scores = [c.confidence_score for c in result.candidate_answers]
        
        # All confidence scores should be within valid range
        assert all(0.0 <= score <= 1.0 for score in confidence_scores)
        
        # Should have reasonable distribution (not all the same)
        if len(confidence_scores) > 1:
            assert len(set(confidence_scores)) > 1  # Some variation
        
        print(f"✅ Confidence scores: {[f'{score:.2f}' for score in confidence_scores]}")
        print(f"   Average confidence: {sum(confidence_scores) / len(confidence_scores):.2f}")
    
    @pytest.mark.asyncio
    async def test_reasoning_chain_generation(self):
        """Test reasoning chain generation"""
        self.generator = await create_candidate_generator()
        
        result = await self.generator.generate_candidates(self.test_analysis)
        
        for candidate in result.candidate_answers:
            assert len(candidate.reasoning_chain) >= 0  # Can be empty for some types
            
            # Check reasoning chain content
            for reasoning in candidate.reasoning_chain:
                assert len(reasoning) > 10  # Reasonable length
                assert isinstance(reasoning, str)
        
        print("✅ Reasoning chains generated correctly")
    
    @pytest.mark.asyncio
    async def test_strengths_and_limitations(self):
        """Test identification of strengths and limitations"""
        self.generator = await create_candidate_generator()
        
        result = await self.generator.generate_candidates(self.test_analysis)
        
        for candidate in result.candidate_answers:
            # Should have at least some strengths or limitations
            assert len(candidate.strengths) > 0 or len(candidate.limitations) > 0
            
            # Check content quality
            for strength in candidate.strengths:
                assert len(strength) > 5
                assert isinstance(strength, str)
            
            for limitation in candidate.limitations:
                assert len(limitation) > 5
                assert isinstance(limitation, str)
        
        print("✅ Strengths and limitations identified correctly")
    
    @pytest.mark.asyncio
    async def test_target_candidates_configuration(self):
        """Test configuring target number of candidates"""
        self.generator = await create_candidate_generator()
        
        # Test with different target numbers
        for target in [3, 5, 10]:
            result = await self.generator.generate_candidates(self.test_analysis, target_candidates=target)
            
            # Should generate up to target number (may be less due to quality filtering)
            assert len(result.candidate_answers) <= target
            print(f"✅ Target {target} candidates: generated {len(result.candidate_answers)}")
    
    @pytest.mark.asyncio
    async def test_parameter_configuration(self):
        """Test configuration of generation parameters"""
        self.generator = await create_candidate_generator()
        
        # Configure parameters
        await self.generator.configure_generation_params(
            target_candidates=6,
            min_confidence_threshold=0.4,
            diversity_threshold=0.8
        )
        
        assert self.generator.target_candidates == 6
        assert self.generator.min_confidence_threshold == 0.4
        assert self.generator.diversity_threshold == 0.8
        
        print("✅ Parameters configured correctly")
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test generation statistics tracking"""
        self.generator = await create_candidate_generator()
        
        # Generate some candidates
        await self.generator.generate_candidates(self.test_analysis)
        await self.generator.generate_candidates(self.test_analysis)
        
        stats = self.generator.get_generation_statistics()
        
        assert stats['total_generations'] == 2
        assert stats['successful_generations'] >= 1
        assert stats['candidates_generated'] > 0
        assert stats['average_generation_time'] > 0
        assert 'success_rate' in stats
        assert 'average_candidates_per_generation' in stats
        
        print("✅ Statistics tracked correctly")
        print(f"   Total generations: {stats['total_generations']}")
        print(f"   Success rate: {stats['success_rate']:.2f}")
        print(f"   Average candidates per generation: {stats['average_candidates_per_generation']:.1f}")
    
    @pytest.mark.asyncio
    async def test_empty_analysis_handling(self):
        """Test handling of empty or low-quality analysis"""
        self.generator = await create_candidate_generator()
        
        # Create empty analysis
        empty_analysis = ContentAnalysisResult(
            query="test query",
            analyzed_papers=[],
            total_concepts_extracted=0,
            analysis_time_seconds=0.0,
            quality_distribution={}
        )
        
        result = await self.generator.generate_candidates(empty_analysis)
        
        assert isinstance(result, CandidateGenerationResult)
        assert len(result.candidate_answers) == 0
        assert result.total_sources_used == 0
        
        print("✅ Empty analysis handled correctly")
    
    @pytest.mark.asyncio
    async def test_answer_text_quality(self):
        """Test quality of generated answer text"""
        self.generator = await create_candidate_generator()
        
        result = await self.generator.generate_candidates(self.test_analysis)
        
        for candidate in result.candidate_answers:
            answer_text = candidate.answer_text
            
            # Basic quality checks
            assert len(answer_text) > 50  # Reasonable length
            assert answer_text.count('.') >= 1  # At least one sentence
            assert any(word in answer_text.lower() for word in ['machine', 'learning', 'classification'])
            
            # Should not be empty or just whitespace
            assert answer_text.strip() != ""
        
        print("✅ Answer text quality verified")
        print(f"   Sample answer: {result.candidate_answers[0].answer_text[:100]}...")


class TestIntegrationWithContentAnalyzer:
    """Test integration with ContentAnalyzer components"""
    
    @pytest.mark.asyncio
    async def test_content_analysis_integration(self):
        """Test integration with ContentAnalysisResult"""
        generator = await create_candidate_generator()
        
        # Create realistic content analysis result
        test_concepts = [
            ExtractedConcept("transformer architecture", "methodology", 0.9, "context", "paper1"),
            ExtractedConcept("attention mechanism", "methodology", 0.8, "context", "paper1"),
            ExtractedConcept("state-of-the-art performance", "finding", 0.7, "context", "paper1"),
            ExtractedConcept("natural language processing", "application", 0.8, "context", "paper1")
        ]
        
        paper = ContentSummary(
            paper_id="integration_test_paper",
            title="Transformer Networks for NLP",
            main_contributions=["New transformer variant", "Improved efficiency"],
            key_concepts=test_concepts,
            methodologies=["attention mechanism", "self-attention"],
            findings=["better performance", "reduced training time"],
            applications=["machine translation", "text summarization"],
            limitations=["memory requirements"],
            quality_score=0.9,
            quality_level=ContentQuality.EXCELLENT
        )
        
        analysis = ContentAnalysisResult(
            query="How do transformers improve NLP performance?",
            analyzed_papers=[paper],
            total_concepts_extracted=4,
            analysis_time_seconds=1.0,
            quality_distribution={ContentQuality.EXCELLENT: 1}
        )
        
        result = await generator.generate_candidates(analysis)
        
        assert len(result.candidate_answers) > 0
        assert result.query == analysis.query
        
        # Check that concepts from content analysis are used
        for candidate in result.candidate_answers:
            assert len(candidate.key_concepts_used) > 0
            
            # Should use concepts from the analyzed paper
            used_concepts = set(candidate.key_concepts_used)
            analysis_concepts = set(c.concept for c in test_concepts)
            assert len(used_concepts.intersection(analysis_concepts)) > 0
        
        print("✅ Integration with ContentAnalyzer working correctly")
        print(f"   Query: {result.query}")
        print(f"   Generated {len(result.candidate_answers)} candidates")


# Test execution functions
def run_concept_synthesizer_tests():
    """Run ConceptSynthesizer tests"""
    print("🧪 Testing ConceptSynthesizer...")
    test_class = TestConceptSynthesizer()
    test_class.setup_method()
    
    test_class.test_synthesis_chains_creation()
    test_class.test_methodological_chains_creation()
    test_class.test_theoretical_chains_creation()
    test_class.test_empirical_chains_creation()
    test_class.test_applied_chains_creation()
    test_class.test_empty_concepts_handling()
    
    print("✅ ConceptSynthesizer tests completed successfully!")


async def run_generator_tests():
    """Run CandidateAnswerGenerator tests"""
    print("\n🧪 Testing CandidateAnswerGenerator...")
    test_class = TestCandidateAnswerGenerator()
    
    test_class.setup_method()
    await test_class.test_generator_initialization()
    
    test_class.setup_method()
    await test_class.test_factory_function()
    
    test_class.setup_method()
    await test_class.test_candidate_generation()
    
    test_class.setup_method()
    await test_class.test_answer_type_diversity()
    
    test_class.setup_method()
    await test_class.test_source_contribution_tracking()
    
    test_class.setup_method()
    await test_class.test_confidence_scoring()
    
    test_class.setup_method()
    await test_class.test_reasoning_chain_generation()
    
    test_class.setup_method()
    await test_class.test_strengths_and_limitations()
    
    test_class.setup_method()
    await test_class.test_target_candidates_configuration()
    
    test_class.setup_method()
    await test_class.test_parameter_configuration()
    
    test_class.setup_method()
    await test_class.test_statistics_tracking()
    
    test_class.setup_method()
    await test_class.test_empty_analysis_handling()
    
    test_class.setup_method()
    await test_class.test_answer_text_quality()
    
    print("✅ CandidateAnswerGenerator tests completed successfully!")


async def run_integration_tests():
    """Run integration tests"""
    print("\n🧪 Testing Integration...")
    test_class = TestIntegrationWithContentAnalyzer()
    
    await test_class.test_content_analysis_integration()
    
    print("✅ Integration tests completed successfully!")


async def main():
    """Run all tests"""
    print("🚀 Starting CandidateAnswerGenerator Test Suite")
    print("=" * 60)
    
    try:
        # Run synchronous tests
        run_concept_synthesizer_tests()
        
        # Run async tests
        await run_generator_tests()
        await run_integration_tests()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! CandidateAnswerGenerator is working correctly!")
        print("   ✅ ConceptSynthesizer: Concept synthesis working")
        print("   ✅ CandidateAnswerGenerator: Candidate generation working")
        print("   ✅ Integration: Content analysis integration working")
        print("   ✅ Quality: Answer quality and diversity verified")
        print("   ✅ Tracking: Source contribution tracking working")
        print("   ✅ Statistics: Performance monitoring working")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
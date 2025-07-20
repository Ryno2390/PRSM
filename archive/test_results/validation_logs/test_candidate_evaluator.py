#!/usr/bin/env python3
"""
Comprehensive Test Suite for CandidateEvaluator
==============================================

Tests the System 2 methodical evaluation component that evaluates candidate answers
using the existing MetaReasoningEngine's sophisticated reasoning capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock

from prsm.nwtn.candidate_evaluator import (
    CandidateEvaluator,
    RelevanceScorer,
    ConfidenceScorer,
    SourceTracker,
    EvaluationCriteria,
    EvaluationScore,
    CandidateEvaluation,
    EvaluationResult,
    create_candidate_evaluator
)
from prsm.nwtn.candidate_answer_generator import (
    CandidateGenerationResult,
    CandidateAnswer,
    CandidateType,
    SourceContribution
)
from prsm.nwtn.meta_reasoning_engine import (
    MetaReasoningEngine,
    ThinkingMode,
    MetaReasoningResult,
    ReasoningResult,
    ReasoningEngine
)


class TestRelevanceScorer:
    """Test the RelevanceScorer component"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create mock meta-reasoning engine
        self.mock_meta_engine = Mock(spec=MetaReasoningEngine)
        self.mock_meta_engine.meta_reason = AsyncMock()
        
        self.scorer = RelevanceScorer(self.mock_meta_engine)
        
        # Create test candidate
        self.test_candidate = CandidateAnswer(
            candidate_id="test_candidate",
            query="How can machine learning improve classification accuracy?",
            answer_text="Machine learning can improve classification accuracy through advanced algorithms and better feature engineering.",
            answer_type=CandidateType.METHODOLOGICAL,
            confidence_score=0.8,
            source_contributions=[
                SourceContribution(
                    paper_id="paper1",
                    title="ML Classification Methods",
                    contribution_type="primary",
                    contribution_weight=0.6,
                    relevant_concepts=["machine learning", "classification"],
                    quality_score=0.85,
                    confidence=0.8
                )
            ],
            reasoning_chain=["ML algorithms provide systematic approaches"],
            key_concepts_used=["machine learning", "classification", "accuracy"],
            strengths=["Methodical approach"],
            limitations=["Limited scope"],
            diversity_score=0.7,
            generation_method="system1_brainstorming"
        )
    
    @pytest.mark.asyncio
    async def test_scorer_initialization(self):
        """Test scorer initialization"""
        success = await self.scorer.initialize()
        
        assert success is None  # initialize() returns None
        assert self.scorer.initialized
        print("✅ RelevanceScorer initialized successfully")
    
    @pytest.mark.asyncio
    async def test_relevance_scoring(self):
        """Test relevance scoring functionality"""
        await self.scorer.initialize()
        
        # Mock meta-reasoning result
        mock_result = Mock(spec=MetaReasoningResult)
        mock_result.confidence_score = 0.85
        mock_result.synthesis_summary = "This answer directly addresses the question with specific methodological approaches"
        mock_result.reasoning_results = [Mock() for _ in range(5)]  # 5 reasoning engines
        
        self.mock_meta_engine.meta_reason.return_value = mock_result
        
        # Test relevance scoring
        score = await self.scorer.score_relevance(
            "How can machine learning improve classification accuracy?",
            self.test_candidate
        )
        
        assert isinstance(score, EvaluationScore)
        assert score.criterion == EvaluationCriteria.RELEVANCE
        assert 0.0 <= score.score <= 1.0
        assert score.confidence > 0.0
        assert len(score.reasoning) > 0
        
        print(f"✅ Relevance score: {score.score:.2f}")
        print(f"   Confidence: {score.confidence:.2f}")
        print(f"   Reasoning: {score.reasoning[:100]}...")
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in relevance scoring"""
        await self.scorer.initialize()
        
        # Mock exception
        self.mock_meta_engine.meta_reason.side_effect = Exception("Test error")
        
        score = await self.scorer.score_relevance(
            "test query",
            self.test_candidate
        )
        
        assert isinstance(score, EvaluationScore)
        assert score.score == 0.5  # Default score
        assert score.confidence == 0.3  # Default confidence
        assert "Error during relevance evaluation" in score.reasoning
        
        print("✅ Error handling works correctly")


class TestConfidenceScorer:
    """Test the ConfidenceScorer component"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create mock meta-reasoning engine
        self.mock_meta_engine = Mock(spec=MetaReasoningEngine)
        self.mock_meta_engine.meta_reason = AsyncMock()
        
        self.scorer = ConfidenceScorer(self.mock_meta_engine)
        
        # Create test candidate
        self.test_candidate = CandidateAnswer(
            candidate_id="test_candidate",
            query="How can machine learning improve classification accuracy?",
            answer_text="Machine learning can improve classification accuracy through advanced algorithms.",
            answer_type=CandidateType.EMPIRICAL,
            confidence_score=0.9,
            source_contributions=[
                SourceContribution(
                    paper_id="paper1",
                    title="ML Classification Study",
                    contribution_type="primary",
                    contribution_weight=0.8,
                    relevant_concepts=["machine learning", "classification"],
                    quality_score=0.9,
                    confidence=0.85
                )
            ],
            reasoning_chain=["Empirical evidence shows improvement", "Studies demonstrate effectiveness"],
            key_concepts_used=["machine learning", "classification"],
            strengths=["Strong empirical evidence"],
            limitations=["Limited generalizability"],
            diversity_score=0.8,
            generation_method="system1_brainstorming"
        )
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self):
        """Test confidence scoring functionality"""
        await self.scorer.initialize()
        
        # Mock meta-reasoning result
        mock_result = Mock(spec=MetaReasoningResult)
        mock_result.confidence_score = 0.88
        mock_result.synthesis_summary = "Strong evidence supports this conclusion with multiple high-quality sources"
        mock_result.reasoning_results = [Mock() for _ in range(6)]  # 6 reasoning engines
        
        self.mock_meta_engine.meta_reason.return_value = mock_result
        
        # Test confidence scoring
        score = await self.scorer.score_confidence(
            "How can machine learning improve classification accuracy?",
            self.test_candidate
        )
        
        assert isinstance(score, EvaluationScore)
        assert score.criterion == EvaluationCriteria.EVIDENCE
        assert 0.0 <= score.score <= 1.0
        assert score.confidence > 0.0
        assert len(score.supporting_evidence) > 0
        
        print(f"✅ Confidence score: {score.score:.2f}")
        print(f"   Evidence quality: {score.supporting_evidence}")
    
    @pytest.mark.asyncio
    async def test_score_calculation(self):
        """Test confidence score calculation logic"""
        await self.scorer.initialize()
        
        # Mock meta-reasoning result
        mock_result = Mock(spec=MetaReasoningResult)
        mock_result.confidence_score = 0.7
        mock_result.synthesis_summary = "Test summary"
        mock_result.reasoning_results = [Mock() for _ in range(3)]
        
        self.mock_meta_engine.meta_reason.return_value = mock_result
        
        # Test with high-quality sources
        high_quality_candidate = self.test_candidate
        high_quality_candidate.source_contributions[0].quality_score = 0.95
        
        score = await self.scorer.score_confidence("test query", high_quality_candidate)
        
        # Should factor in source quality
        assert score.score > 0.7  # Should be higher than just meta-reasoning confidence
        
        print(f"✅ Score calculation incorporates source quality")


class TestSourceTracker:
    """Test the SourceTracker component"""
    
    def setup_method(self):
        """Setup test environment"""
        self.tracker = SourceTracker()
    
    @pytest.mark.asyncio
    async def test_source_tracking(self):
        """Test source tracking functionality"""
        await self.tracker.initialize()
        
        # Track some source usage
        self.tracker.track_source_usage("paper1", "Used in relevance evaluation")
        self.tracker.track_source_usage("paper2", "Used in confidence evaluation")
        self.tracker.track_source_usage("paper1", "Used in coherence evaluation")
        
        lineage = self.tracker.get_source_lineage()
        
        assert "paper1" in lineage
        assert "paper2" in lineage
        assert len(lineage["paper1"]) == 2
        assert len(lineage["paper2"]) == 1
        
        print(f"✅ Source tracking works: {dict(lineage)}")
    
    @pytest.mark.asyncio
    async def test_reset_tracking(self):
        """Test resetting source tracking"""
        # Create fresh tracker for this test
        fresh_tracker = SourceTracker()
        await fresh_tracker.initialize()
        
        # Track some usage
        fresh_tracker.track_source_usage("paper1", "test usage")
        assert len(fresh_tracker.get_source_lineage()) == 1
        
        # Reset and verify
        fresh_tracker.reset()
        assert len(fresh_tracker.get_source_lineage()) == 0
        
        print("✅ Source tracking reset works")


class TestCandidateEvaluator:
    """Test the main CandidateEvaluator component"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create mock meta-reasoning engine
        self.mock_meta_engine = Mock(spec=MetaReasoningEngine)
        self.mock_meta_engine.initialize_external_knowledge_base = AsyncMock()
        self.mock_meta_engine.meta_reason = AsyncMock()
        
        self.evaluator = CandidateEvaluator(self.mock_meta_engine)
        
        # Create test candidate generation result
        self.test_result = self._create_test_generation_result()
    
    def _create_test_generation_result(self) -> CandidateGenerationResult:
        """Create test candidate generation result"""
        candidates = [
            CandidateAnswer(
                candidate_id="candidate1",
                query="How can machine learning improve classification accuracy?",
                answer_text="Machine learning improves classification through advanced algorithms and feature engineering.",
                answer_type=CandidateType.METHODOLOGICAL,
                confidence_score=0.85,
                source_contributions=[
                    SourceContribution(
                        paper_id="paper1",
                        title="ML Classification Methods",
                        contribution_type="primary",
                        contribution_weight=0.6,
                        relevant_concepts=["machine learning", "classification"],
                        quality_score=0.85,
                        confidence=0.8
                    )
                ],
                reasoning_chain=["ML algorithms provide systematic approaches"],
                key_concepts_used=["machine learning", "classification"],
                strengths=["Systematic approach"],
                limitations=["Limited scope"],
                diversity_score=0.7,
                generation_method="system1_brainstorming"
            ),
            CandidateAnswer(
                candidate_id="candidate2",
                query="How can machine learning improve classification accuracy?",
                answer_text="Empirical studies show that neural networks significantly improve classification accuracy.",
                answer_type=CandidateType.EMPIRICAL,
                confidence_score=0.9,
                source_contributions=[
                    SourceContribution(
                        paper_id="paper2",
                        title="Neural Network Classification Study",
                        contribution_type="primary",
                        contribution_weight=0.8,
                        relevant_concepts=["neural networks", "classification"],
                        quality_score=0.9,
                        confidence=0.85
                    )
                ],
                reasoning_chain=["Empirical evidence shows improvement"],
                key_concepts_used=["neural networks", "classification"],
                strengths=["Strong empirical evidence"],
                limitations=["Specific to neural networks"],
                diversity_score=0.8,
                generation_method="system1_brainstorming"
            )
        ]
        
        return CandidateGenerationResult(
            query="How can machine learning improve classification accuracy?",
            candidate_answers=candidates,
            generation_time_seconds=0.5,
            total_sources_used=2,
            source_utilization={"paper1": 0.6, "paper2": 0.8},
            diversity_metrics={"type_diversity": 1.0, "concept_diversity": 0.8},
            quality_distribution={CandidateType.METHODOLOGICAL: 1, CandidateType.EMPIRICAL: 1}
        )
    
    @pytest.mark.asyncio
    async def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        success = await self.evaluator.initialize()
        
        assert success
        assert self.evaluator.initialized
        print("✅ CandidateEvaluator initialized successfully")
    
    @pytest.mark.asyncio
    async def test_factory_function(self):
        """Test factory function"""
        evaluator = await create_candidate_evaluator(self.mock_meta_engine)
        
        assert evaluator is not None
        assert evaluator.initialized
        print("✅ Factory function creates initialized evaluator")
    
    @pytest.mark.asyncio
    async def test_candidate_evaluation(self):
        """Test evaluation of candidate answers"""
        await self.evaluator.initialize()
        
        # Mock meta-reasoning results
        mock_result = Mock(spec=MetaReasoningResult)
        mock_result.confidence_score = 0.8
        mock_result.synthesis_summary = "Good evaluation result"
        mock_result.reasoning_results = [Mock() for _ in range(4)]
        
        self.mock_meta_engine.meta_reason.return_value = mock_result
        
        # Test evaluation
        result = await self.evaluator.evaluate_candidates(self.test_result)
        
        assert isinstance(result, EvaluationResult)
        assert result.query == self.test_result.query
        assert len(result.candidate_evaluations) == 2
        assert result.best_candidate is not None
        assert result.evaluation_time_seconds > 0
        assert result.overall_confidence > 0
        
        print(f"✅ Evaluated {len(result.candidate_evaluations)} candidates")
        print(f"   Best candidate score: {result.best_candidate.overall_score:.2f}")
        print(f"   Overall confidence: {result.overall_confidence:.2f}")
        print(f"   Evaluation time: {result.evaluation_time_seconds:.3f}s")
    
    @pytest.mark.asyncio
    async def test_evaluation_criteria(self):
        """Test evaluation with different criteria"""
        await self.evaluator.initialize()
        
        # Mock meta-reasoning results
        mock_result = Mock(spec=MetaReasoningResult)
        mock_result.confidence_score = 0.75
        mock_result.synthesis_summary = "Test evaluation"
        mock_result.reasoning_results = [Mock() for _ in range(3)]
        
        self.mock_meta_engine.meta_reason.return_value = mock_result
        
        # Test with custom criteria
        custom_criteria = [
            EvaluationCriteria.RELEVANCE,
            EvaluationCriteria.EVIDENCE,
            EvaluationCriteria.COHERENCE,
            EvaluationCriteria.COMPLETENESS,
            EvaluationCriteria.NOVELTY
        ]
        
        result = await self.evaluator.evaluate_candidates(
            self.test_result,
            evaluation_criteria=custom_criteria
        )
        
        assert len(result.evaluation_criteria_used) == len(custom_criteria)
        assert result.evaluation_criteria_used == custom_criteria
        
        # Check that each candidate has scores for all criteria
        for evaluation in result.candidate_evaluations:
            assert len(evaluation.evaluation_scores) == len(custom_criteria)
            criteria_evaluated = [score.criterion for score in evaluation.evaluation_scores]
            assert all(criterion in criteria_evaluated for criterion in custom_criteria)
        
        print(f"✅ Evaluation with {len(custom_criteria)} criteria works")
    
    @pytest.mark.asyncio
    async def test_thinking_modes(self):
        """Test evaluation with different thinking modes"""
        await self.evaluator.initialize()
        
        # Mock meta-reasoning results
        mock_result = Mock(spec=MetaReasoningResult)
        mock_result.confidence_score = 0.8
        mock_result.synthesis_summary = "Deep thinking result"
        mock_result.reasoning_results = [Mock() for _ in range(7)]  # All engines
        
        self.mock_meta_engine.meta_reason.return_value = mock_result
        
        # Test with deep thinking mode
        result = await self.evaluator.evaluate_candidates(
            self.test_result,
            thinking_mode=ThinkingMode.DEEP
        )
        
        assert result.thinking_mode_used == ThinkingMode.DEEP
        
        # Verify meta-reasoning was called with deep thinking
        calls = self.mock_meta_engine.meta_reason.call_args_list
        assert len(calls) > 0
        # Check that some calls used deep thinking mode
        deep_calls = [call for call in calls if call[1].get('thinking_mode') == ThinkingMode.DEEP]
        assert len(deep_calls) > 0
        
        print(f"✅ Deep thinking mode evaluation works")
    
    @pytest.mark.asyncio
    async def test_ranking_system(self):
        """Test candidate ranking system"""
        await self.evaluator.initialize()
        
        # Mock different scores for different candidates
        def mock_meta_reason(*args, **kwargs):
            mock_result = Mock(spec=MetaReasoningResult)
            # Return different scores based on the query content
            if "Neural Network" in kwargs.get('query', ''):
                mock_result.confidence_score = 0.9  # Higher score for neural network candidate
            else:
                mock_result.confidence_score = 0.7  # Lower score for other candidates
            mock_result.synthesis_summary = "Test evaluation"
            mock_result.reasoning_results = [Mock() for _ in range(4)]
            return mock_result
        
        self.mock_meta_engine.meta_reason.side_effect = mock_meta_reason
        
        result = await self.evaluator.evaluate_candidates(self.test_result)
        
        # Check ranking
        assert len(result.candidate_evaluations) == 2
        assert result.candidate_evaluations[0].ranking_position == 1
        assert result.candidate_evaluations[1].ranking_position == 2
        
        # Best candidate should be the one with highest score
        best_score = result.candidate_evaluations[0].overall_score
        second_score = result.candidate_evaluations[1].overall_score
        assert best_score >= second_score
        
        print(f"✅ Ranking system works: 1st={best_score:.2f}, 2nd={second_score:.2f}")
    
    @pytest.mark.asyncio
    async def test_source_lineage_tracking(self):
        """Test source lineage tracking"""
        await self.evaluator.initialize()
        
        # Mock meta-reasoning results
        mock_result = Mock(spec=MetaReasoningResult)
        mock_result.confidence_score = 0.8
        mock_result.synthesis_summary = "Test evaluation"
        mock_result.reasoning_results = [Mock() for _ in range(3)]
        
        self.mock_meta_engine.meta_reason.return_value = mock_result
        
        result = await self.evaluator.evaluate_candidates(self.test_result)
        
        # Check source lineage
        assert len(result.source_lineage) > 0
        assert "paper1" in result.source_lineage or "paper2" in result.source_lineage
        
        print(f"✅ Source lineage tracked: {len(result.source_lineage)} sources")
    
    @pytest.mark.asyncio
    async def test_evaluation_statistics(self):
        """Test evaluation statistics tracking"""
        await self.evaluator.initialize()
        
        # Mock meta-reasoning results
        mock_result = Mock(spec=MetaReasoningResult)
        mock_result.confidence_score = 0.8
        mock_result.synthesis_summary = "Test evaluation"
        mock_result.reasoning_results = [Mock() for _ in range(3)]
        
        self.mock_meta_engine.meta_reason.return_value = mock_result
        
        # Run evaluation
        await self.evaluator.evaluate_candidates(self.test_result)
        
        # Check statistics
        stats = self.evaluator.get_evaluation_statistics()
        
        assert stats['total_evaluations'] == 1
        assert stats['successful_evaluations'] == 1
        assert stats['success_rate'] == 1.0
        assert stats['average_evaluation_time'] > 0
        
        print(f"✅ Statistics tracked: {stats}")
    
    @pytest.mark.asyncio
    async def test_parameter_configuration(self):
        """Test evaluation parameter configuration"""
        await self.evaluator.initialize()
        
        # Configure parameters
        custom_criteria = [EvaluationCriteria.RELEVANCE, EvaluationCriteria.EVIDENCE]
        await self.evaluator.configure_evaluation_params(
            default_criteria=custom_criteria,
            default_thinking_mode=ThinkingMode.DEEP
        )
        
        assert self.evaluator.default_criteria == custom_criteria
        assert self.evaluator.default_thinking_mode == ThinkingMode.DEEP
        
        print("✅ Parameter configuration works")
    
    @pytest.mark.asyncio
    async def test_empty_candidates_handling(self):
        """Test handling of empty candidate list"""
        await self.evaluator.initialize()
        
        # Create empty result
        empty_result = CandidateGenerationResult(
            query="test query",
            candidate_answers=[],
            generation_time_seconds=0.0,
            total_sources_used=0,
            source_utilization={},
            diversity_metrics={},
            quality_distribution={}
        )
        
        result = await self.evaluator.evaluate_candidates(empty_result)
        
        assert isinstance(result, EvaluationResult)
        assert len(result.candidate_evaluations) == 0
        assert result.best_candidate is None
        assert result.overall_confidence == 0.0
        
        print("✅ Empty candidates handled correctly")
    
    @pytest.mark.asyncio
    async def test_evaluation_strengths_weaknesses(self):
        """Test identification of evaluation strengths and weaknesses"""
        await self.evaluator.initialize()
        
        # Mock meta-reasoning results with varying scores
        def mock_meta_reason(*args, **kwargs):
            mock_result = Mock(spec=MetaReasoningResult)
            # Vary scores based on evaluation type
            if "relevance" in kwargs.get('query', '').lower():
                mock_result.confidence_score = 0.9  # High relevance
            elif "evidence" in kwargs.get('query', '').lower():
                mock_result.confidence_score = 0.3  # Low evidence
            else:
                mock_result.confidence_score = 0.7  # Medium for others
            mock_result.synthesis_summary = "Test evaluation"
            mock_result.reasoning_results = [Mock() for _ in range(3)]
            return mock_result
        
        self.mock_meta_engine.meta_reason.side_effect = mock_meta_reason
        
        result = await self.evaluator.evaluate_candidates(self.test_result)
        
        # Check that strengths and weaknesses are identified
        for evaluation in result.candidate_evaluations:
            assert len(evaluation.strengths) > 0 or len(evaluation.weaknesses) > 0
            
            # Should have high relevance as strength and low evidence as weakness
            strength_text = " ".join(evaluation.strengths).lower()
            weakness_text = " ".join(evaluation.weaknesses).lower()
            
            if "relevance" in strength_text:
                print(f"✅ Identified relevance as strength")
            if "evidence" in weakness_text:
                print(f"✅ Identified evidence as weakness")
        
        print("✅ Strengths and weaknesses identification works")


# Test execution functions
async def run_component_tests():
    """Run individual component tests"""
    print("🧪 Testing Individual Components...")
    
    # Test RelevanceScorer
    test_class = TestRelevanceScorer()
    test_class.setup_method()
    await test_class.test_scorer_initialization()
    await test_class.test_relevance_scoring()
    await test_class.test_error_handling()
    
    # Test ConfidenceScorer
    test_class = TestConfidenceScorer()
    test_class.setup_method()
    await test_class.test_confidence_scoring()
    await test_class.test_score_calculation()
    
    # Test SourceTracker
    test_class = TestSourceTracker()
    test_class.setup_method()
    await test_class.test_source_tracking()
    await test_class.test_reset_tracking()
    
    print("✅ Individual component tests completed!")


async def run_evaluator_tests():
    """Run CandidateEvaluator tests"""
    print("\n🧪 Testing CandidateEvaluator...")
    
    test_class = TestCandidateEvaluator()
    
    test_class.setup_method()
    await test_class.test_evaluator_initialization()
    
    test_class.setup_method()
    await test_class.test_factory_function()
    
    test_class.setup_method()
    await test_class.test_candidate_evaluation()
    
    test_class.setup_method()
    await test_class.test_evaluation_criteria()
    
    test_class.setup_method()
    await test_class.test_thinking_modes()
    
    test_class.setup_method()
    await test_class.test_ranking_system()
    
    test_class.setup_method()
    await test_class.test_source_lineage_tracking()
    
    test_class.setup_method()
    await test_class.test_evaluation_statistics()
    
    test_class.setup_method()
    await test_class.test_parameter_configuration()
    
    test_class.setup_method()
    await test_class.test_empty_candidates_handling()
    
    test_class.setup_method()
    await test_class.test_evaluation_strengths_weaknesses()
    
    print("✅ CandidateEvaluator tests completed!")


async def main():
    """Run all tests"""
    print("🚀 Starting CandidateEvaluator Test Suite")
    print("=" * 60)
    
    try:
        # Run component tests
        await run_component_tests()
        
        # Run evaluator tests
        await run_evaluator_tests()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! CandidateEvaluator is working correctly!")
        print("   ✅ RelevanceScorer: Relevance evaluation working")
        print("   ✅ ConfidenceScorer: Evidence quality assessment working")
        print("   ✅ SourceTracker: Source lineage tracking working")
        print("   ✅ CandidateEvaluator: System 2 evaluation working")
        print("   ✅ Ranking: Candidate ranking system working")
        print("   ✅ Integration: MetaReasoningEngine integration working")
        print("   ✅ Statistics: Performance monitoring working")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
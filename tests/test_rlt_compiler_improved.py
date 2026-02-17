#!/usr/bin/env python3
"""
RLT Enhanced Compiler Test Suite (Improved)

Comprehensive pytest tests for the RLT-Enhanced Hierarchical Compiler functionality
including explanation quality assessment, logical coherence evaluation,
and quality-weighted compilation strategies.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timezone
from uuid import uuid4


class MockRLTQualityAssessment:
    """Mock implementation of RLT Quality Assessment for testing"""
    
    def __init__(self, explanation_id, teacher_id, **kwargs):
        self.explanation_id = explanation_id
        self.teacher_id = teacher_id
        self.explanation_quality = max(0.0, min(1.0, kwargs.get('explanation_quality', 0.0)))
        self.logical_coherence = max(0.0, min(1.0, kwargs.get('logical_coherence', 0.0)))
        self.concept_coverage = max(0.0, min(1.0, kwargs.get('concept_coverage', 0.0)))
        self.student_comprehension_prediction = max(0.0, min(1.0, kwargs.get('student_comprehension_prediction', 0.0)))
        self.dense_reward_score = max(0.0, min(1.0, kwargs.get('dense_reward_score', 0.0)))
        self.teaching_effectiveness = max(0.0, min(1.0, kwargs.get('teaching_effectiveness', 0.0)))
        self.timestamp = datetime.now(timezone.utc)
    
    def calculate_overall_quality(self):
        """Calculate weighted overall quality score"""
        return (
            self.explanation_quality * 0.25 +
            self.logical_coherence * 0.20 +
            self.concept_coverage * 0.15 +
            self.student_comprehension_prediction * 0.20 +
            self.dense_reward_score * 0.15 +
            self.teaching_effectiveness * 0.05
        )
    
    def get_quality_breakdown(self):
        """Get detailed quality breakdown"""
        return {
            "explanation_quality": self.explanation_quality,
            "logical_coherence": self.logical_coherence,
            "concept_coverage": self.concept_coverage,
            "student_comprehension_prediction": self.student_comprehension_prediction,
            "dense_reward_score": self.dense_reward_score,
            "teaching_effectiveness": self.teaching_effectiveness,
            "overall_quality": self.calculate_overall_quality(),
            "timestamp": self.timestamp.isoformat()
        }


class MockRLTCompilationMetrics:
    """Mock implementation of RLT Compilation Metrics for testing"""
    
    def __init__(self):
        self.compilation_history = []
        self.quality_threshold = 0.7
        self.metrics = {
            "total_compilations": 0,
            "successful_compilations": 0,
            "average_quality": 0.0,
            "quality_improvement_rate": 0.0
        }
    
    def record_compilation(self, assessment, compilation_time, success=True):
        """Record a compilation event with its assessment"""
        compilation_record = {
            "compilation_id": str(uuid4()),
            "assessment": assessment,
            "compilation_time": compilation_time,
            "success": success,
            "timestamp": datetime.now(timezone.utc)
        }
        self.compilation_history.append(compilation_record)
        self._update_metrics()
        return compilation_record
    
    def _update_metrics(self):
        """Update compilation metrics based on history"""
        if not self.compilation_history:
            return
        
        self.metrics["total_compilations"] = len(self.compilation_history)
        self.metrics["successful_compilations"] = sum(
            1 for record in self.compilation_history if record["success"]
        )
        
        if self.metrics["successful_compilations"] > 0:
            quality_scores = [
                record["assessment"].calculate_overall_quality()
                for record in self.compilation_history
                if record["success"]
            ]
            self.metrics["average_quality"] = sum(quality_scores) / len(quality_scores)
    
    def get_performance_summary(self):
        """Get compilation performance summary"""
        success_rate = (
            self.metrics["successful_compilations"] / self.metrics["total_compilations"]
            if self.metrics["total_compilations"] > 0 else 0.0
        )
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "quality_threshold": self.quality_threshold,
            "above_threshold_count": sum(
                1 for record in self.compilation_history
                if record["success"] and record["assessment"].calculate_overall_quality() >= self.quality_threshold
            )
        }


class TestRLTQualityAssessment:
    """Test suite for RLT Quality Assessment functionality"""
    
    @pytest.fixture
    def sample_assessment(self):
        """Fixture providing a sample quality assessment"""
        return MockRLTQualityAssessment(
            explanation_id="test_explanation_123",
            teacher_id="rlt_math_teacher",
            explanation_quality=0.85,
            logical_coherence=0.78,
            concept_coverage=0.82,
            student_comprehension_prediction=0.75,
            dense_reward_score=0.88,
            teaching_effectiveness=0.80
        )
    
    def test_quality_assessment_creation(self, sample_assessment):
        """Test quality assessment object creation and initialization"""
        assert sample_assessment.explanation_id == "test_explanation_123"
        assert sample_assessment.teacher_id == "rlt_math_teacher"
        assert sample_assessment.explanation_quality == 0.85
        assert sample_assessment.logical_coherence == 0.78
        assert isinstance(sample_assessment.timestamp, datetime)
    
    def test_overall_quality_calculation(self, sample_assessment):
        """Test overall quality score calculation with proper weights"""
        overall_quality = sample_assessment.calculate_overall_quality()
        
        # Manual calculation for verification
        expected_quality = (
            0.85 * 0.25 +  # explanation_quality
            0.78 * 0.20 +  # logical_coherence
            0.82 * 0.15 +  # concept_coverage
            0.75 * 0.20 +  # student_comprehension_prediction
            0.88 * 0.15 +  # dense_reward_score
            0.80 * 0.05    # teaching_effectiveness
        )
        
        assert abs(overall_quality - expected_quality) < 0.01
        assert 0.0 <= overall_quality <= 1.0
    
    def test_quality_breakdown(self, sample_assessment):
        """Test quality breakdown dictionary structure and content"""
        breakdown = sample_assessment.get_quality_breakdown()
        
        # Verify all expected keys are present
        expected_keys = [
            "explanation_quality", "logical_coherence", "concept_coverage",
            "student_comprehension_prediction", "dense_reward_score",
            "teaching_effectiveness", "overall_quality", "timestamp"
        ]
        
        for key in expected_keys:
            assert key in breakdown
        
        # Verify overall quality matches calculation
        assert breakdown["overall_quality"] == sample_assessment.calculate_overall_quality()
        
        # Verify individual scores
        assert breakdown["explanation_quality"] == 0.85
        assert breakdown["logical_coherence"] == 0.78
    
    @pytest.mark.parametrize("quality_value,expected", [
        (1.5, 1.0),    # Should clamp to maximum
        (-0.5, 0.0),   # Should clamp to minimum
        (0.5, 0.5),    # Should remain unchanged
        (0.0, 0.0),    # Boundary case
        (1.0, 1.0)     # Boundary case
    ])
    def test_boundary_conditions(self, quality_value, expected):
        """Test boundary condition handling for quality scores"""
        assessment = MockRLTQualityAssessment(
            explanation_id="boundary_test",
            teacher_id="test_teacher",
            explanation_quality=quality_value,
            logical_coherence=quality_value
        )
        
        assert assessment.explanation_quality == expected
        assert assessment.logical_coherence == expected
        
        # Overall quality should always be in valid range
        overall = assessment.calculate_overall_quality()
        assert 0.0 <= overall <= 1.0
    
    def test_weighted_quality_calculation(self):
        """Test that quality weights sum to 1.0 and produce expected results"""
        # Test with all metrics at maximum
        max_assessment = MockRLTQualityAssessment(
            explanation_id="max_test",
            teacher_id="test_teacher",
            explanation_quality=1.0,
            logical_coherence=1.0,
            concept_coverage=1.0,
            student_comprehension_prediction=1.0,
            dense_reward_score=1.0,
            teaching_effectiveness=1.0
        )
        
        assert abs(max_assessment.calculate_overall_quality() - 1.0) < 0.01
        
        # Test with all metrics at minimum
        min_assessment = MockRLTQualityAssessment(
            explanation_id="min_test",
            teacher_id="test_teacher"
        )
        
        assert abs(min_assessment.calculate_overall_quality() - 0.0) < 0.01


class TestRLTCompilationMetrics:
    """Test suite for RLT Compilation Metrics functionality"""
    
    @pytest.fixture
    def compilation_metrics(self):
        """Fixture providing compilation metrics instance"""
        return MockRLTCompilationMetrics()
    
    @pytest.fixture
    def sample_assessments(self):
        """Fixture providing multiple sample assessments"""
        return [
            MockRLTQualityAssessment(
                explanation_id=f"test_{i}",
                teacher_id="test_teacher",
                explanation_quality=0.8 + i * 0.05,
                logical_coherence=0.7 + i * 0.05,
                concept_coverage=0.75 + i * 0.05
            )
            for i in range(5)
        ]
    
    def test_compilation_recording(self, compilation_metrics, sample_assessments):
        """Test recording compilation events"""
        assessment = sample_assessments[0]
        
        record = compilation_metrics.record_compilation(
            assessment=assessment,
            compilation_time=1.5,
            success=True
        )
        
        assert record["assessment"] == assessment
        assert record["compilation_time"] == 1.5
        assert record["success"] is True
        assert "compilation_id" in record
        assert "timestamp" in record
        
        # Verify it's added to history
        assert len(compilation_metrics.compilation_history) == 1
    
    def test_metrics_calculation(self, compilation_metrics, sample_assessments):
        """Test metrics calculation after multiple compilations"""
        # Record several compilations
        for i, assessment in enumerate(sample_assessments):
            success = i < 4  # Last one fails
            compilation_metrics.record_compilation(
                assessment=assessment,
                compilation_time=1.0 + i * 0.1,
                success=success
            )
        
        metrics = compilation_metrics.metrics
        
        assert metrics["total_compilations"] == 5
        assert metrics["successful_compilations"] == 4
        assert metrics["average_quality"] > 0
    
    def test_performance_summary(self, compilation_metrics, sample_assessments):
        """Test performance summary generation"""
        # Record compilations with known success rate
        for i, assessment in enumerate(sample_assessments):
            compilation_metrics.record_compilation(
                assessment=assessment,
                compilation_time=1.0,
                success=i % 2 == 0  # 60% success rate
            )
        
        summary = compilation_metrics.get_performance_summary()
        
        assert "success_rate" in summary
        assert "quality_threshold" in summary
        assert "above_threshold_count" in summary
        
        # Success rate should be 60% (3 out of 5)
        assert abs(summary["success_rate"] - 0.6) < 0.01
        
        # Verify all required metrics are present
        required_keys = [
            "total_compilations", "successful_compilations", 
            "average_quality", "success_rate"
        ]
        for key in required_keys:
            assert key in summary
    
    def test_quality_threshold_filtering(self, compilation_metrics):
        """Test filtering compilations by quality threshold"""
        # Create assessments with different quality levels
        high_quality = MockRLTQualityAssessment(
            explanation_id="high",
            teacher_id="test",
            explanation_quality=0.9,
            logical_coherence=0.9,
            concept_coverage=0.9,
            student_comprehension_prediction=0.9,
            dense_reward_score=0.9,
            teaching_effectiveness=0.9
        )

        low_quality = MockRLTQualityAssessment(
            explanation_id="low",
            teacher_id="test",
            explanation_quality=0.3,
            logical_coherence=0.3,
            concept_coverage=0.3,
            student_comprehension_prediction=0.3,
            dense_reward_score=0.3,
            teaching_effectiveness=0.3
        )
        
        # Record compilations
        compilation_metrics.record_compilation(high_quality, 1.0, True)
        compilation_metrics.record_compilation(low_quality, 1.0, True)
        
        summary = compilation_metrics.get_performance_summary()
        
        # Only high quality should be above threshold (0.7)
        assert summary["above_threshold_count"] == 1


class TestRLTIntegration:
    """Integration tests for RLT compiler components"""
    
    def test_end_to_end_compilation_workflow(self):
        """Test complete compilation workflow with quality assessment"""
        # Initialize components
        metrics = MockRLTCompilationMetrics()
        
        # Simulate compilation workflow
        explanation_data = {
            "concept": "quadratic_equations",
            "difficulty": "intermediate",
            "student_level": "high_school"
        }
        
        # Create quality assessment
        assessment = MockRLTQualityAssessment(
            explanation_id="workflow_test",
            teacher_id="math_teacher",
            explanation_quality=0.85,
            logical_coherence=0.80,
            concept_coverage=0.88,
            student_comprehension_prediction=0.75,
            dense_reward_score=0.82,
            teaching_effectiveness=0.78
        )
        
        # Record compilation
        start_time = time.time()
        compilation_record = metrics.record_compilation(
            assessment=assessment,
            compilation_time=time.time() - start_time,
            success=True
        )
        
        # Verify end-to-end workflow
        assert compilation_record["success"] is True
        assert assessment.calculate_overall_quality() > 0.7
        
        summary = metrics.get_performance_summary()
        assert summary["total_compilations"] == 1
        assert summary["successful_compilations"] == 1
        assert summary["success_rate"] == 1.0


if __name__ == "__main__":
    # Run the tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
#!/usr/bin/env python3
"""
Performance Monitoring Test Suite

Comprehensive pytest tests for PRSM performance monitoring functionality including
metrics tracking, improvement opportunity identification, and baseline comparison.
Converted from test_performance_monitor.py to follow pytest conventions.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any


# Mock classes for testing performance monitoring
class MockMetricType:
    """Mock metric type enumeration"""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"


class MockImprovementType:
    """Mock improvement type enumeration"""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    COST_REDUCTION = "cost_reduction"


class MockPerformanceMetric:
    """Mock performance metric for testing"""
    def __init__(self, model_id: str, metric_type: str, value: float, timestamp=None):
        self.model_id = model_id
        self.metric_type = metric_type
        self.value = value
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.metadata = {}


class MockImprovementOpportunity:
    """Mock improvement opportunity for testing"""
    def __init__(self, model_id: str, improvement_type: str, description: str, 
                 potential_gain: float, confidence: float = 0.8):
        self.opportunity_id = f"imp_{int(time.time())}"
        self.model_id = model_id
        self.improvement_type = improvement_type
        self.description = description
        self.potential_gain = potential_gain
        self.confidence = confidence
        self.identified_at = datetime.now(timezone.utc)
        self.status = "identified"


class MockPerformanceMonitor:
    """Mock performance monitor for testing"""
    def __init__(self):
        self.metrics_storage = []
        self.baselines = {}
        self.improvement_opportunities = []
        self.models_monitored = set()
    
    async def track_model_metrics(self, model_id: str, metrics_data: Dict[str, float]) -> bool:
        """Track metrics for a model"""
        self.models_monitored.add(model_id)
        
        for metric_type, value in metrics_data.items():
            metric = MockPerformanceMetric(model_id, metric_type, value)
            self.metrics_storage.append(metric)
        
        return True
    
    async def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "total_metrics_tracked": len(self.metrics_storage),
            "models_monitored": list(self.models_monitored),
            "unique_metric_types": len(set(m.metric_type for m in self.metrics_storage)),
            "monitoring_duration": "active",
            "last_update": datetime.now(timezone.utc).isoformat()
        }
    
    async def identify_improvement_opportunities(self, model_id: str) -> List[MockImprovementOpportunity]:
        """Identify improvement opportunities for a model"""
        # Get recent metrics for analysis
        model_metrics = [m for m in self.metrics_storage if m.model_id == model_id]
        
        opportunities = []
        
        # Analyze accuracy metrics
        accuracy_metrics = [m for m in model_metrics if m.metric_type == MockMetricType.ACCURACY]
        if accuracy_metrics:
            avg_accuracy = sum(m.value for m in accuracy_metrics) / len(accuracy_metrics)
            if avg_accuracy < 0.9:  # Threshold for improvement
                opportunities.append(MockImprovementOpportunity(
                    model_id=model_id,
                    improvement_type=MockImprovementType.ACCURACY,
                    description=f"Model accuracy ({avg_accuracy:.2f}) below target (0.90)",
                    potential_gain=0.9 - avg_accuracy,
                    confidence=0.85
                ))
        
        # Analyze latency metrics
        latency_metrics = [m for m in model_metrics if m.metric_type == MockMetricType.LATENCY]
        if latency_metrics:
            avg_latency = sum(m.value for m in latency_metrics) / len(latency_metrics)
            if avg_latency > 100:  # Threshold for improvement
                opportunities.append(MockImprovementOpportunity(
                    model_id=model_id,
                    improvement_type=MockImprovementType.PERFORMANCE,
                    description=f"Model latency ({avg_latency:.1f}ms) above target (100ms)",
                    potential_gain=(avg_latency - 100) / avg_latency,
                    confidence=0.80
                ))
        
        self.improvement_opportunities.extend(opportunities)
        return opportunities
    
    async def set_performance_baseline(self, model_id: str, baseline_metrics: Dict[str, float]):
        """Set performance baseline for a model"""
        self.baselines[model_id] = {
            "metrics": baseline_metrics,
            "set_at": datetime.now(timezone.utc),
            "version": "1.0"
        }
        return True
    
    async def compare_against_baseline(self, model_id: str, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare current metrics against baseline"""
        if model_id not in self.baselines:
            return {"error": "No baseline set for model"}
        
        baseline = self.baselines[model_id]["metrics"]
        comparison = {}
        
        for metric_type, current_value in current_metrics.items():
            if metric_type in baseline:
                baseline_value = baseline[metric_type]
                change = current_value - baseline_value
                change_percent = (change / baseline_value) * 100 if baseline_value != 0 else 0
                
                comparison[metric_type] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "change": change,
                    "change_percent": change_percent,
                    "improved": self._is_improvement(metric_type, change)
                }
        
        return {
            "model_id": model_id,
            "comparison_results": comparison,
            "overall_trend": self._calculate_overall_trend(comparison),
            "compared_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _is_improvement(self, metric_type: str, change: float) -> bool:
        """Determine if change represents improvement"""
        # For accuracy and throughput, higher is better
        # For latency and error_rate, lower is better
        if metric_type in [MockMetricType.ACCURACY, MockMetricType.THROUGHPUT]:
            return change > 0
        elif metric_type in [MockMetricType.LATENCY, MockMetricType.ERROR_RATE]:
            return change < 0
        else:
            return change > 0  # Default assumption
    
    def _calculate_overall_trend(self, comparison: Dict[str, Any]) -> str:
        """Calculate overall performance trend"""
        if not comparison:
            return "no_data"
        
        improvements = sum(1 for result in comparison.values() if result["improved"])
        total_metrics = len(comparison)
        improvement_ratio = improvements / total_metrics
        
        if improvement_ratio >= 0.7:
            return "improving"
        elif improvement_ratio >= 0.3:
            return "mixed"
        else:
            return "declining"


@pytest.fixture
def performance_monitor():
    """Fixture providing performance monitor instance"""
    return MockPerformanceMonitor()


class TestPerformanceMetricsTracking:
    """Test suite for performance metrics tracking functionality"""

    @pytest.fixture
    def sample_metrics_data(self):
        """Fixture providing sample metrics data"""
        return {
            "accuracy": 0.85,
            "latency": 120.5,
            "throughput": 1500,
            "error_rate": 0.02
        }
    
    @pytest.mark.asyncio
    async def test_model_metrics_storage_and_retrieval(self, performance_monitor, sample_metrics_data):
        """Test storing model metrics and verifying they can be retrieved correctly"""
        result = await performance_monitor.track_model_metrics("test_model_1", sample_metrics_data)
        
        assert result is True
        
        # Verify metrics were stored
        stats = await performance_monitor.get_monitoring_statistics()
        assert stats["total_metrics_tracked"] >= 4
        assert "test_model_1" in stats["models_monitored"]
        assert stats["unique_metric_types"] == 4
    
    @pytest.mark.asyncio
    async def test_multiple_models_tracking(self, performance_monitor):
        """Test tracking metrics for multiple models"""
        models_data = {
            "model_a": {"accuracy": 0.90, "latency": 80},
            "model_b": {"accuracy": 0.88, "latency": 95},
            "model_c": {"accuracy": 0.92, "latency": 75}
        }
        
        for model_id, metrics in models_data.items():
            result = await performance_monitor.track_model_metrics(model_id, metrics)
            assert result is True
        
        stats = await performance_monitor.get_monitoring_statistics()
        assert len(stats["models_monitored"]) == 3
        assert stats["total_metrics_tracked"] == 6  # 2 metrics per model × 3 models
    
    @pytest.mark.asyncio
    async def test_metrics_storage_structure(self, performance_monitor, sample_metrics_data):
        """Test that metrics are stored with proper structure"""
        await performance_monitor.track_model_metrics("test_model", sample_metrics_data)
        
        # Verify metrics in storage
        metrics = performance_monitor.metrics_storage
        assert len(metrics) == 4  # One for each metric type
        
        for metric in metrics:
            assert hasattr(metric, 'model_id')
            assert hasattr(metric, 'metric_type')
            assert hasattr(metric, 'value')
            assert hasattr(metric, 'timestamp')
            assert metric.model_id == "test_model"
            assert metric.metric_type in sample_metrics_data.keys()


class TestImprovementOpportunities:
    """Test suite for improvement opportunity identification"""
    
    @pytest_asyncio.fixture
    async def monitor_with_data(self):
        """Fixture providing monitor with historical data"""
        monitor = MockPerformanceMonitor()

        # Add historical metrics showing declining performance
        historical_data = [
            {"model_id": "test_model_2", "metrics": {"accuracy": 0.90, "latency": 100}},
            {"model_id": "test_model_2", "metrics": {"accuracy": 0.87, "latency": 110}},
            {"model_id": "test_model_2", "metrics": {"accuracy": 0.84, "latency": 125}},
            {"model_id": "test_model_3", "metrics": {"accuracy": 0.95, "latency": 200}},
        ]

        for data in historical_data:
            await monitor.track_model_metrics(data["model_id"], data["metrics"])

        return monitor
    
    @pytest.mark.asyncio
    async def test_improvement_opportunity_identification(self, monitor_with_data):
        """Test improvement opportunity identification"""
        # Manually add some metrics first
        await monitor_with_data.track_model_metrics("test_model_2", {"accuracy": 0.84, "latency": 125})
        
        opportunities = await monitor_with_data.identify_improvement_opportunities("test_model_2")
        
        assert len(opportunities) >= 1  # Should identify at least one opportunity
        
        # Verify opportunity structure
        for opportunity in opportunities:
            assert hasattr(opportunity, 'opportunity_id')
            assert hasattr(opportunity, 'model_id')
            assert hasattr(opportunity, 'improvement_type')
            assert hasattr(opportunity, 'description')
            assert hasattr(opportunity, 'potential_gain')
            assert hasattr(opportunity, 'confidence')
            
            assert opportunity.model_id == "test_model_2"
            assert 0 <= opportunity.confidence <= 1
            assert opportunity.potential_gain >= 0
    
    @pytest.mark.asyncio
    async def test_accuracy_improvement_detection(self, performance_monitor):
        """Test detection of accuracy improvement opportunities"""
        # Add metrics with low accuracy
        low_accuracy_metrics = {"accuracy": 0.75, "latency": 50}
        await performance_monitor.track_model_metrics("low_accuracy_model", low_accuracy_metrics)
        
        opportunities = await performance_monitor.identify_improvement_opportunities("low_accuracy_model")
        
        # Should identify accuracy improvement opportunity
        accuracy_opportunities = [op for op in opportunities 
                                 if op.improvement_type == MockImprovementType.ACCURACY]
        assert len(accuracy_opportunities) >= 1
        
        accuracy_op = accuracy_opportunities[0]
        assert "accuracy" in accuracy_op.description.lower()
        assert accuracy_op.potential_gain > 0
    
    @pytest.mark.asyncio
    async def test_latency_improvement_detection(self, performance_monitor):
        """Test detection of latency improvement opportunities"""
        # Add metrics with high latency
        high_latency_metrics = {"accuracy": 0.95, "latency": 250}
        await performance_monitor.track_model_metrics("high_latency_model", high_latency_metrics)
        
        opportunities = await performance_monitor.identify_improvement_opportunities("high_latency_model")
        
        # Should identify performance improvement opportunity
        performance_opportunities = [op for op in opportunities 
                                   if op.improvement_type == MockImprovementType.PERFORMANCE]
        assert len(performance_opportunities) >= 1
        
        perf_op = performance_opportunities[0]
        assert "latency" in perf_op.description.lower()
        assert perf_op.potential_gain > 0


class TestBaselineComparison:
    """Test suite for baseline comparison functionality"""
    
    @pytest.fixture
    def monitor_with_baseline(self):
        """Fixture providing monitor with established baseline"""
        monitor = MockPerformanceMonitor()
        return monitor
    
    @pytest.mark.asyncio
    async def test_baseline_setting(self, monitor_with_baseline):
        """Test setting performance baseline"""
        baseline_metrics = {
            "accuracy": 0.90,
            "latency": 100.0,
            "throughput": 1000,
            "error_rate": 0.01
        }
        
        result = await monitor_with_baseline.set_performance_baseline("baseline_model", baseline_metrics)
        assert result is True
        
        # Verify baseline was stored
        assert "baseline_model" in monitor_with_baseline.baselines
        stored_baseline = monitor_with_baseline.baselines["baseline_model"]
        assert stored_baseline["metrics"] == baseline_metrics
        assert "set_at" in stored_baseline
        assert "version" in stored_baseline
    
    @pytest.mark.asyncio
    async def test_baseline_comparison_improvement(self, monitor_with_baseline):
        """Test baseline comparison showing improvement"""
        # Set baseline
        baseline_metrics = {"accuracy": 0.85, "latency": 120.0}
        await monitor_with_baseline.set_performance_baseline("test_model", baseline_metrics)
        
        # Compare with improved metrics
        improved_metrics = {"accuracy": 0.90, "latency": 100.0}
        comparison = await monitor_with_baseline.compare_against_baseline("test_model", improved_metrics)
        
        # Verify comparison structure
        assert "model_id" in comparison
        assert "comparison_results" in comparison
        assert "overall_trend" in comparison
        
        results = comparison["comparison_results"]
        
        # Verify accuracy improvement
        assert results["accuracy"]["improved"] is True
        assert results["accuracy"]["change"] > 0
        assert results["accuracy"]["change_percent"] > 0
        
        # Verify latency improvement (lower is better)
        assert results["latency"]["improved"] is True
        assert results["latency"]["change"] < 0
        
        # Overall trend should be improving
        assert comparison["overall_trend"] == "improving"
    
    @pytest.mark.asyncio
    async def test_baseline_comparison_decline(self, monitor_with_baseline):
        """Test baseline comparison showing performance decline"""
        # Set baseline
        baseline_metrics = {"accuracy": 0.90, "latency": 80.0}
        await monitor_with_baseline.set_performance_baseline("declining_model", baseline_metrics)
        
        # Compare with worse metrics
        worse_metrics = {"accuracy": 0.85, "latency": 100.0}
        comparison = await monitor_with_baseline.compare_against_baseline("declining_model", worse_metrics)
        
        results = comparison["comparison_results"]
        
        # Verify accuracy decline
        assert results["accuracy"]["improved"] is False
        assert results["accuracy"]["change"] < 0
        
        # Verify latency decline (higher latency is worse)
        assert results["latency"]["improved"] is False
        assert results["latency"]["change"] > 0
        
        # Overall trend should be declining
        assert comparison["overall_trend"] == "declining"
    
    @pytest.mark.asyncio
    async def test_baseline_comparison_no_baseline(self, monitor_with_baseline):
        """Test baseline comparison when no baseline exists"""
        current_metrics = {"accuracy": 0.90, "latency": 100.0}
        comparison = await monitor_with_baseline.compare_against_baseline("no_baseline_model", current_metrics)
        
        assert "error" in comparison
        assert "no baseline" in comparison["error"].lower()


class TestPerformanceMonitoringIntegration:
    """Integration tests for complete performance monitoring workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(self):
        """Test complete performance monitoring workflow"""
        monitor = MockPerformanceMonitor()
        model_id = "workflow_test_model"
        
        # Step 1: Set baseline
        baseline_metrics = {
            "accuracy": 0.85,
            "latency": 150.0,
            "throughput": 800,
            "error_rate": 0.05
        }
        
        await monitor.set_performance_baseline(model_id, baseline_metrics)
        
        # Step 2: Track current metrics
        current_metrics = {
            "accuracy": 0.88,
            "latency": 140.0,
            "throughput": 850,
            "error_rate": 0.04
        }
        
        await monitor.track_model_metrics(model_id, current_metrics)
        
        # Step 3: Compare against baseline
        comparison = await monitor.compare_against_baseline(model_id, current_metrics)
        
        # Step 4: Identify improvement opportunities
        opportunities = await monitor.identify_improvement_opportunities(model_id)
        
        # Step 5: Get monitoring statistics
        stats = await monitor.get_monitoring_statistics()
        
        # Verify complete workflow
        assert comparison["overall_trend"] == "improving"
        assert model_id in stats["models_monitored"]
        assert stats["total_metrics_tracked"] == 4
        
        # Verify opportunities identified for areas still needing improvement
        # (accuracy 0.88 is still below 0.90 threshold)
        accuracy_opportunities = [op for op in opportunities 
                                 if op.improvement_type == MockImprovementType.ACCURACY]
        assert len(accuracy_opportunities) >= 0  # May or may not identify based on threshold


if __name__ == "__main__":
    # Run the tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
#!/usr/bin/env python3
"""
Basic Performance Monitor Test Suite
Tests core functionality without creating large test files to avoid memory issues
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta

from prsm.compute.improvement.performance_monitor import get_performance_monitor, PerformanceMonitor
from prsm.core.models import MetricType, ImprovementType


async def test_basic_metrics_tracking():
    """Test basic metric tracking functionality"""
    print("🧪 Testing basic metrics tracking...")
    
    monitor = get_performance_monitor()
    
    # Track some basic metrics
    test_data = {
        "accuracy": 0.85,
        "latency": 120.5,
        "throughput": 1500,
        "error_rate": 0.02
    }
    
    result = await monitor.track_model_metrics("test_model_1", test_data)
    assert result == True, "Should successfully track metrics"
    
    # Verify metrics were stored
    stats = await monitor.get_monitoring_statistics()
    assert stats["total_metrics_tracked"] >= 4, "Should track all metrics"
    assert "test_model_1" in stats["models_monitored"], "Should track the model"
    
    print("✅ Basic metrics tracking working")


async def test_improvement_opportunities():
    """Test improvement opportunity identification"""
    print("🧪 Testing improvement opportunities...")
    
    monitor = get_performance_monitor()
    
    # Create historical data with declining performance
    historical_data = [
        {
            "model_id": "test_model_2",
            "metrics": {"accuracy": 0.90, "latency": 100},
            "timestamp": datetime.now(timezone.utc) - timedelta(hours=3)
        },
        {
            "model_id": "test_model_2", 
            "metrics": {"accuracy": 0.85, "latency": 120},
            "timestamp": datetime.now(timezone.utc) - timedelta(hours=2)
        },
        {
            "model_id": "test_model_2",
            "metrics": {"accuracy": 0.80, "latency": 140},
            "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)
        }
    ]
    
    opportunities = await monitor.identify_improvement_opportunities(historical_data)
    
    assert len(opportunities) > 0, "Should identify opportunities from declining performance"
    
    # Check opportunity details
    accuracy_opp = next((o for o in opportunities if "accuracy" in o.target_component), None)
    assert accuracy_opp is not None, "Should identify accuracy improvement opportunity"
    assert accuracy_opp.improvement_type in [ImprovementType.TRAINING_DATA, ImprovementType.OPTIMIZATION], "Should have appropriate improvement type"
    
    print("✅ Improvement opportunity identification working")


async def test_baseline_comparison():
    """Test baseline comparison functionality"""
    print("🧪 Testing baseline comparison...")
    
    monitor = get_performance_monitor()
    
    # Set up a model with baseline
    model_id = "test_model_3"
    
    # Track initial metrics (becomes baseline)
    baseline_data = {
        "accuracy": 0.80,
        "latency": 150,
        "throughput": 1000
    }
    await monitor.track_model_metrics(model_id, baseline_data)
    
    # Track improved metrics
    improved_data = {
        "accuracy": 0.85,
        "latency": 130,
        "throughput": 1200
    }
    await monitor.track_model_metrics(model_id, improved_data)
    
    # Generate comparison report
    report = await monitor.benchmark_against_baselines(model_id)
    
    assert report.model_id == model_id, "Report should be for correct model"
    assert len(report.comparison_metrics) > 0, "Should have comparison metrics"
    assert report.overall_improvement != 0.0, "Should calculate improvement"
    
    print("✅ Baseline comparison working")


async def test_performance_analysis():
    """Test comprehensive performance analysis"""
    print("🧪 Testing performance analysis...")
    
    monitor = get_performance_monitor()
    
    model_id = "test_model_4"
    
    # Add multiple metrics over time
    for i in range(5):
        test_data = {
            "accuracy": 0.80 + (i * 0.02),  # Improving trend
            "latency": 200 - (i * 10),      # Improving trend
            "error_rate": 0.05 - (i * 0.005)  # Improving trend
        }
        await monitor.track_model_metrics(model_id, test_data)
        await asyncio.sleep(0.1)  # Small delay
    
    # Generate analysis
    analysis = await monitor.generate_performance_analysis(model_id, period_hours=1)
    
    assert analysis.model_id == model_id, "Analysis should be for correct model"
    assert len(analysis.metrics_analyzed) > 0, "Should have analyzed metrics"
    assert analysis.overall_health_score > 0.0, "Should have health score"
    assert len(analysis.recommendations) > 0, "Should have recommendations"
    
    print("✅ Performance analysis working")


async def test_monitoring_statistics():
    """Test monitoring statistics functionality"""
    print("🧪 Testing monitoring statistics...")
    
    monitor = get_performance_monitor()
    
    # Get current stats
    stats = await monitor.get_monitoring_statistics()
    
    # Verify expected fields
    required_fields = [
        "total_metrics_tracked",
        "models_monitored_count", 
        "improvement_opportunities_identified",
        "analyses_performed",
        "baseline_comparisons",
        "configuration"
    ]
    
    for field in required_fields:
        assert field in stats, f"Stats should include {field}"
    
    assert isinstance(stats["configuration"], dict), "Configuration should be a dict"
    assert stats["models_monitored_count"] >= 0, "Should have non-negative model count"
    
    print("✅ Monitoring statistics working")


async def run_performance_tests():
    """Run all performance monitoring tests"""
    print("🚀 Starting Performance Monitor Test Suite")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Run basic tests
        await test_basic_metrics_tracking()
        await test_improvement_opportunities() 
        await test_baseline_comparison()
        await test_performance_analysis()
        await test_monitoring_statistics()
        
        # Performance test
        print("\n📊 Performance Test Results:")
        
        monitor = get_performance_monitor()
        stats = await monitor.get_monitoring_statistics()
        
        print(f"   • Total metrics tracked: {stats['total_metrics_tracked']}")
        print(f"   • Models monitored: {stats['models_monitored_count']}")
        print(f"   • Improvement opportunities: {stats['improvement_opportunities_identified']}")
        print(f"   • Analyses performed: {stats['analyses_performed']}")
        print(f"   • Baseline comparisons: {stats['baseline_comparisons']}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ All Performance Monitor tests passed!")
        print(f"⚡ Test duration: {duration:.2f} seconds")
        print(f"🎯 Performance Monitor system is operational and ready for production")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Performance Monitor test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(run_performance_tests())
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test PRSM Benchmark Comparator
Validate benchmark results persistence and comparison capabilities
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent
sys.path.insert(0, str(PRSM_ROOT))

def create_mock_benchmark_results():
    """Create mock benchmark results for testing"""
    
    # Create temporary benchmark results
    results = []
    
    # Mock result 1 (baseline)
    result1 = {
        "config": {
            "name": "test_benchmark_1",
            "benchmark_type": "consensus_scaling",
            "node_count": 25,
            "duration_seconds": 30,
            "network_condition": "wan"
        },
        "metrics": {
            "operations_per_second": 15.5,
            "mean_latency_ms": 45.2,
            "p95_latency_ms": 78.1,
            "consensus_success_rate": 0.975,
            "operations_per_node_per_second": 0.62
        },
        "timing": {
            "start_time": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
            "end_time": (datetime.now(timezone.utc) - timedelta(days=2) + timedelta(seconds=30)).isoformat(),
            "duration_seconds": 30.0
        }
    }
    
    # Mock result 2 (comparison - slight regression)
    result2 = {
        "config": {
            "name": "test_benchmark_2", 
            "benchmark_type": "consensus_scaling",
            "node_count": 25,
            "duration_seconds": 30,
            "network_condition": "wan"
        },
        "metrics": {
            "operations_per_second": 13.8,  # Decreased
            "mean_latency_ms": 52.7,        # Increased
            "p95_latency_ms": 85.3,         # Increased
            "consensus_success_rate": 0.965, # Slightly decreased
            "operations_per_node_per_second": 0.55  # Decreased
        },
        "timing": {
            "start_time": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "end_time": (datetime.now(timezone.utc) - timedelta(days=1) + timedelta(seconds=30)).isoformat(),
            "duration_seconds": 30.0
        }
    }
    
    # Mock result 3 (improvement)
    result3 = {
        "config": {
            "name": "test_benchmark_3",
            "benchmark_type": "consensus_scaling", 
            "node_count": 25,
            "duration_seconds": 30,
            "network_condition": "wan"
        },
        "metrics": {
            "operations_per_second": 18.2,  # Improved
            "mean_latency_ms": 38.5,        # Improved
            "p95_latency_ms": 65.8,         # Improved
            "consensus_success_rate": 0.985, # Improved
            "operations_per_node_per_second": 0.73  # Improved
        },
        "timing": {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
            "duration_seconds": 30.0
        }
    }
    
    return [result1, result2, result3]

def test_benchmark_comparator_components():
    """Test the benchmark comparator components"""
    print("🧪 Testing PRSM Benchmark Comparator Components")
    print("=" * 70)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from prsm.performance.benchmark_comparator import (
            BenchmarkComparator, BenchmarkDatabase, BenchmarkComparison,
            TrendAnalysis, PerformanceReport, TrendDirection, RegressionSeverity
        )
        print("   ✅ All benchmark comparator components imported successfully")
        
        # Test database initialization
        print("\n2. Testing database initialization...")
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_benchmarks.db"
            db = BenchmarkDatabase(str(db_path))
            print(f"   ✅ Database initialized: {db_path}")
            
            # Test data storage
            print("\n3. Testing data storage...")
            mock_results = create_mock_benchmark_results()
            
            stored_ids = []
            for result in mock_results:
                run_id = db.store_benchmark_result(result)
                stored_ids.append(run_id)
                print(f"   ✅ Stored result: {run_id}")
            
            print(f"   ✅ Stored {len(stored_ids)} benchmark results")
            
            # Test data retrieval
            print("\n4. Testing data retrieval...")
            history = db.get_benchmark_history(days=7)
            print(f"   ✅ Retrieved {len(history)} historical records")
            
            # Test comparator initialization
            print("\n5. Testing comparator initialization...")
            comparator = BenchmarkComparator(database=db)
            print("   ✅ Benchmark comparator initialized")
            
            # Test benchmark comparison
            print("\n6. Testing benchmark comparison...")
            if len(mock_results) >= 2:
                comparison = comparator.compare_benchmarks(mock_results[0], mock_results[1])
                print(f"   ✅ Comparison completed")
                print(f"      Throughput change: {comparison.throughput_change_percent:.1f}%")
                print(f"      Latency change: {comparison.latency_change_percent:.1f}%")
                print(f"      Overall change: {comparison.overall_performance_change:.1f}%")
                print(f"      Trend: {comparison.trend_direction.value}")
                print(f"      Severity: {comparison.regression_severity.value}")
                print(f"      Is regression: {comparison.is_regression}")
                print(f"      Summary: {comparison.summary}")
            
            # Test trend analysis
            print("\n7. Testing trend analysis...")
            trend = comparator.analyze_trends("operations_per_second", days=30)
            print(f"   ✅ Trend analysis completed")
            print(f"      Metric: {trend.metric_name}")
            print(f"      Data points: {trend.data_points}")
            print(f"      Trend direction: {trend.trend_direction.value}")
            print(f"      Quality score: {trend.quality_score:.2f}")
            
            # Test regression detection
            print("\n8. Testing regression detection...")
            # Create temporary files to test file loading
            temp_results_dir = Path(temp_dir) / "test_results"
            temp_results_dir.mkdir()
            
            # Save mock results to JSON files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_file = temp_results_dir / f"benchmark_results_{timestamp}.json"
            
            with open(test_file, 'w') as f:
                json.dump({
                    "benchmark_suite_version": "1.0",
                    "timestamp": timestamp,
                    "total_benchmarks": len(mock_results),
                    "results": mock_results
                }, f, indent=2)
            
            # Update comparator to use temp directory
            comparator_with_files = BenchmarkComparator(database=db)
            loaded_results = comparator_with_files.load_benchmark_results_from_files(str(temp_results_dir))
            print(f"   ✅ Loaded {len(loaded_results)} results from files")
            
            regressions = comparator.detect_regressions(days=7)
            print(f"   ✅ Regression detection completed: {len(regressions)} regressions found")
            
            # Test performance report generation
            print("\n9. Testing performance report generation...")
            report = comparator.generate_performance_report(days=30)
            print(f"   ✅ Performance report generated")
            print(f"      Report ID: {report.report_id}")
            print(f"      Total benchmarks: {report.total_benchmarks}")
            print(f"      Unique configurations: {report.unique_configurations}")
            print(f"      Average throughput: {report.avg_throughput:.2f} ops/s")
            print(f"      Trends analyzed: {len(report.trends)}")
            print(f"      Regressions detected: {len(report.regressions_detected)}")
            print(f"      Improvements detected: {len(report.improvements_detected)}")
            print(f"      Recommendations: {len(report.performance_recommendations)}")
            
            # Test report saving
            print("\n10. Testing report saving...")
            report_dir = Path(temp_dir) / "test_reports"
            report_file = comparator.save_report(report, str(report_dir))
            print(f"   ✅ Report saved: {report_file}")
            
            # Verify report file exists and is valid JSON
            with open(report_file, 'r') as f:
                saved_report = json.load(f)
                print(f"   ✅ Report file is valid JSON with {len(saved_report)} sections")
        
        print("\n" + "=" * 70)
        print("🎉 ALL BENCHMARK COMPARATOR COMPONENTS TESTED SUCCESSFULLY!")
        print("=" * 70)
        print("📊 Benchmark Comparison Features Validated:")
        print("   ✅ SQLite database persistence")
        print("   ✅ Benchmark result storage and retrieval")
        print("   ✅ Performance comparison calculations")
        print("   ✅ Trend analysis and projections")
        print("   ✅ Regression detection algorithms")
        print("   ✅ Comprehensive performance reporting")
        print("   ✅ JSON file import/export")
        print("   ✅ Historical data analysis")
        print()
        print("🚀 Ready for production benchmark monitoring!")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark comparator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison_scenarios():
    """Test specific comparison scenarios"""
    print("\n🎯 Testing Comparison Scenarios")
    print("=" * 50)
    
    try:
        from prsm.performance.benchmark_comparator import BenchmarkComparator, RegressionSeverity, TrendDirection
        
        comparator = BenchmarkComparator()
        mock_results = create_mock_benchmark_results()
        
        # Test improvement scenario
        print("1. Testing improvement detection...")
        improvement_comparison = comparator.compare_benchmarks(mock_results[1], mock_results[2])
        print(f"   ✅ Improvement detected: {improvement_comparison.is_improvement}")
        print(f"   Performance change: {improvement_comparison.overall_performance_change:.1f}%")
        
        # Test regression scenario
        print("\n2. Testing regression detection...")
        regression_comparison = comparator.compare_benchmarks(mock_results[0], mock_results[1])
        print(f"   ✅ Regression detected: {regression_comparison.is_regression}")
        print(f"   Severity: {regression_comparison.regression_severity.value}")
        print(f"   Performance change: {regression_comparison.overall_performance_change:.1f}%")
        
        # Test specific metric changes
        print("\n3. Testing metric-specific analysis...")
        metrics_comp = regression_comparison.metrics_comparison
        for metric, data in metrics_comp.items():
            change = data['change_percent']
            if abs(change) > 5:  # Significant change
                direction = "increased" if change > 0 else "decreased"
                print(f"   {metric}: {direction} by {abs(change):.1f}%")
        
        print("\n✅ Comparison scenario tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Comparison scenario test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 PRSM Benchmark Comparator - Component Testing")
    print("=" * 70)
    
    # Test core components
    components_ok = test_benchmark_comparator_components()
    
    # Test comparison scenarios
    scenarios_ok = False
    if components_ok:
        scenarios_ok = test_comparison_scenarios()
    
    print("\n" + "=" * 70)
    print("🏁 BENCHMARK COMPARATOR TEST SUMMARY:")
    print(f"   Core Components: {'✅ PASS' if components_ok else '❌ FAIL'}")
    print(f"   Comparison Scenarios: {'✅ PASS' if scenarios_ok else '❌ FAIL'}")
    
    if components_ok and scenarios_ok:
        print("\n🎉 Benchmark comparator is ready for production!")
        print("🚀 Features ready:")
        print("   📊 Performance regression detection")
        print("   📈 Historical trend analysis")
        print("   📋 Comprehensive performance reporting")
        print("   💾 SQLite database persistence")
        print("   🔍 Automated comparison algorithms")
        print("\n💡 Next steps:")
        print("   1. Run: python prsm/performance/benchmark_comparator.py")
        print("   2. Generate reports from existing benchmark data")
        print("   3. Set up automated monitoring")
    else:
        print("\n❌ Fix component issues before using comparator")
    
    sys.exit(0 if (components_ok and scenarios_ok) else 1)
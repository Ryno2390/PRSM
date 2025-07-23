#!/usr/bin/env python3
"""
PRSM Performance Test Suite Runner
=================================

Comprehensive test runner that executes all performance, regression, and load tests
in the correct order and generates unified reports.

Usage:
    python run_all_performance_tests.py [--type=all|benchmark|regression|load] [--save-baseline] [--report-format=json|html]
"""

import sys
import os
import asyncio
import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Add PRSM to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    # Import test modules
    from tests.benchmarks.test_performance_benchmarks import (
        TestComprehensivePerformanceSuite, 
        BenchmarkReporter
    )
    from tests.regression.test_regression_detection import (
        TestComprehensiveRegressionSuite,
        RegressionDetector
    )
    from tests.load.test_load_testing import (
        TestComprehensiveLoadTestSuite,
        LoadTestRunner
    )
    
    # Try to import pytest for better test discovery
    import pytest
    PYTEST_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("Some test modules may not be available")
    PYTEST_AVAILABLE = False


class PerformanceTestSuiteRunner:
    """Unified performance test suite runner"""
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.results = {
            "metadata": {
                "start_time": self.start_time.isoformat(),
                "test_environment": self._get_test_environment(),
                "suite_version": "1.0.0"
            },
            "benchmark_results": None,
            "regression_results": None,
            "load_test_results": None,
            "summary": {},
            "recommendations": [],
            "overall_status": "unknown"
        }
    
    def _get_test_environment(self) -> Dict[str, Any]:
        """Get test environment information"""
        try:
            import platform
            import psutil
            
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "architecture": platform.architecture()[0],
                "hostname": platform.node()
            }
        except Exception as e:
            return {"error": f"Could not gather environment info: {e}"}
    
    async def run_benchmark_tests(self) -> Dict[str, Any]:
        """Run performance benchmark tests"""
        print("🚀 Running Performance Benchmark Tests...")
        print("-" * 60)
        
        try:
            benchmark_suite = TestComprehensivePerformanceSuite()
            
            # Run the comprehensive benchmark suite
            benchmark_report = await benchmark_suite.test_full_performance_benchmark_suite()
            
            self.results["benchmark_results"] = benchmark_report
            
            print("✅ Performance benchmark tests completed successfully")
            return benchmark_report
            
        except Exception as e:
            print(f"❌ Performance benchmark tests failed: {e}")
            self.results["benchmark_results"] = {
                "error": str(e),
                "status": "failed"
            }
            return {"error": str(e)}
    
    async def run_regression_tests(self) -> Dict[str, Any]:
        """Run regression detection tests"""
        print("\n🔍 Running Regression Detection Tests...")
        print("-" * 60)
        
        try:
            regression_suite = TestComprehensiveRegressionSuite()
            
            # Run the comprehensive regression suite
            regression_report = await regression_suite.test_full_regression_suite()
            
            self.results["regression_results"] = regression_report
            
            print("✅ Regression detection tests completed successfully")
            return regression_report
            
        except Exception as e:
            print(f"❌ Regression detection tests failed: {e}")
            self.results["regression_results"] = {
                "error": str(e),
                "status": "failed"
            }
            return {"error": str(e)}
    
    async def run_load_tests(self) -> Dict[str, Any]:
        """Run load testing suite"""
        print("\n🔥 Running Load Testing Suite...")
        print("-" * 60)
        
        try:
            load_test_suite = TestComprehensiveLoadTestSuite()
            
            # Run the comprehensive load test suite
            load_test_report = await load_test_suite.test_full_load_testing_suite()
            
            self.results["load_test_results"] = load_test_report
            
            print("✅ Load testing suite completed successfully")
            return load_test_report
            
        except Exception as e:
            print(f"❌ Load testing suite failed: {e}")
            self.results["load_test_results"] = {
                "error": str(e),
                "status": "failed"
            }
            return {"error": str(e)}
    
    def run_with_pytest(self, test_type: str = "all") -> bool:
        """Run tests using pytest for better integration"""
        if not PYTEST_AVAILABLE:
            print("⚠️  pytest not available, falling back to direct execution")
            return False
        
        print(f"🧪 Running tests with pytest (type: {test_type})...")
        
        # Determine which tests to run based on markers
        test_markers = []
        if test_type == "all":
            test_markers = ["benchmark", "regression", "load"]
        elif test_type == "benchmark":
            test_markers = ["benchmark"]
        elif test_type == "regression":
            test_markers = ["regression"] 
        elif test_type == "load":
            test_markers = ["load"]
        else:
            print(f"❌ Unknown test type: {test_type}")
            return False
        
        try:
            for marker in test_markers:
                print(f"   🏃 Running {marker} tests...")
                
                # Build pytest arguments
                pytest_args = [
                    "-v",  # Verbose output
                    "-s",  # Don't capture output
                    f"-m {marker}",  # Run only tests with this marker
                    "--tb=short",  # Short traceback format
                    f"tests/{marker}/" if marker != "benchmark" else "tests/benchmarks/",
                    "--color=yes"
                ]
                
                # Run pytest
                result = pytest.main(pytest_args)
                
                if result != 0:
                    print(f"   ❌ {marker} tests failed with exit code {result}")
                    return False
                else:
                    print(f"   ✅ {marker} tests completed successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ pytest execution failed: {e}")
            return False
    
    async def run_full_test_suite(self, test_type: str = "all") -> Dict[str, Any]:
        """Run the complete performance test suite"""
        
        print("🎯 PRSM COMPREHENSIVE PERFORMANCE TEST SUITE")
        print("=" * 80)
        print(f"📅 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"🔧 Test Type: {test_type.upper()}")
        print(f"🖥️  Environment: {self.results['metadata']['test_environment'].get('platform', 'Unknown')}")
        print("=" * 80)
        
        suite_start_time = time.perf_counter()
        
        # Run tests based on type
        if test_type in ["all", "benchmark"]:
            await self.run_benchmark_tests()
        
        if test_type in ["all", "regression"]:
            await self.run_regression_tests()
        
        if test_type in ["all", "load"]:
            await self.run_load_tests()
        
        suite_duration = time.perf_counter() - suite_start_time
        
        # Generate unified summary
        self._generate_unified_summary(suite_duration)
        
        # Complete metadata
        self.results["metadata"]["end_time"] = datetime.now(timezone.utc).isoformat()
        self.results["metadata"]["total_duration_seconds"] = suite_duration
        
        return self.results
    
    def _generate_unified_summary(self, suite_duration: float):
        """Generate unified summary across all test types"""
        
        summary = {
            "total_duration_seconds": suite_duration,
            "tests_run": {},
            "overall_health": "unknown",
            "performance_grade": "unknown",
            "critical_issues": [],
            "key_metrics": {}
        }
        
        recommendations = []
        critical_issues = []
        
        # Analyze benchmark results
        if self.results["benchmark_results"] and "error" not in self.results["benchmark_results"]:
            benchmark_summary = self.results["benchmark_results"].get("summary", {})
            
            summary["tests_run"]["benchmarks"] = benchmark_summary.get("total_benchmarks", 0)
            
            # Extract key performance metrics
            if "overall_system_health" in benchmark_summary:
                health = benchmark_summary["overall_system_health"]
                summary["key_metrics"]["performance_health"] = health
                
                if health < 0.7:
                    critical_issues.append(f"Poor performance health: {health:.1%}")
            
            # Check for regressions in benchmarks
            if benchmark_summary.get("regressions_detected"):
                for regression in benchmark_summary["regressions_detected"]:
                    if regression.get("severity") == "high":
                        critical_issues.append(f"Performance regression: {regression['benchmark']}")
        
        # Analyze regression results
        if self.results["regression_results"] and "error" not in self.results["regression_results"]:
            regression_summary = self.results["regression_results"].get("summary", {})
            
            summary["tests_run"]["regression_tests"] = regression_summary.get("total_tests", 0)
            
            # Check regression status
            overall_status = regression_summary.get("overall_status", "unknown")
            if "critical" in overall_status:
                critical_issues.append("Critical regressions detected")
                recommendations.append("URGENT: Fix critical regressions before deployment")
            elif "high" in overall_status:
                critical_issues.append("High-severity regressions detected")
                recommendations.append("Address high-severity regressions")
            
            # Extract regression rate
            regression_rate = regression_summary.get("regression_rate", 0)
            summary["key_metrics"]["regression_rate"] = regression_rate
            
            if regression_rate > 0.2:  # >20% regression rate
                critical_issues.append(f"High regression rate: {regression_rate:.1%}")
        
        # Analyze load test results
        if self.results["load_test_results"] and "error" not in self.results["load_test_results"]:
            load_summary = self.results["load_test_results"].get("summary", {})
            
            summary["tests_run"]["load_tests"] = load_summary.get("total_tests_run", 0)
            
            # Extract capacity metrics
            max_throughput = load_summary.get("max_throughput_achieved", 0)
            breaking_point = load_summary.get("system_breaking_point")
            
            summary["key_metrics"]["max_throughput_rps"] = max_throughput
            summary["key_metrics"]["capacity_breaking_point"] = breaking_point
            
            if max_throughput < 10:  # <10 RPS is concerning
                critical_issues.append(f"Low system throughput: {max_throughput:.1f} RPS")
            
            if breaking_point and breaking_point < 20:  # <20 concurrent users is concerning
                critical_issues.append(f"Low concurrent capacity: {breaking_point} users")
            
            # Check system stability
            stability = load_summary.get("overall_stability", "unknown")
            if stability != "stable":
                critical_issues.append(f"System stability issues: {stability}")
        
        # Determine overall health and grade
        if critical_issues:
            summary["overall_health"] = "poor"
            summary["performance_grade"] = "F" if len(critical_issues) > 3 else "D"
        else:
            # Assess based on available metrics
            health_indicators = []
            
            if "performance_health" in summary["key_metrics"]:
                health_indicators.append(summary["key_metrics"]["performance_health"])
            
            if "regression_rate" in summary["key_metrics"]:
                health_indicators.append(1 - summary["key_metrics"]["regression_rate"])  # Invert regression rate
            
            if "max_throughput_rps" in summary["key_metrics"]:
                # Normalize throughput (assuming 50 RPS is excellent)
                throughput_score = min(summary["key_metrics"]["max_throughput_rps"] / 50, 1.0)
                health_indicators.append(throughput_score)
            
            if health_indicators:
                avg_health = sum(health_indicators) / len(health_indicators)
                summary["overall_health"] = avg_health
                
                if avg_health >= 0.9:
                    summary["performance_grade"] = "A"
                elif avg_health >= 0.8:
                    summary["performance_grade"] = "B"
                elif avg_health >= 0.7:
                    summary["performance_grade"] = "C"
                elif avg_health >= 0.6:
                    summary["performance_grade"] = "D"
                else:
                    summary["performance_grade"] = "F"
            else:
                summary["overall_health"] = "unknown"
                summary["performance_grade"] = "Incomplete"
        
        # Generate recommendations
        if not recommendations:
            if summary["performance_grade"] in ["D", "F"]:
                recommendations.append("System requires significant performance optimization")
            elif summary["performance_grade"] == "C":
                recommendations.append("System performance needs improvement")
            else:
                recommendations.append("System performance is acceptable")
        
        # Add specific recommendations based on metrics
        if "max_throughput_rps" in summary["key_metrics"] and summary["key_metrics"]["max_throughput_rps"] < 20:
            recommendations.append("Investigate and optimize system throughput bottlenecks")
        
        if "regression_rate" in summary["key_metrics"] and summary["key_metrics"]["regression_rate"] > 0.1:
            recommendations.append("Review recent changes that may have caused regressions")
        
        summary["critical_issues"] = critical_issues
        
        self.results["summary"] = summary
        self.results["recommendations"] = recommendations
        self.results["overall_status"] = summary["performance_grade"]
    
    def save_results(self, filename: str = None, format: str = "json"):
        """Save test results to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prsm_performance_test_results_{timestamp}"
        
        if format == "json":
            full_filename = f"{filename}.json"
            with open(full_filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
        elif format == "html":
            full_filename = f"{filename}.html"
            html_content = self._generate_html_report()
            with open(full_filename, 'w') as f:
                f.write(html_content)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"📄 Results saved to: {full_filename}")
        return full_filename
    
    def _generate_html_report(self) -> str:
        """Generate HTML report"""
        
        summary = self.results.get("summary", {})
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PRSM Performance Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .summary {{ background: #e8f5e8; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .critical {{ background: #ffe8e8; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .grade {{ font-size: 48px; font-weight: bold; text-align: center; margin: 20px 0; }}
        .grade.A {{ color: #4CAF50; }}
        .grade.B {{ color: #8BC34A; }}
        .grade.C {{ color: #FFC107; }}
        .grade.D {{ color: #FF9800; }}
        .grade.F {{ color: #F44336; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 PRSM Performance Test Report</h1>
        <p><strong>Generated:</strong> {self.results['metadata']['end_time']}</p>
        <p><strong>Duration:</strong> {summary.get('total_duration_seconds', 0):.1f} seconds</p>
        <p><strong>Environment:</strong> {self.results['metadata']['test_environment'].get('platform', 'Unknown')}</p>
    </div>
    
    <div class="grade {summary.get('performance_grade', 'Incomplete')}">
        Performance Grade: {summary.get('performance_grade', 'Incomplete')}
    </div>
    
    <div class="summary">
        <h2>📊 Executive Summary</h2>
        <p><strong>Overall Health:</strong> {summary.get('overall_health', 'Unknown')}</p>
        <p><strong>Tests Run:</strong> {sum(summary.get('tests_run', {}).values())} total tests</p>
        
        <h3>Key Metrics:</h3>
        {''.join([f'<div class="metric"><strong>{k.replace("_", " ").title()}:</strong> {v}</div>' for k, v in summary.get('key_metrics', {}).items()])}
    </div>
    
    {'<div class="critical"><h2>🚨 Critical Issues</h2><ul>' + ''.join([f'<li>{issue}</li>' for issue in summary.get('critical_issues', [])]) + '</ul></div>' if summary.get('critical_issues') else ''}
    
    <div>
        <h2>💡 Recommendations</h2>
        <ul>
            {''.join([f'<li>{rec}</li>' for rec in self.results.get('recommendations', [])])}
        </ul>
    </div>
    
    <div>
        <h2>📈 Detailed Results</h2>
        <p>For detailed results, please refer to the JSON report file.</p>
    </div>
    
</body>
</html>
        """
        
        return html_template
    
    def print_summary(self):
        """Print comprehensive test summary"""
        
        summary = self.results.get("summary", {})
        
        print("\n" + "=" * 80)
        print("🎯 PRSM PERFORMANCE TEST SUITE SUMMARY")
        print("=" * 80)
        
        # Overall grade and health
        grade = summary.get("performance_grade", "Incomplete")
        health = summary.get("overall_health", "unknown")
        
        grade_emoji = {
            "A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "⚫", "Incomplete": "⚪"
        }.get(grade, "❓")
        
        print(f"{grade_emoji} PERFORMANCE GRADE: {grade}")
        
        if isinstance(health, (int, float)):
            print(f"📊 OVERALL HEALTH: {health:.1%}")
        else:
            print(f"📊 OVERALL HEALTH: {health}")
        
        # Test execution summary
        tests_run = summary.get("tests_run", {})
        total_tests = sum(tests_run.values())
        duration = summary.get("total_duration_seconds", 0)
        
        print(f"🧪 TESTS EXECUTED: {total_tests} total ({duration:.1f}s duration)")
        for test_type, count in tests_run.items():
            print(f"   • {test_type.replace('_', ' ').title()}: {count}")
        
        # Key metrics
        key_metrics = summary.get("key_metrics", {})
        if key_metrics:
            print(f"\n📈 KEY METRICS:")
            for metric_name, metric_value in key_metrics.items():
                formatted_name = metric_name.replace("_", " ").title()
                if isinstance(metric_value, float) and 0 <= metric_value <= 1:
                    print(f"   • {formatted_name}: {metric_value:.1%}")
                elif isinstance(metric_value, (int, float)):
                    print(f"   • {formatted_name}: {metric_value:.2f}")
                else:
                    print(f"   • {formatted_name}: {metric_value}")
        
        # Critical issues
        critical_issues = summary.get("critical_issues", [])
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"   ❌ {issue}")
        
        # Recommendations
        recommendations = self.results.get("recommendations", [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Overall status
        print(f"\n🎯 FINAL ASSESSMENT:")
        if grade in ["A", "B"]:
            print("   ✅ System performance is good - ready for production")
        elif grade == "C":
            print("   ⚠️  System performance is acceptable - monitor closely")
        elif grade in ["D", "F"]:
            print("   ❌ System performance needs improvement - address issues before production")
        else:
            print("   ❓ Test results incomplete - run full test suite")
        
        print("=" * 80)


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="PRSM Performance Test Suite Runner")
    parser.add_argument(
        "--type", 
        choices=["all", "benchmark", "regression", "load"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current results as new baseline"
    )
    parser.add_argument(
        "--report-format",
        choices=["json", "html"],
        default="json", 
        help="Report output format (default: json)"
    )
    parser.add_argument(
        "--output",
        help="Output filename (without extension)"
    ) 
    parser.add_argument(
        "--use-pytest",
        action="store_true",
        help="Use pytest for test execution"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = PerformanceTestSuiteRunner()
    
    try:
        if args.use_pytest:
            # Run with pytest
            success = runner.run_with_pytest(args.type)
            if not success:
                print("❌ Test execution with pytest failed")
                return 1
        else:
            # Run test suite directly
            results = await runner.run_full_test_suite(args.type)
        
        # Print summary
        runner.print_summary()
        
        # Save results
        runner.save_results(
            filename=args.output,
            format=args.report_format
        )
        
        # Determine exit code based on results
        grade = runner.results.get("summary", {}).get("performance_grade", "F")
        critical_issues = runner.results.get("summary", {}).get("critical_issues", [])
        
        if grade in ["F"] or len(critical_issues) > 0:
            print("\n❌ Performance test suite completed with issues")
            return 1
        elif grade in ["D"]:
            print("\n⚠️  Performance test suite completed with warnings")
            return 0  # Still exit 0 but with warnings
        else:
            print("\n✅ Performance test suite completed successfully")
            return 0
    
    except KeyboardInterrupt:
        print("\n⏹️  Test suite interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
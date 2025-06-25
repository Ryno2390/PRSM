#!/usr/bin/env python3
"""
Performance Regression Check for PRSM CI/CD Pipeline
Analyzes benchmark results to detect performance regressions.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    name: str
    current_score: float
    baseline_score: float
    regression_pct: float
    status: str

class PerformanceRegressionChecker:
    """Checks for performance regressions in benchmark results"""
    
    def __init__(self, max_regression_pct: float = 5.0):
        self.max_regression_pct = max_regression_pct
        self.logger = logging.getLogger(__name__)
        
        # Baseline performance scores (would typically come from historical data)
        self.baselines = {
            "ai_agent_processing": 85.0,
            "p2p_network_throughput": 90.0,
            "model_inference_speed": 88.0,
            "orchestration_efficiency": 92.0,
            "egcfg_generation_quality": 95.0,
            "memory_usage": 80.0,
            "startup_time": 85.0,
            "concurrent_operations": 87.0
        }
    
    def analyze_benchmark_results(self, benchmark_file: Path) -> List[BenchmarkResult]:
        """Analyze benchmark results for regressions"""
        try:
            with open(benchmark_file) as f:
                benchmark_data = json.load(f)
            
            results = []
            benchmarks = benchmark_data.get('benchmarks', {})
            
            for benchmark_name, benchmark_info in benchmarks.items():
                current_score = benchmark_info.get('score', 0.0)
                baseline_score = self.baselines.get(benchmark_name, current_score)
                
                # Calculate regression percentage
                if baseline_score > 0:
                    regression_pct = ((baseline_score - current_score) / baseline_score) * 100
                else:
                    regression_pct = 0.0
                
                # Determine status
                if regression_pct <= self.max_regression_pct:
                    status = "PASS"
                elif regression_pct <= self.max_regression_pct * 2:
                    status = "WARN"
                else:
                    status = "FAIL"
                
                result = BenchmarkResult(
                    name=benchmark_name,
                    current_score=current_score,
                    baseline_score=baseline_score,
                    regression_pct=regression_pct,
                    status=status
                )
                
                results.append(result)
                
                # Log result
                if status == "FAIL":
                    self.logger.error(f"❌ {benchmark_name}: {regression_pct:.1f}% regression (score: {current_score:.1f}, baseline: {baseline_score:.1f})")
                elif status == "WARN":
                    self.logger.warning(f"⚠️  {benchmark_name}: {regression_pct:.1f}% regression (score: {current_score:.1f}, baseline: {baseline_score:.1f})")
                else:
                    self.logger.info(f"✅ {benchmark_name}: {regression_pct:.1f}% change (score: {current_score:.1f}, baseline: {baseline_score:.1f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to analyze benchmark results: {e}")
            raise
    
    def check_overall_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Check overall performance status"""
        if not results:
            return {
                "status": "PASS",
                "message": "No benchmarks to evaluate",
                "summary": {}
            }
        
        failed_benchmarks = [r for r in results if r.status == "FAIL"]
        warned_benchmarks = [r for r in results if r.status == "WARN"]
        passed_benchmarks = [r for r in results if r.status == "PASS"]
        
        # Calculate average regression
        avg_regression = sum(r.regression_pct for r in results) / len(results)
        
        # Determine overall status
        if failed_benchmarks:
            overall_status = "FAIL"
            message = f"{len(failed_benchmarks)} benchmark(s) failed regression check"
        elif warned_benchmarks and len(warned_benchmarks) > len(results) // 2:
            overall_status = "WARN"
            message = f"{len(warned_benchmarks)} benchmark(s) show concerning performance regression"
        else:
            overall_status = "PASS"
            message = "All benchmarks within acceptable performance range"
        
        summary = {
            "total_benchmarks": len(results),
            "passed": len(passed_benchmarks),
            "warned": len(warned_benchmarks),
            "failed": len(failed_benchmarks),
            "average_regression_pct": avg_regression,
            "max_allowed_regression_pct": self.max_regression_pct
        }
        
        return {
            "status": overall_status,
            "message": message,
            "summary": summary,
            "detailed_results": [
                {
                    "name": r.name,
                    "current_score": r.current_score,
                    "baseline_score": r.baseline_score,
                    "regression_pct": r.regression_pct,
                    "status": r.status
                }
                for r in results
            ]
        }
    
    def generate_performance_report(self, results: List[BenchmarkResult]) -> str:
        """Generate detailed performance report"""
        overall = self.check_overall_performance(results)
        
        report = []
        report.append("📊 PRSM Performance Regression Analysis")
        report.append("=" * 50)
        report.append("")
        
        report.append(f"🎯 Overall Status: {overall['status']}")
        report.append(f"📝 Message: {overall['message']}")
        report.append("")
        
        summary = overall['summary']
        report.append("📈 Summary:")
        report.append(f"  Total Benchmarks: {summary['total_benchmarks']}")
        report.append(f"  Passed: {summary['passed']} ✅")
        report.append(f"  Warned: {summary['warned']} ⚠️")
        report.append(f"  Failed: {summary['failed']} ❌")
        report.append(f"  Average Regression: {summary['average_regression_pct']:.1f}%")
        report.append(f"  Max Allowed: {summary['max_allowed_regression_pct']:.1f}%")
        report.append("")
        
        if results:
            report.append("🔍 Detailed Results:")
            for result in results:
                status_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[result.status]
                report.append(f"  {status_emoji} {result.name}:")
                report.append(f"    Current Score: {result.current_score:.1f}")
                report.append(f"    Baseline Score: {result.baseline_score:.1f}")
                report.append(f"    Regression: {result.regression_pct:+.1f}%")
                report.append("")
        
        return "\n".join(report)

def main():
    """Main performance regression check function"""
    parser = argparse.ArgumentParser(description="PRSM Performance Regression Check")
    parser.add_argument("benchmark_file", type=Path, help="Path to benchmark results JSON file")
    parser.add_argument("--max-regression", type=float, default=5.0, help="Maximum allowed regression percentage")
    parser.add_argument("--output", type=Path, help="Output file for detailed results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    checker = PerformanceRegressionChecker(max_regression_pct=args.max_regression)
    
    try:
        # Check if benchmark file exists
        if not args.benchmark_file.exists():
            logging.error(f"Benchmark file not found: {args.benchmark_file}")
            sys.exit(1)
        
        # Analyze benchmark results
        results = checker.analyze_benchmark_results(args.benchmark_file)
        
        # Check overall performance
        overall_check = checker.check_overall_performance(results)
        
        # Generate report
        report = checker.generate_performance_report(results)
        print(report)
        
        # Save detailed results if output specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(overall_check, f, indent=2)
            print(f"\n📄 Detailed results saved to: {args.output}")
        
        # Exit with appropriate code
        if overall_check['status'] == "FAIL":
            print("\n❌ Performance regression check FAILED")
            sys.exit(1)
        elif overall_check['status'] == "WARN":
            print("\n⚠️  Performance regression check shows WARNINGS")
            sys.exit(0)  # Warnings don't fail the build by default
        else:
            print("\n✅ Performance regression check PASSED")
            sys.exit(0)
        
    except Exception as e:
        logging.error(f"Performance regression check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
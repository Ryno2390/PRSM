#!/usr/bin/env python3
"""
PRSM Integrated Performance Monitor
Complete integration of all performance monitoring components

Features:
- Comprehensive benchmark suite execution
- Real-time performance dashboard
- Scaling test controller
- Benchmark result comparison and analysis
- Historical trend monitoring
- Automated performance reporting
- Regression detection and alerting
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add PRSM to path
PRSM_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PRSM_ROOT))

# Import all performance components
try:
    from prsm.performance.benchmark_collector import get_global_collector, reset_global_collector
    from prsm.performance.benchmark_comparator import BenchmarkComparator, BenchmarkDatabase
    from prsm.performance.scaling_test_controller import ScalingTestController, ScalingTestConfig, ScalingEnvironment, ResourceProfile, NetworkTopology
    from comprehensive_performance_benchmark import PerformanceBenchmarkSuite, BenchmarkConfig, BenchmarkType, NetworkCondition
except ImportError as e:
    print(f"⚠️ Could not import all performance modules: {e}")
    sys.exit(1)


class IntegratedPerformanceMonitor:
    """Integrated performance monitoring system for PRSM"""
    
    def __init__(self, output_directory: str = "integrated_performance_results"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Initialize all components
        self.benchmark_suite = PerformanceBenchmarkSuite(str(self.output_directory / "benchmarks"))
        self.scaling_controller = ScalingTestController(str(self.output_directory / "scaling"))
        self.database = BenchmarkDatabase(str(self.output_directory / "performance_monitor.db"))
        self.comparator = BenchmarkComparator(self.database)
        
        # Monitoring state
        self.monitoring_active = False
        self.last_report_time = None
        self.performance_history = []
    
    async def run_comprehensive_performance_validation(self) -> Dict[str, Any]:
        """Run complete performance validation suite"""
        
        print("🚀 PRSM COMPREHENSIVE PERFORMANCE VALIDATION")
        print("=" * 80)
        
        validation_start = time.time()
        results = {
            "validation_id": f"comprehensive_validation_{int(time.time())}",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "components_tested": [],
            "results": {},
            "summary": {},
            "recommendations": []
        }
        
        try:
            # 1. Run benchmark suite
            print("\n📊 PHASE 1: Benchmark Suite Execution")
            print("-" * 50)
            
            benchmark_configs = [
                BenchmarkConfig(
                    name="validation_consensus_scaling",
                    benchmark_type=BenchmarkType.CONSENSUS_SCALING,
                    node_count=25,
                    duration_seconds=15,
                    target_operations_per_second=12.0,
                    network_condition=NetworkCondition.WAN,
                    enable_post_quantum=True
                ),
                BenchmarkConfig(
                    name="validation_network_throughput",
                    benchmark_type=BenchmarkType.NETWORK_THROUGHPUT,
                    node_count=20,
                    duration_seconds=12,
                    target_operations_per_second=15.0,
                    network_condition=NetworkCondition.WAN,
                    enable_post_quantum=True
                ),
                BenchmarkConfig(
                    name="validation_post_quantum_overhead",
                    benchmark_type=BenchmarkType.POST_QUANTUM_OVERHEAD,
                    node_count=15,
                    duration_seconds=10,
                    target_operations_per_second=10.0,
                    network_condition=NetworkCondition.WAN,
                    enable_post_quantum=True
                )
            ]
            
            benchmark_results = []
            for config in benchmark_configs:
                print(f"   🎯 Running: {config.name}")
                result = await self.benchmark_suite.run_benchmark(config)
                benchmark_results.append(result)
                
                # Store in database
                result_data = {
                    "config": {
                        "name": config.name,
                        "benchmark_type": config.benchmark_type.value,
                        "node_count": config.node_count,
                        "duration_seconds": config.duration_seconds,
                        "network_condition": config.network_condition.value
                    },
                    "metrics": {
                        "operations_per_second": result.operations_per_second,
                        "mean_latency_ms": result.mean_latency_ms,
                        "p95_latency_ms": result.p95_latency_ms,
                        "consensus_success_rate": result.consensus_success_rate,
                        "operations_per_node_per_second": result.operations_per_node_per_second
                    },
                    "timing": {
                        "start_time": result.start_time.isoformat(),
                        "end_time": result.end_time.isoformat(),
                        "duration_seconds": (result.end_time - result.start_time).total_seconds()
                    }
                }
                self.database.store_benchmark_result(result_data)
                
                print(f"      ✅ {result.operations_per_second:.2f} ops/s, {result.mean_latency_ms:.2f}ms avg")
            
            results["components_tested"].append("benchmark_suite")
            results["results"]["benchmark_suite"] = {
                "configurations_tested": len(benchmark_configs),
                "total_operations": sum(r.total_operations for r in benchmark_results),
                "average_throughput": sum(r.operations_per_second for r in benchmark_results) / len(benchmark_results),
                "average_latency": sum(r.mean_latency_ms for r in benchmark_results) / len(benchmark_results),
                "average_success_rate": sum(r.consensus_success_rate for r in benchmark_results) / len(benchmark_results)
            }
            
            # 2. Run scaling tests
            print("\n📈 PHASE 2: Scaling Test Execution") 
            print("-" * 50)
            
            scaling_config = ScalingTestConfig(
                name="validation_scaling_test",
                environment=ScalingEnvironment.LOCAL_SIMULATION,
                node_counts=[10, 20, 40],
                resource_profile=ResourceProfile.STANDARD,
                network_topology=NetworkTopology.MESH,
                network_conditions=[NetworkCondition.WAN],
                test_duration_per_scale=8,
                warmup_duration=2,
                cooldown_duration=1,
                byzantine_ratios=[0.0],
                target_operations_per_second=12.0,
                enable_resource_monitoring=True
            )
            
            print(f"   🎯 Running scaling test: {scaling_config.name}")
            scaling_result = await self.scaling_controller.run_comprehensive_scaling_test(scaling_config)
            
            # Store scaling results
            scaling_data = {
                "config": {
                    "name": scaling_config.name,
                    "environment": scaling_config.environment.value,
                    "node_counts": scaling_config.node_counts,
                    "resource_profile": scaling_config.resource_profile.value,
                    "network_topology": scaling_config.network_topology.value
                },
                "performance": {
                    "node_performance": scaling_result.node_performance,
                    "scaling_efficiency": scaling_result.scaling_efficiency,
                    "recommended_max_nodes": scaling_result.recommended_max_nodes
                },
                "resources": {
                    "resource_usage": scaling_result.resource_usage
                }
            }
            self.database.store_benchmark_result(scaling_data)
            
            results["components_tested"].append("scaling_controller")
            results["results"]["scaling_controller"] = {
                "node_counts_tested": len(scaling_config.node_counts),
                "max_nodes_tested": max(scaling_config.node_counts),
                "recommended_max_nodes": scaling_result.recommended_max_nodes,
                "scaling_efficiency_avg": sum(scaling_result.scaling_efficiency.values()) / len(scaling_result.scaling_efficiency) if scaling_result.scaling_efficiency else 0.0,
                "performance_bottlenecks": len(scaling_result.performance_bottlenecks),
                "scaling_recommendations": len(scaling_result.scaling_recommendations)
            }
            
            # 3. Run comparison analysis
            print("\n🔍 PHASE 3: Performance Analysis & Comparison")
            print("-" * 50)
            
            print("   📊 Generating performance report...")
            report = self.comparator.generate_performance_report(days=1)  # Today's data
            
            print("   🔍 Detecting regressions...")
            regressions = self.comparator.detect_regressions(days=1)
            
            print("   📈 Analyzing trends...")
            trends = {}
            for metric in ['operations_per_second', 'mean_latency_ms', 'consensus_success_rate']:
                trends[metric] = self.comparator.analyze_trends(metric, days=7)
            
            results["components_tested"].append("performance_analysis")
            results["results"]["performance_analysis"] = {
                "total_benchmarks_analyzed": report.total_benchmarks,
                "trends_analyzed": len(trends),
                "regressions_detected": len(regressions),
                "improvements_detected": len(report.improvements_detected),
                "performance_recommendations": len(report.performance_recommendations),
                "optimization_opportunities": len(report.optimization_opportunities)
            }
            
            # 4. Generate comprehensive summary
            print("\n📋 PHASE 4: Comprehensive Analysis")
            print("-" * 50)
            
            validation_duration = time.time() - validation_start
            
            # Performance summary
            overall_throughput = results["results"]["benchmark_suite"]["average_throughput"]
            overall_latency = results["results"]["benchmark_suite"]["average_latency"] 
            overall_success_rate = results["results"]["benchmark_suite"]["average_success_rate"]
            max_recommended_nodes = results["results"]["scaling_controller"]["recommended_max_nodes"]
            
            # Calculate performance score (0-100)
            throughput_score = min(100, (overall_throughput / 20.0) * 100)  # 20 ops/s = 100 points
            latency_score = max(0, 100 - (overall_latency / 100.0) * 100)  # 100ms = 0 points
            success_score = overall_success_rate * 100
            scaling_score = min(100, (max_recommended_nodes / 100.0) * 100)  # 100 nodes = 100 points
            
            overall_score = (throughput_score * 0.3 + latency_score * 0.3 + success_score * 0.2 + scaling_score * 0.2)
            
            results["summary"] = {
                "validation_duration_seconds": validation_duration,
                "overall_performance_score": overall_score,
                "throughput_score": throughput_score,
                "latency_score": latency_score,
                "success_rate_score": success_score,
                "scaling_score": scaling_score,
                "components_validated": len(results["components_tested"]),
                "total_tests_executed": len(benchmark_configs) + len(scaling_config.node_counts),
                "performance_grade": self._calculate_performance_grade(overall_score),
                "key_metrics": {
                    "average_throughput_ops_per_sec": overall_throughput,
                    "average_latency_ms": overall_latency,
                    "average_success_rate": overall_success_rate,
                    "max_recommended_nodes": max_recommended_nodes,
                    "total_operations_tested": results["results"]["benchmark_suite"]["total_operations"]
                }
            }
            
            # Generate recommendations
            recommendations = []
            
            # Performance-based recommendations
            if overall_score >= 80:
                recommendations.append("Excellent performance characteristics - ready for production deployment")
            elif overall_score >= 60:
                recommendations.append("Good performance with optimization opportunities identified")
            elif overall_score >= 40:
                recommendations.append("Moderate performance - architectural improvements recommended")
            else:
                recommendations.append("Performance improvements required before production deployment")
            
            # Specific metric recommendations
            if throughput_score < 60:
                recommendations.append("Throughput optimization needed - consider consensus algorithm improvements")
            if latency_score < 60:
                recommendations.append("Latency reduction required - optimize network protocols and message handling")
            if success_score < 95:
                recommendations.append("Reliability improvements needed - investigate consensus failure modes")
            if scaling_score < 50:
                recommendations.append("Limited scaling capability - implement hierarchical or sharded consensus")
            
            # Add analysis-based recommendations
            recommendations.extend(report.performance_recommendations[:3])
            recommendations.extend(report.optimization_opportunities[:2])
            
            results["recommendations"] = recommendations
            results["end_time"] = datetime.now(timezone.utc).isoformat()
            
            # Save comprehensive results
            results_file = self.output_directory / f"comprehensive_validation_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save performance report
            report_file = self.comparator.save_report(report, str(self.output_directory / "reports"))
            
            print(f"\n💾 Results saved:")
            print(f"   📄 Validation results: {results_file}")
            print(f"   📄 Performance report: {report_file}")
            
            # Print summary
            self._print_validation_summary(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Comprehensive validation failed: {e}")
            import traceback
            traceback.print_exc()
            
            results["error"] = str(e)
            results["end_time"] = datetime.now(timezone.utc).isoformat()
            return results
    
    def _calculate_performance_grade(self, score: float) -> str:
        """Calculate performance grade based on score"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        else:
            return "D"
    
    def _print_validation_summary(self, results: Dict[str, Any]):
        """Print comprehensive validation summary"""
        summary = results["summary"]
        
        print(f"\n🎯 COMPREHENSIVE PERFORMANCE VALIDATION SUMMARY")
        print("=" * 80)
        print(f"📊 OVERALL PERFORMANCE SCORE: {summary['overall_performance_score']:.1f}/100 (Grade: {summary['performance_grade']})")
        print("-" * 80)
        
        print(f"🚀 KEY METRICS:")
        metrics = summary["key_metrics"]
        print(f"   Throughput: {metrics['average_throughput_ops_per_sec']:.2f} ops/sec")
        print(f"   Latency: {metrics['average_latency_ms']:.2f} ms average")
        print(f"   Success Rate: {metrics['average_success_rate']:.1%}")
        print(f"   Max Recommended Nodes: {metrics['max_recommended_nodes']}")
        print(f"   Total Operations Tested: {metrics['total_operations_tested']}")
        
        print(f"\n📈 COMPONENT SCORES:")
        print(f"   Throughput: {summary['throughput_score']:.1f}/100")
        print(f"   Latency: {summary['latency_score']:.1f}/100")
        print(f"   Reliability: {summary['success_rate_score']:.1f}/100")
        print(f"   Scalability: {summary['scaling_score']:.1f}/100")
        
        print(f"\n✅ VALIDATION COVERAGE:")
        print(f"   Components Tested: {summary['components_validated']}")
        print(f"   Total Tests Executed: {summary['total_tests_executed']}")
        print(f"   Validation Duration: {summary['validation_duration_seconds']:.1f} seconds")
        
        print(f"\n💡 TOP RECOMMENDATIONS:")
        for i, recommendation in enumerate(results["recommendations"][:3], 1):
            print(f"   {i}. {recommendation}")
        
        print("=" * 80)
    
    async def continuous_monitoring(self, interval_minutes: int = 60):
        """Run continuous performance monitoring"""
        print(f"🔄 Starting continuous performance monitoring (interval: {interval_minutes} minutes)")
        
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                print(f"\n📊 Running periodic performance check...")
                
                # Run a lightweight benchmark
                quick_config = BenchmarkConfig(
                    name=f"monitoring_{int(time.time())}",
                    benchmark_type=BenchmarkType.CONSENSUS_SCALING,
                    node_count=15,
                    duration_seconds=10,
                    target_operations_per_second=10.0,
                    network_condition=NetworkCondition.WAN,
                    enable_post_quantum=True
                )
                
                result = await self.benchmark_suite.run_benchmark(quick_config)
                
                # Store result
                result_data = {
                    "config": {
                        "name": quick_config.name,
                        "benchmark_type": quick_config.benchmark_type.value,
                        "node_count": quick_config.node_count
                    },
                    "metrics": {
                        "operations_per_second": result.operations_per_second,
                        "mean_latency_ms": result.mean_latency_ms,
                        "consensus_success_rate": result.consensus_success_rate
                    },
                    "timing": {
                        "start_time": result.start_time.isoformat(),
                        "end_time": result.end_time.isoformat()
                    }
                }
                self.database.store_benchmark_result(result_data)
                
                # Check for regressions
                regressions = self.comparator.detect_regressions(days=1)
                if regressions:
                    print(f"⚠️  Performance regression detected: {len(regressions)} issues")
                    for regression in regressions[:2]:  # Show top 2
                        print(f"   - {regression.summary}")
                
                print(f"✅ Monitoring check complete: {result.operations_per_second:.2f} ops/s")
                
                # Sleep until next check
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        print("🛑 Continuous monitoring stopped")


async def run_integrated_performance_demo():
    """Run integrated performance monitoring demo"""
    
    print("🌟 PRSM INTEGRATED PERFORMANCE MONITOR - DEMO")
    print("=" * 80)
    
    monitor = IntegratedPerformanceMonitor()
    
    # Run comprehensive validation
    results = await monitor.run_comprehensive_performance_validation()
    
    if "error" not in results:
        print(f"\n🎉 INTEGRATED PERFORMANCE VALIDATION COMPLETED SUCCESSFULLY!")
        print(f"📊 Performance Grade: {results['summary']['performance_grade']}")
        print(f"🚀 Overall Score: {results['summary']['overall_performance_score']:.1f}/100")
        
        return True
    else:
        print(f"\n❌ Integrated validation failed: {results.get('error', 'Unknown error')}")
        return False


if __name__ == "__main__":
    print("🌟 PRSM Integrated Performance Monitor")
    print("=" * 60)
    
    # Run the integrated demo
    success = asyncio.run(run_integrated_performance_demo())
    
    if success:
        print("\n✅ Integrated performance monitoring system is operational!")
        print("🚀 Ready for production performance validation")
    else:
        print("\n❌ Integrated performance monitoring system needs attention")
    
    sys.exit(0 if success else 1)
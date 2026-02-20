"""
PRSM Benchmark Orchestrator Module

Orchestrates benchmark scenarios, manages test execution, and generates
comprehensive performance reports for the PRSM system.
"""

import asyncio
import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import structlog

from .benchmark_collector import BenchmarkCollector, get_global_collector

logger = structlog.get_logger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CONSENSUS_SCALING = "consensus_scaling"
    NETWORK_LATENCY = "network_latency"
    STORAGE_PERFORMANCE = "storage_performance"
    COMPUTE_PERFORMANCE = "compute_performance"
    MIXED_WORKLOAD = "mixed_workload"


class LoadProfile(Enum):
    """Load profile types"""
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    STRESS = "stress"
    SPIKE = "spike"


@dataclass
class BenchmarkScenario:
    """Configuration for a benchmark scenario"""
    name: str
    benchmark_type: BenchmarkType
    load_profile: LoadProfile
    node_count: int = 10
    duration_seconds: int = 60
    target_tps: float = 10.0
    network_latency_ms: float = 50.0
    message_size_bytes: int = 1024
    warmup_seconds: int = 5
    ramp_up_seconds: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result from a benchmark execution"""
    scenario_name: str
    benchmark_type: BenchmarkType
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_operations: int
    successful_operations: int
    failed_operations: int
    operations_per_second: float
    mean_operation_time_ms: float
    p50_operation_time_ms: float
    p95_operation_time_ms: float
    p99_operation_time_ms: float
    min_operation_time_ms: float
    max_operation_time_ms: float
    message_success_rate: float
    error_messages: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class BenchmarkOrchestrator:
    """
    Orchestrates benchmark execution and reporting
    
    Manages benchmark scenarios, executes tests, and generates reports.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collector = get_global_collector()
        self._results: List[BenchmarkResult] = []
        self._scenarios: List[BenchmarkScenario] = []
    
    def create_predefined_scenarios(self) -> List[BenchmarkScenario]:
        """Create predefined benchmark scenarios"""
        scenarios = [
            BenchmarkScenario(
                name="light_consensus_10_nodes",
                benchmark_type=BenchmarkType.CONSENSUS_SCALING,
                load_profile=LoadProfile.LIGHT,
                node_count=10,
                duration_seconds=60,
                target_tps=5.0,
            ),
            BenchmarkScenario(
                name="moderate_consensus_20_nodes",
                benchmark_type=BenchmarkType.CONSENSUS_SCALING,
                load_profile=LoadProfile.MODERATE,
                node_count=20,
                duration_seconds=120,
                target_tps=20.0,
            ),
            BenchmarkScenario(
                name="heavy_throughput_50_nodes",
                benchmark_type=BenchmarkType.THROUGHPUT,
                load_profile=LoadProfile.HEAVY,
                node_count=50,
                duration_seconds=180,
                target_tps=100.0,
            ),
            BenchmarkScenario(
                name="stress_latency_test",
                benchmark_type=BenchmarkType.LATENCY,
                load_profile=LoadProfile.STRESS,
                node_count=30,
                duration_seconds=60,
                target_tps=50.0,
            ),
            BenchmarkScenario(
                name="network_latency_simulation",
                benchmark_type=BenchmarkType.NETWORK_LATENCY,
                load_profile=LoadProfile.MODERATE,
                node_count=15,
                duration_seconds=90,
                network_latency_ms=100.0,
            ),
            BenchmarkScenario(
                name="storage_performance_test",
                benchmark_type=BenchmarkType.STORAGE_PERFORMANCE,
                load_profile=LoadProfile.MODERATE,
                node_count=10,
                duration_seconds=60,
                message_size_bytes=4096,
            ),
            BenchmarkScenario(
                name="mixed_workload_realistic",
                benchmark_type=BenchmarkType.MIXED_WORKLOAD,
                load_profile=LoadProfile.MODERATE,
                node_count=25,
                duration_seconds=300,
                target_tps=30.0,
            ),
        ]
        
        self._scenarios = scenarios
        return scenarios
    
    async def run_benchmark_scenario(self, scenario: BenchmarkScenario) -> BenchmarkResult:
        """Execute a single benchmark scenario"""
        logger.info("Starting benchmark scenario", scenario=scenario.name)
        
        self.collector.reset()
        start_time = datetime.now(timezone.utc)
        operation_times: List[float] = []
        errors: List[str] = []
        successful = 0
        failed = 0
        
        warmup_end = time.time() + scenario.warmup_seconds
        while time.time() < warmup_end:
            await asyncio.sleep(0.1)
        
        end_time = time.time() + scenario.duration_seconds
        operations_interval = 1.0 / scenario.target_tps if scenario.target_tps > 0 else 0.1
        
        while time.time() < end_time:
            op_start = time.perf_counter()
            
            try:
                await self._simulate_operation(scenario)
                op_time = (time.perf_counter() - op_start) * 1000
                operation_times.append(op_time)
                successful += 1
                self.collector.record(f"{scenario.name}_operation", op_time)
            except Exception as e:
                failed += 1
                errors.append(str(e))
            
            await asyncio.sleep(operations_interval)
        
        finish_time = datetime.now(timezone.utc)
        duration = (finish_time - start_time).total_seconds()
        total_ops = successful + failed
        
        if operation_times:
            sorted_times = sorted(operation_times)
            n = len(sorted_times)
            
            def percentile(p: float) -> float:
                if n == 1:
                    return sorted_times[0]
                k = (n - 1) * p / 100
                f = int(k)
                c = min(f + 1, n - 1)
                return sorted_times[f] + (k - f) * (sorted_times[c] - sorted_times[f])
            
            mean_time = sum(operation_times) / n
            p50 = percentile(50)
            p95 = percentile(95)
            p99 = percentile(99)
            min_time = sorted_times[0]
            max_time = sorted_times[-1]
        else:
            mean_time = p50 = p95 = p99 = min_time = max_time = 0.0
        
        result = BenchmarkResult(
            scenario_name=scenario.name,
            benchmark_type=scenario.benchmark_type,
            start_time=start_time,
            end_time=finish_time,
            duration_seconds=duration,
            total_operations=total_ops,
            successful_operations=successful,
            failed_operations=failed,
            operations_per_second=total_ops / duration if duration > 0 else 0,
            mean_operation_time_ms=mean_time,
            p50_operation_time_ms=p50,
            p95_operation_time_ms=p95,
            p99_operation_time_ms=p99,
            min_operation_time_ms=min_time,
            max_operation_time_ms=max_time,
            message_success_rate=successful / total_ops if total_ops > 0 else 0,
            error_messages=errors[:10],
            metrics=self.collector.get_summary_stats(),
        )
        
        self._results.append(result)
        logger.info(
            "Benchmark scenario completed",
            scenario=scenario.name,
            ops_per_second=result.operations_per_second,
            success_rate=result.message_success_rate,
        )
        
        return result
    
    async def _simulate_operation(self, scenario: BenchmarkScenario) -> None:
        """Simulate an operation for benchmarking"""
        base_latency = scenario.network_latency_ms / 1000
        variation = base_latency * 0.2
        actual_latency = max(0.001, base_latency + (hash(time.time()) % 1000 / 1000) * variation)
        await asyncio.sleep(actual_latency)
    
    async def run_all_scenarios(self) -> List[BenchmarkResult]:
        """Run all predefined scenarios"""
        if not self._scenarios:
            self.create_predefined_scenarios()
        
        results = []
        for scenario in self._scenarios:
            result = await self.run_benchmark_scenario(scenario)
            results.append(result)
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results"""
        if not self._results:
            return {"total_scenarios_run": 0}
        
        total_ops = sum(r.total_operations for r in self._results)
        avg_ops_per_second = sum(r.operations_per_second for r in self._results) / len(self._results)
        avg_success_rate = sum(r.message_success_rate for r in self._results) / len(self._results)
        
        return {
            "total_scenarios_run": len(self._results),
            "total_operations": total_ops,
            "average_ops_per_second": avg_ops_per_second,
            "average_success_rate": avg_success_rate,
            "scenarios": [
                {
                    "name": r.scenario_name,
                    "type": r.benchmark_type.value,
                    "ops_per_second": r.operations_per_second,
                    "success_rate": r.message_success_rate,
                    "mean_latency_ms": r.mean_operation_time_ms,
                }
                for r in self._results
            ],
        }
    
    def save_results_to_csv(self, filename: str = "benchmark_results.csv") -> str:
        """Save results to CSV file"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'scenario_name', 'benchmark_type', 'start_time', 'end_time',
                'duration_seconds', 'total_operations', 'successful_operations',
                'failed_operations', 'operations_per_second', 'mean_operation_time_ms',
                'p50_operation_time_ms', 'p95_operation_time_ms', 'p99_operation_time_ms',
                'message_success_rate'
            ])
            
            for r in self._results:
                writer.writerow([
                    r.scenario_name, r.benchmark_type.value, r.start_time.isoformat(),
                    r.end_time.isoformat(), r.duration_seconds, r.total_operations,
                    r.successful_operations, r.failed_operations, r.operations_per_second,
                    r.mean_operation_time_ms, r.p50_operation_time_ms,
                    r.p95_operation_time_ms, r.p99_operation_time_ms, r.message_success_rate
                ])
        
        logger.info("Saved benchmark results to CSV", path=str(filepath))
        return str(filepath)
    
    def save_results_to_json(self, filename: str = "benchmark_results.json") -> str:
        """Save results to JSON file"""
        filepath = self.output_dir / filename
        
        data = {
            "benchmark_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_scenarios": len(self._results),
                "benchmark_type": "prsm_performance",
            },
            "summary_statistics": self.get_performance_summary(),
            "detailed_results": [
                {
                    "scenario_name": r.scenario_name,
                    "benchmark_type": r.benchmark_type.value,
                    "start_time": r.start_time.isoformat(),
                    "end_time": r.end_time.isoformat(),
                    "duration_seconds": r.duration_seconds,
                    "total_operations": r.total_operations,
                    "successful_operations": r.successful_operations,
                    "failed_operations": r.failed_operations,
                    "operations_per_second": r.operations_per_second,
                    "mean_operation_time_ms": r.mean_operation_time_ms,
                    "p50_operation_time_ms": r.p50_operation_time_ms,
                    "p95_operation_time_ms": r.p95_operation_time_ms,
                    "p99_operation_time_ms": r.p99_operation_time_ms,
                    "min_operation_time_ms": r.min_operation_time_ms,
                    "max_operation_time_ms": r.max_operation_time_ms,
                    "message_success_rate": r.message_success_rate,
                    "errors": r.error_messages[:5],
                }
                for r in self._results
            ],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info("Saved benchmark results to JSON", path=str(filepath))
        return str(filepath)
    
    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results"""
        return self._results
    
    def clear_results(self) -> None:
        """Clear all stored results"""
        self._results.clear()

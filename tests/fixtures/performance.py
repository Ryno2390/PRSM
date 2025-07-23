"""
Performance Testing Fixtures
=============================

Comprehensive fixtures for performance testing, benchmarking,
and load testing of PRSM components.
"""

import pytest
import asyncio
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from unittest.mock import Mock, patch
from dataclasses import dataclass, field
from datetime import datetime, timezone
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import resource
import gc

try:
    import pytest_benchmark
    from memory_profiler import profile as memory_profile
    from prsm.core.performance import get_performance_monitor, get_profiler
except ImportError:
    # Create mock implementations if dependencies not available
    pytest_benchmark = None
    memory_profile = None
    get_performance_monitor = lambda: Mock()
    get_profiler = lambda: Mock()


@dataclass
class PerformanceMetrics:
    """Performance test metrics"""
    execution_time_ms: float
    memory_usage_mb: float
    cpu_percent: float
    peak_memory_mb: float
    throughput_ops_per_sec: float
    error_rate: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LoadTestResults:
    """Load test results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    max_response_time: float
    min_response_time: float
    requests_per_second: float
    error_rate: float
    response_time_percentiles: Dict[int, float]
    memory_usage: Dict[str, float]
    cpu_usage: Dict[str, float]


class PerformanceTestRunner:
    """Performance test execution runner"""
    
    def __init__(self):
        self.monitor = get_performance_monitor()
        self.profiler = get_profiler()
        self.results = []
        
    def run_performance_test(
        self,
        test_function: Callable,
        iterations: int = 100,
        warmup_iterations: int = 10,
        concurrent_users: int = 1,
        track_memory: bool = True
    ) -> PerformanceMetrics:
        """Run performance test with comprehensive metrics"""
        
        # Warmup runs
        for _ in range(warmup_iterations):
            try:
                test_function()
            except:
                pass
        
        # Collect garbage before actual test
        gc.collect()
        
        # Record initial system state
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)
        initial_cpu = psutil.cpu_percent()
        
        start_time = time.perf_counter()
        response_times = []
        errors = 0
        
        if concurrent_users == 1:
            # Sequential execution
            for i in range(iterations):
                iteration_start = time.perf_counter()
                try:
                    test_function()
                except Exception as e:
                    errors += 1
                iteration_end = time.perf_counter()
                response_times.append((iteration_end - iteration_start) * 1000)
        else:
            # Concurrent execution
            response_times, errors = self._run_concurrent_test(
                test_function, iterations, concurrent_users
            )
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time_ms = (end_time - start_time) * 1000
        successful_ops = iterations - errors
        throughput = successful_ops / (total_time_ms / 1000) if total_time_ms > 0 else 0
        
        # Memory and CPU metrics
        final_memory = psutil.virtual_memory().used / (1024 * 1024)
        final_cpu = psutil.cpu_percent()
        memory_usage = final_memory - initial_memory
        
        # Response time percentiles
        if response_times:
            p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        else:
            p95_time = p99_time = 0
        
        return PerformanceMetrics(
            execution_time_ms=total_time_ms,
            memory_usage_mb=memory_usage,
            cpu_percent=final_cpu - initial_cpu,
            peak_memory_mb=max(psutil.virtual_memory().used / (1024 * 1024), final_memory),
            throughput_ops_per_sec=throughput,
            error_rate=errors / iterations if iterations > 0 else 0,
            p95_response_time=p95_time,
            p99_response_time=p99_time
        )
    
    def _run_concurrent_test(
        self, 
        test_function: Callable, 
        iterations: int, 
        concurrent_users: int
    ) -> tuple[List[float], int]:
        """Run concurrent performance test"""
        response_times = []
        errors = 0
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            for _ in range(iterations):
                future = executor.submit(self._timed_execution, test_function)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    execution_time = future.result()
                    response_times.append(execution_time)
                except Exception:
                    errors += 1
        
        return response_times, errors
    
    def _timed_execution(self, test_function: Callable) -> float:
        """Execute function with timing"""
        start_time = time.perf_counter()
        test_function()
        end_time = time.perf_counter()
        return (end_time - start_time) * 1000


@pytest.fixture
def performance_runner():
    """Performance test runner fixture"""
    return PerformanceTestRunner()


class LoadTestRunner:
    """Load testing runner for API endpoints"""
    
    def __init__(self):
        self.results = []
        
    async def run_load_test(
        self,
        test_function: Callable,
        concurrent_users: int = 10,
        duration_seconds: int = 60,
        ramp_up_seconds: int = 10
    ) -> LoadTestResults:
        """Run load test with gradual ramp-up"""
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        # Create semaphore for concurrent users
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def worker():
            nonlocal successful_requests, failed_requests
            
            while time.time() < end_time:
                async with semaphore:
                    request_start = time.time()
                    try:
                        await test_function()
                        successful_requests += 1
                    except Exception:
                        failed_requests += 1
                    request_end = time.time()
                    response_times.append((request_end - request_start) * 1000)
        
        # Gradual ramp-up
        tasks = []
        ramp_up_interval = ramp_up_seconds / concurrent_users
        
        for i in range(concurrent_users):
            if i > 0:
                await asyncio.sleep(ramp_up_interval)
            task = asyncio.create_task(worker())
            tasks.append(task)
        
        # Wait for test completion
        await asyncio.sleep(duration_seconds)
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Calculate results
        total_requests = successful_requests + failed_requests
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            # Calculate percentiles
            percentiles = {}
            for p in [50, 90, 95, 99]:
                percentiles[p] = statistics.quantiles(response_times, n=100)[p-1]
        else:
            avg_response_time = max_response_time = min_response_time = 0
            percentiles = {}
        
        return LoadTestResults(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            requests_per_second=total_requests / duration_seconds,
            error_rate=failed_requests / total_requests if total_requests > 0 else 0,
            response_time_percentiles=percentiles,
            memory_usage=self._get_memory_stats(),
            cpu_usage=self._get_cpu_stats()
        )
    
    def _get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        memory = psutil.virtual_memory()
        return {
            "used_mb": memory.used / (1024 * 1024),
            "available_mb": memory.available / (1024 * 1024),
            "percent": memory.percent
        }
    
    def _get_cpu_stats(self) -> Dict[str, float]:
        """Get CPU usage statistics"""
        return {
            "percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count()
        }


@pytest.fixture
def load_test_runner():
    """Load test runner fixture"""
    return LoadTestRunner()


@pytest.fixture
def memory_profiler():
    """Memory profiling fixture"""
    class MemoryProfiler:
        def __init__(self):
            self.snapshots = []
        
        def take_snapshot(self, label: str = ""):
            """Take memory snapshot"""
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            self.snapshots.append({
                "label": label,
                "timestamp": datetime.now(),
                "total_memory": sum(stat.size for stat in top_stats),
                "top_allocations": [
                    {
                        "size": stat.size,
                        "count": stat.count,
                        "traceback": stat.traceback.format()
                    }
                    for stat in top_stats[:10]
                ]
            })
        
        def compare_snapshots(self, label1: str, label2: str) -> Dict[str, Any]:
            """Compare two memory snapshots"""
            snap1 = next((s for s in self.snapshots if s["label"] == label1), None)
            snap2 = next((s for s in self.snapshots if s["label"] == label2), None)
            
            if not snap1 or not snap2:
                return {"error": "Snapshots not found"}
            
            return {
                "memory_diff": snap2["total_memory"] - snap1["total_memory"],
                "time_diff": (snap2["timestamp"] - snap1["timestamp"]).total_seconds(),
                "allocations_diff": len(snap2["top_allocations"]) - len(snap1["top_allocations"])
            }
    
    return MemoryProfiler()


@pytest.fixture
def performance_baseline():
    """Performance baseline expectations"""
    return {
        "nwtn_query_response_time_ms": 2000,  # Max 2 seconds
        "api_response_time_ms": 500,  # Max 500ms
        "database_query_time_ms": 100,  # Max 100ms
        "memory_usage_increase_mb": 50,  # Max 50MB increase per operation
        "throughput_ops_per_sec": 10,  # Min 10 ops/sec
        "error_rate_percent": 1.0,  # Max 1% error rate
        "cpu_usage_percent": 80  # Max 80% CPU usage
    }


@pytest.fixture
def stress_test_scenarios():
    """Predefined stress test scenarios"""
    return {
        "light_load": {
            "concurrent_users": 5,
            "duration_seconds": 30,
            "ramp_up_seconds": 5
        },
        "moderate_load": {
            "concurrent_users": 20,
            "duration_seconds": 60,
            "ramp_up_seconds": 10
        },
        "heavy_load": {
            "concurrent_users": 50,
            "duration_seconds": 120,
            "ramp_up_seconds": 20
        },
        "spike_test": {
            "concurrent_users": 100,
            "duration_seconds": 30,
            "ramp_up_seconds": 1
        }
    }


class PerformanceRegressionDetector:
    """Detect performance regressions"""
    
    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = baseline_file
        self.current_results = {}
        
    def record_result(self, test_name: str, metrics: PerformanceMetrics):
        """Record performance test result"""
        self.current_results[test_name] = {
            "execution_time_ms": metrics.execution_time_ms,
            "memory_usage_mb": metrics.memory_usage_mb,
            "throughput_ops_per_sec": metrics.throughput_ops_per_sec,
            "error_rate": metrics.error_rate,
            "timestamp": metrics.timestamp.isoformat()
        }
    
    def check_regression(
        self, 
        test_name: str, 
        tolerance_percent: float = 10.0
    ) -> Dict[str, Any]:
        """Check for performance regression"""
        try:
            import json
            
            # Load baseline
            try:
                with open(self.baseline_file, 'r') as f:
                    baseline = json.load(f)
            except FileNotFoundError:
                return {"status": "no_baseline", "message": "No baseline found"}
            
            if test_name not in baseline:
                return {"status": "no_baseline", "message": f"No baseline for {test_name}"}
            
            if test_name not in self.current_results:
                return {"status": "no_current", "message": f"No current results for {test_name}"}
            
            baseline_metrics = baseline[test_name]
            current_metrics = self.current_results[test_name]
            
            regressions = []
            
            # Check execution time regression
            baseline_time = baseline_metrics["execution_time_ms"]
            current_time = current_metrics["execution_time_ms"]
            time_increase = ((current_time - baseline_time) / baseline_time) * 100
            
            if time_increase > tolerance_percent:
                regressions.append({
                    "metric": "execution_time_ms",
                    "baseline": baseline_time,
                    "current": current_time,
                    "increase_percent": time_increase
                })
            
            # Check memory regression
            baseline_memory = baseline_metrics["memory_usage_mb"]
            current_memory = current_metrics["memory_usage_mb"]
            memory_increase = ((current_memory - baseline_memory) / baseline_memory) * 100
            
            if memory_increase > tolerance_percent:
                regressions.append({
                    "metric": "memory_usage_mb",
                    "baseline": baseline_memory,
                    "current": current_memory,
                    "increase_percent": memory_increase
                })
            
            # Check throughput regression
            baseline_throughput = baseline_metrics["throughput_ops_per_sec"]
            current_throughput = current_metrics["throughput_ops_per_sec"]
            throughput_decrease = ((baseline_throughput - current_throughput) / baseline_throughput) * 100
            
            if throughput_decrease > tolerance_percent:
                regressions.append({
                    "metric": "throughput_ops_per_sec",
                    "baseline": baseline_throughput,
                    "current": current_throughput,
                    "decrease_percent": throughput_decrease
                })
            
            return {
                "status": "regression" if regressions else "passed",
                "regressions": regressions,
                "summary": f"{len(regressions)} regression(s) detected"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def save_as_baseline(self):
        """Save current results as new baseline"""
        import json
        with open(self.baseline_file, 'w') as f:
            json.dump(self.current_results, f, indent=2)


@pytest.fixture
def regression_detector():
    """Performance regression detector fixture"""
    return PerformanceRegressionDetector()


@pytest.fixture
def benchmark_runner():
    """Pytest-benchmark runner fixture"""
    if pytest_benchmark is None:
        pytest.skip("pytest-benchmark not available")
    
    class BenchmarkRunner:
        @staticmethod
        def run_benchmark(benchmark, func, *args, **kwargs):
            """Run benchmark with pytest-benchmark"""
            return benchmark(func, *args, **kwargs)
        
        @staticmethod
        def compare_benchmarks(benchmark_results: List[Dict]) -> Dict[str, Any]:
            """Compare multiple benchmark results"""
            if len(benchmark_results) < 2:
                return {"error": "Need at least 2 results to compare"}
            
            comparisons = []
            baseline = benchmark_results[0]
            
            for i, result in enumerate(benchmark_results[1:], 1):
                comparison = {
                    "name": f"comparison_{i}",
                    "baseline_mean": baseline.get("mean", 0),
                    "current_mean": result.get("mean", 0),
                    "improvement_factor": baseline.get("mean", 1) / result.get("mean", 1) if result.get("mean", 0) > 0 else 0
                }
                comparisons.append(comparison)
            
            return {"comparisons": comparisons}
    
    return BenchmarkRunner()


# Resource monitoring fixtures

@pytest.fixture
def resource_monitor():
    """System resource monitoring during tests"""
    class ResourceMonitor:
        def __init__(self):
            self.monitoring = False
            self.samples = []
            self.monitor_thread = None
        
        def start_monitoring(self, interval_seconds: float = 1.0):
            """Start resource monitoring"""
            self.monitoring = True
            self.samples = []
            
            def monitor_loop():
                while self.monitoring:
                    self.samples.append({
                        "timestamp": time.time(),
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                        "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
                    })
                    time.sleep(interval_seconds)
            
            self.monitor_thread = threading.Thread(target=monitor_loop)
            self.monitor_thread.start()
        
        def stop_monitoring(self):
            """Stop resource monitoring"""
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
        
        def get_summary(self) -> Dict[str, Any]:
            """Get monitoring summary"""
            if not self.samples:
                return {}
            
            cpu_values = [s["cpu_percent"] for s in self.samples]
            memory_values = [s["memory_percent"] for s in self.samples]
            
            return {
                "duration_seconds": self.samples[-1]["timestamp"] - self.samples[0]["timestamp"],
                "cpu_stats": {
                    "avg": statistics.mean(cpu_values),
                    "max": max(cpu_values),
                    "min": min(cpu_values)
                },
                "memory_stats": {
                    "avg": statistics.mean(memory_values),
                    "max": max(memory_values),
                    "min": min(memory_values)
                },
                "sample_count": len(self.samples)
            }
    
    return ResourceMonitor()
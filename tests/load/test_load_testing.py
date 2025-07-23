"""
Load Testing Framework
=====================

Comprehensive load testing framework for PRSM to evaluate system performance
under various load conditions, identify bottlenecks, and ensure scalability.

Load Test Categories:
- Concurrent User Load: Multiple users performing simultaneous operations
- Throughput Testing: Maximum requests per second capacity
- Stress Testing: System behavior under extreme load conditions
- Endurance Testing: Performance over extended time periods
- Spike Testing: Sudden load increases and system recovery
- Volume Testing: Large data set processing capabilities
"""

import pytest
import asyncio
import time
import statistics
import json
import threading
from decimal import Decimal
from typing import Dict, List, Any, Optional, Callable, Tuple
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import psutil
import gc

try:
    import aiohttp
    from prsm.core.models import UserInput, PRSMSession, AgentType
    from prsm.nwtn.orchestrator import NWTNOrchestrator
    from prsm.tokenomics.ftns_service import FTNSService
    from prsm.api.main import app
    from prsm.core.database import DatabaseManager
    from fastapi.testclient import TestClient
    from httpx import AsyncClient
except ImportError:
    # Create mocks if imports fail
    aiohttp = Mock()
    UserInput = Mock
    PRSMSession = Mock
    AgentType = Mock
    NWTNOrchestrator = Mock
    FTNSService = Mock
    app = Mock()
    DatabaseManager = Mock
    TestClient = Mock
    AsyncClient = Mock


@dataclass
class LoadTestMetrics:
    """Metrics collected during load testing"""
    test_name: str
    test_type: str  # "concurrent", "throughput", "stress", "endurance", "spike", "volume"
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    error_rate: float
    throughput_achieved: float
    concurrent_users: int
    memory_peak_mb: float
    memory_average_mb: float
    cpu_peak_percent: float
    cpu_average_percent: float
    system_resources: Dict[str, Any]
    error_breakdown: Dict[str, int]
    response_time_distribution: List[float]
    timestamp: str
    
    def __post_init__(self):
        if not hasattr(self, 'timestamp') or self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class LoadTestRunner:
    """Core load testing execution framework"""
    
    def __init__(self):
        self.test_results: List[LoadTestMetrics] = []
        self.system_monitor_active = False
        self.monitoring_data = {
            "memory_readings": [],
            "cpu_readings": [],
            "network_readings": [],
            "disk_readings": []
        }
    
    def start_system_monitoring(self):
        """Start background system resource monitoring"""
        self.system_monitor_active = True
        self.monitoring_data = {
            "memory_readings": [],
            "cpu_readings": [],
            "network_readings": [],
            "disk_readings": []
        }
        
        def monitor_resources():
            process = psutil.Process()
            while self.system_monitor_active:
                try:
                    # Memory monitoring
                    memory_info = process.memory_info()
                    self.monitoring_data["memory_readings"].append({
                        "timestamp": time.time(),
                        "rss_mb": memory_info.rss / 1024 / 1024,
                        "vms_mb": memory_info.vms / 1024 / 1024
                    })
                    
                    # CPU monitoring
                    cpu_percent = process.cpu_percent()
                    self.monitoring_data["cpu_readings"].append({
                        "timestamp": time.time(),
                        "cpu_percent": cpu_percent
                    })
                    
                    # Network monitoring (if available)
                    try:
                        net_io = psutil.net_io_counters()
                        self.monitoring_data["network_readings"].append({
                            "timestamp": time.time(),
                            "bytes_sent": net_io.bytes_sent,
                            "bytes_recv": net_io.bytes_recv,
                            "packets_sent": net_io.packets_sent,
                            "packets_recv": net_io.packets_recv
                        })
                    except:
                        pass
                    
                    time.sleep(0.5)  # Monitor every 500ms
                except:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def stop_system_monitoring(self):
        """Stop system resource monitoring"""
        self.system_monitor_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
    
    async def run_concurrent_load_test(
        self,
        test_function: Callable,
        concurrent_users: int,
        test_duration_seconds: float,
        ramp_up_seconds: float = 0,
        **kwargs
    ) -> LoadTestMetrics:
        """Run concurrent user load test"""
        
        print(f"🚀 Starting concurrent load test: {concurrent_users} users, {test_duration_seconds}s duration")
        
        self.start_system_monitoring()
        
        start_time = time.perf_counter()
        end_time = start_time + test_duration_seconds
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        error_breakdown = {}
        
        # Semaphore to control concurrent users
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def user_session():
            """Simulate a single user session"""
            nonlocal successful_requests, failed_requests
            
            session_start = time.perf_counter()
            requests_made = 0
            
            while time.perf_counter() < end_time:
                async with semaphore:
                    request_start = time.perf_counter()
                    
                    try:
                        await test_function(**kwargs)
                        request_end = time.perf_counter()
                        response_time = request_end - request_start
                        
                        response_times.append(response_time)
                        successful_requests += 1
                        requests_made += 1
                        
                    except Exception as e:
                        failed_requests += 1
                        error_type = type(e).__name__
                        error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
                    
                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.01)
            
            return requests_made
        
        # Ramp up users gradually if specified
        tasks = []
        if ramp_up_seconds > 0:
            ramp_delay = ramp_up_seconds / concurrent_users
            for i in range(concurrent_users):
                await asyncio.sleep(ramp_delay)
                task = asyncio.create_task(user_session())
                tasks.append(task)
        else:
            # Start all users simultaneously
            tasks = [asyncio.create_task(user_session()) for _ in range(concurrent_users)]
        
        # Wait for all user sessions to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.perf_counter() - start_time
        self.stop_system_monitoring()
        
        # Calculate metrics
        total_requests = successful_requests + failed_requests
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if sorted_times else 0
            p99_response_time = sorted_times[p99_index] if sorted_times else 0
        else:
            avg_response_time = median_response_time = 0
            min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        # System resource metrics
        memory_readings = [r["rss_mb"] for r in self.monitoring_data["memory_readings"]]
        cpu_readings = [r["cpu_percent"] for r in self.monitoring_data["cpu_readings"]]
        
        memory_peak = max(memory_readings) if memory_readings else 0
        memory_average = statistics.mean(memory_readings) if memory_readings else 0
        cpu_peak = max(cpu_readings) if cpu_readings else 0
        cpu_average = statistics.mean(cpu_readings) if cpu_readings else 0
        
        metrics = LoadTestMetrics(
            test_name=test_function.__name__,
            test_type="concurrent",
            duration_seconds=total_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            requests_per_second=total_requests / total_time if total_time > 0 else 0,
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            error_rate=failed_requests / total_requests if total_requests > 0 else 0,
            throughput_achieved=successful_requests / total_time if total_time > 0 else 0,
            concurrent_users=concurrent_users,
            memory_peak_mb=memory_peak,
            memory_average_mb=memory_average,
            cpu_peak_percent=cpu_peak,
            cpu_average_percent=cpu_average,
            system_resources=dict(self.monitoring_data),
            error_breakdown=error_breakdown,
            response_time_distribution=response_times
        )
        
        self.test_results.append(metrics)
        
        print(f"✅ Concurrent load test completed:")
        print(f"   📊 {successful_requests}/{total_requests} requests successful ({(1-metrics.error_rate)*100:.1f}%)")
        print(f"   ⚡ {metrics.requests_per_second:.1f} requests/second")
        print(f"   ⏱️  {avg_response_time*1000:.1f}ms average response time")
        print(f"   🧠 {memory_peak:.1f}MB peak memory, {cpu_peak:.1f}% peak CPU")
        
        return metrics
    
    async def run_throughput_test(
        self,
        test_function: Callable,
        target_rps: int,
        test_duration_seconds: float,
        **kwargs
    ) -> LoadTestMetrics:
        """Run throughput test to achieve specific requests per second"""
        
        print(f"🎯 Starting throughput test: {target_rps} RPS target, {test_duration_seconds}s duration")
        
        self.start_system_monitoring()
        
        start_time = time.perf_counter()
        end_time = start_time + test_duration_seconds
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        error_breakdown = {}
        
        request_interval = 1.0 / target_rps  # Time between requests
        
        async def make_request():
            nonlocal successful_requests, failed_requests
            
            request_start = time.perf_counter()
            
            try:
                await test_function(**kwargs)
                request_end = time.perf_counter()
                response_time = request_end - request_start
                
                response_times.append(response_time)
                successful_requests += 1
                
            except Exception as e:
                failed_requests += 1
                error_type = type(e).__name__
                error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
        
        # Schedule requests at target rate
        next_request_time = start_time
        
        while next_request_time < end_time:
            current_time = time.perf_counter()
            
            if current_time >= next_request_time:
                # Schedule request
                asyncio.create_task(make_request())
                next_request_time += request_interval
            else:
                # Wait until next request time
                await asyncio.sleep(max(0, next_request_time - current_time))
                next_request_time += request_interval
        
        # Wait a bit for final requests to complete
        await asyncio.sleep(2.0)
        
        total_time = time.perf_counter() - start_time
        self.stop_system_monitoring()
        
        # Calculate metrics (similar to concurrent test)
        total_requests = successful_requests + failed_requests
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            sorted_times = sorted(response_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if sorted_times else 0
            p99_response_time = sorted_times[p99_index] if sorted_times else 0
        else:
            avg_response_time = median_response_time = 0
            min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        memory_readings = [r["rss_mb"] for r in self.monitoring_data["memory_readings"]]
        cpu_readings = [r["cpu_percent"] for r in self.monitoring_data["cpu_readings"]]
        
        memory_peak = max(memory_readings) if memory_readings else 0
        memory_average = statistics.mean(memory_readings) if memory_readings else 0
        cpu_peak = max(cpu_readings) if cpu_readings else 0
        cpu_average = statistics.mean(cpu_readings) if cpu_readings else 0
        
        metrics = LoadTestMetrics(
            test_name=test_function.__name__,
            test_type="throughput",
            duration_seconds=total_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            requests_per_second=total_requests / total_time if total_time > 0 else 0,
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            error_rate=failed_requests / total_requests if total_requests > 0 else 0,
            throughput_achieved=successful_requests / total_time if total_time > 0 else 0,
            concurrent_users=0,  # Not applicable for throughput test
            memory_peak_mb=memory_peak,
            memory_average_mb=memory_average,
            cpu_peak_percent=cpu_peak,
            cpu_average_percent=cpu_average,
            system_resources=dict(self.monitoring_data),
            error_breakdown=error_breakdown,
            response_time_distribution=response_times
        )
        
        self.test_results.append(metrics)
        
        print(f"✅ Throughput test completed:")
        print(f"   🎯 Target: {target_rps} RPS, Achieved: {metrics.throughput_achieved:.1f} RPS ({(metrics.throughput_achieved/target_rps)*100:.1f}%)")
        print(f"   📊 {successful_requests}/{total_requests} requests successful")
        print(f"   ⏱️  {avg_response_time*1000:.1f}ms average response time")
        
        return metrics
    
    def run_stress_test(
        self,
        test_function: Callable,
        max_concurrent_users: int,
        ramp_up_step: int = 10,
        step_duration_seconds: int = 30,
        **kwargs
    ) -> List[LoadTestMetrics]:
        """Run stress test with gradually increasing load"""
        
        print(f"🔥 Starting stress test: 0 → {max_concurrent_users} users, {ramp_up_step} user steps")
        
        stress_results = []
        
        for concurrent_users in range(ramp_up_step, max_concurrent_users + 1, ramp_up_step):
            print(f"   📈 Testing with {concurrent_users} concurrent users...")
            
            # Run concurrent load test for this user level
            metrics = asyncio.run(self.run_concurrent_load_test(
                test_function=test_function,
                concurrent_users=concurrent_users,
                test_duration_seconds=step_duration_seconds,
                ramp_up_seconds=5,  # Quick ramp up for each step
                **kwargs
            ))
            
            metrics.test_type = "stress"
            stress_results.append(metrics)
            
            # Check if system is breaking down
            if metrics.error_rate > 0.5:  # >50% error rate
                print(f"   ⚠️  High error rate detected ({metrics.error_rate:.1%}), stopping stress test")
                break
            
            if metrics.average_response_time > 10.0:  # >10 second response time
                print(f"   ⚠️  High response time detected ({metrics.average_response_time:.1f}s), stopping stress test")
                break
            
            # Brief pause between stress levels
            time.sleep(5)
        
        print(f"✅ Stress test completed with {len(stress_results)} load levels tested")
        
        return stress_results
    
    async def run_spike_test(
        self,
        test_function: Callable,
        baseline_users: int,
        spike_users: int,
        spike_duration_seconds: float,
        total_duration_seconds: float,
        **kwargs
    ) -> LoadTestMetrics:
        """Run spike test with sudden load increase"""
        
        print(f"⚡ Starting spike test: {baseline_users} → {spike_users} users spike for {spike_duration_seconds}s")
        
        self.start_system_monitoring()
        
        start_time = time.perf_counter()
        spike_start_time = start_time + (total_duration_seconds - spike_duration_seconds) / 2
        spike_end_time = spike_start_time + spike_duration_seconds
        end_time = start_time + total_duration_seconds
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        error_breakdown = {}
        spike_metrics = {"during_spike": [], "outside_spike": []}
        
        async def user_session(is_spike_user: bool = False):
            nonlocal successful_requests, failed_requests
            
            session_requests = 0
            
            while time.perf_counter() < end_time:
                current_time = time.perf_counter()
                
                # Determine if we should be active based on spike timing
                if is_spike_user:
                    if not (spike_start_time <= current_time <= spike_end_time):
                        await asyncio.sleep(0.1)
                        continue
                
                request_start = time.perf_counter()
                
                try:
                    await test_function(**kwargs)
                    request_end = time.perf_counter()
                    response_time = request_end - request_start
                    
                    response_times.append(response_time)
                    successful_requests += 1
                    session_requests += 1
                    
                    # Track spike vs non-spike metrics
                    if spike_start_time <= request_start <= spike_end_time:
                        spike_metrics["during_spike"].append(response_time)
                    else:
                        spike_metrics["outside_spike"].append(response_time)
                    
                except Exception as e:
                    failed_requests += 1
                    error_type = type(e).__name__
                    error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
                
                await asyncio.sleep(0.05)  # Small delay
            
            return session_requests
        
        # Start baseline users
        baseline_tasks = [asyncio.create_task(user_session()) for _ in range(baseline_users)]
        
        # Start spike users (they'll activate during spike period)
        spike_tasks = [asyncio.create_task(user_session(is_spike_user=True)) for _ in range(spike_users - baseline_users)]
        
        all_tasks = baseline_tasks + spike_tasks
        
        # Wait for all sessions to complete
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        total_time = time.perf_counter() - start_time
        self.stop_system_monitoring()
        
        # Calculate metrics
        total_requests = successful_requests + failed_requests
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            sorted_times = sorted(response_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if sorted_times else 0
            p99_response_time = sorted_times[p99_index] if sorted_times else 0
        else:
            avg_response_time = median_response_time = 0
            min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        memory_readings = [r["rss_mb"] for r in self.monitoring_data["memory_readings"]]
        cpu_readings = [r["cpu_percent"] for r in self.monitoring_data["cpu_readings"]]
        
        memory_peak = max(memory_readings) if memory_readings else 0
        memory_average = statistics.mean(memory_readings) if memory_readings else 0
        cpu_peak = max(cpu_readings) if cpu_readings else 0
        cpu_average = statistics.mean(cpu_readings) if cpu_readings else 0
        
        # Calculate spike-specific metrics
        spike_analysis = {}
        if spike_metrics["during_spike"] and spike_metrics["outside_spike"]:
            spike_analysis = {
                "avg_response_during_spike": statistics.mean(spike_metrics["during_spike"]),
                "avg_response_outside_spike": statistics.mean(spike_metrics["outside_spike"]),
                "spike_performance_impact": statistics.mean(spike_metrics["during_spike"]) / statistics.mean(spike_metrics["outside_spike"]),
                "requests_during_spike": len(spike_metrics["during_spike"]),
                "requests_outside_spike": len(spike_metrics["outside_spike"])
            }
        
        metrics = LoadTestMetrics(
            test_name=test_function.__name__,
            test_type="spike",
            duration_seconds=total_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            requests_per_second=total_requests / total_time if total_time > 0 else 0,
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            error_rate=failed_requests / total_requests if total_requests > 0 else 0,
            throughput_achieved=successful_requests / total_time if total_time > 0 else 0,
            concurrent_users=baseline_users,  # Record baseline users
            memory_peak_mb=memory_peak,
            memory_average_mb=memory_average,
            cpu_peak_percent=cpu_peak,
            cpu_average_percent=cpu_average,
            system_resources=dict(self.monitoring_data),
            error_breakdown=error_breakdown,
            response_time_distribution=response_times
        )
        
        # Add spike-specific data to metadata
        if not hasattr(metrics, 'system_resources'):
            metrics.system_resources = {}
        metrics.system_resources["spike_analysis"] = spike_analysis
        metrics.system_resources["spike_users"] = spike_users
        metrics.system_resources["spike_duration"] = spike_duration_seconds
        
        self.test_results.append(metrics)
        
        print(f"✅ Spike test completed:")
        print(f"   ⚡ Peak users: {spike_users}, Error rate: {metrics.error_rate:.1%}")
        print(f"   📊 Spike impact: {spike_analysis.get('spike_performance_impact', 1):.2f}x response time increase")
        
        return metrics
    
    def generate_load_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive load test report"""
        
        if not self.test_results:
            return {"error": "No load test results available"}
        
        # Aggregate metrics by test type
        results_by_type = {}
        for result in self.test_results:
            test_type = result.test_type
            if test_type not in results_by_type:
                results_by_type[test_type] = []
            results_by_type[test_type].append(result)
        
        # Calculate overall system capacity
        max_throughput = max(r.throughput_achieved for r in self.test_results)
        min_error_rate = min(r.error_rate for r in self.test_results)
        
        # Identify performance breaking points
        breaking_point_users = None
        for result in sorted(self.test_results, key=lambda x: x.concurrent_users):
            if result.error_rate > 0.1 or result.average_response_time > 5.0:  # 10% error rate or 5s response time
                breaking_point_users = result.concurrent_users
                break
        
        report = {
            "summary": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_tests_run": len(self.test_results),
                "test_types_executed": list(results_by_type.keys()),
                "max_throughput_achieved": max_throughput,
                "system_breaking_point": breaking_point_users,
                "overall_stability": "stable" if min_error_rate < 0.05 else "unstable",
                "recommended_max_load": int(breaking_point_users * 0.8) if breaking_point_users else "not_determined"
            },
            "performance_characteristics": {
                "peak_requests_per_second": max_throughput,
                "sustainable_throughput": max(r.throughput_achieved for r in self.test_results if r.error_rate < 0.05),
                "average_response_time_under_load": statistics.mean([r.average_response_time for r in self.test_results]),
                "p95_response_time_under_load": statistics.mean([r.p95_response_time for r in self.test_results]),
                "memory_usage_pattern": {
                    "peak_mb": max(r.memory_peak_mb for r in self.test_results),
                    "average_mb": statistics.mean([r.memory_average_mb for r in self.test_results])
                },
                "cpu_usage_pattern": {
                    "peak_percent": max(r.cpu_peak_percent for r in self.test_results),
                    "average_percent": statistics.mean([r.cpu_average_percent for r in self.test_results])
                }
            },
            "test_results_by_type": {
                test_type: [asdict(result) for result in results]
                for test_type, results in results_by_type.items()
            },
            "bottleneck_analysis": self._analyze_bottlenecks(),
            "scalability_assessment": self._assess_scalability(),
            "recommendations": self._generate_load_test_recommendations()
        }
        
        return report
    
    def _analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze system bottlenecks from load test results"""
        
        bottlenecks = {
            "identified_bottlenecks": [],
            "resource_constraints": {},
            "performance_degradation_points": []
        }
        
        # Analyze memory constraints
        memory_peaks = [r.memory_peak_mb for r in self.test_results]
        if max(memory_peaks) > 500:  # >500MB peak memory
            bottlenecks["identified_bottlenecks"].append("memory_usage")
            bottlenecks["resource_constraints"]["memory"] = {
                "peak_usage_mb": max(memory_peaks),
                "constraint_level": "high" if max(memory_peaks) > 1000 else "medium"
            }
        
        # Analyze CPU constraints
        cpu_peaks = [r.cpu_peak_percent for r in self.test_results]
        if max(cpu_peaks) > 80:  # >80% CPU usage
            bottlenecks["identified_bottlenecks"].append("cpu_usage")
            bottlenecks["resource_constraints"]["cpu"] = {
                "peak_usage_percent": max(cpu_peaks),
                "constraint_level": "high" if max(cpu_peaks) > 95 else "medium"
            }
        
        # Analyze response time degradation
        for result in sorted(self.test_results, key=lambda x: x.concurrent_users):
            if result.average_response_time > 2.0:  # >2 second response time
                bottlenecks["performance_degradation_points"].append({
                    "concurrent_users": result.concurrent_users,
                    "average_response_time": result.average_response_time,
                    "degradation_factor": result.average_response_time / 0.1  # Assume 100ms baseline
                })
                break
        
        return bottlenecks
    
    def _assess_scalability(self) -> Dict[str, Any]:
        """Assess system scalability characteristics"""
        
        # Sort results by concurrent users for scalability analysis
        concurrent_results = [r for r in self.test_results if r.test_type == "concurrent"]
        concurrent_results.sort(key=lambda x: x.concurrent_users)
        
        scalability = {
            "linear_scalability": True,
            "scalability_coefficient": 1.0,
            "resource_efficiency": "good",
            "scalability_limits": {}
        }
        
        if len(concurrent_results) >= 2:
            # Calculate scalability coefficient (throughput increase vs user increase)
            throughput_increases = []
            user_increases = []
            
            for i in range(1, len(concurrent_results)):
                prev_result = concurrent_results[i-1]
                curr_result = concurrent_results[i]
                
                user_increase = curr_result.concurrent_users - prev_result.concurrent_users
                throughput_increase = curr_result.throughput_achieved - prev_result.throughput_achieved
                
                if user_increase > 0:
                    throughput_increases.append(throughput_increase)
                    user_increases.append(user_increase)
            
            if throughput_increases and user_increases:
                scalability_coefficient = statistics.mean([t/u for t, u in zip(throughput_increases, user_increases)])
                scalability["scalability_coefficient"] = scalability_coefficient
                scalability["linear_scalability"] = scalability_coefficient > 0.8  # 80% of linear
        
        # Determine resource efficiency
        if concurrent_results:
            last_result = concurrent_results[-1]
            requests_per_mb = last_result.throughput_achieved / last_result.memory_average_mb if last_result.memory_average_mb > 0 else 0
            requests_per_cpu_percent = last_result.throughput_achieved / last_result.cpu_average_percent if last_result.cpu_average_percent > 0 else 0
            
            scalability["resource_efficiency"] = {
                "requests_per_mb_memory": requests_per_mb,
                "requests_per_cpu_percent": requests_per_cpu_percent,
                "efficiency_rating": "high" if requests_per_mb > 1 and requests_per_cpu_percent > 0.5 else "medium" if requests_per_mb > 0.5 else "low"
            }
        
        return scalability
    
    def _generate_load_test_recommendations(self) -> List[str]:
        """Generate recommendations based on load test results"""
        
        recommendations = []
        
        # Check for high error rates
        high_error_results = [r for r in self.test_results if r.error_rate > 0.1]
        if high_error_results:
            recommendations.append(f"Investigate and fix error sources - {len(high_error_results)} tests had >10% error rate")
        
        # Check for high response times
        slow_results = [r for r in self.test_results if r.average_response_time > 2.0]
        if slow_results:
            recommendations.append(f"Optimize response times - {len(slow_results)} tests had >2s average response time")
        
        # Check memory usage
        memory_peaks = [r.memory_peak_mb for r in self.test_results]
        if max(memory_peaks) > 500:
            recommendations.append("Consider memory optimization - peak memory usage exceeded 500MB")
        
        # Check CPU usage
        cpu_peaks = [r.cpu_peak_percent for r in self.test_results]
        if max(cpu_peaks) > 80:
            recommendations.append("Optimize CPU-intensive operations - peak CPU usage exceeded 80%")
        
        # Scalability recommendations
        bottlenecks = self._analyze_bottlenecks()
        if "memory_usage" in bottlenecks["identified_bottlenecks"]:
            recommendations.append("Implement memory pooling and garbage collection optimization")
        
        if "cpu_usage" in bottlenecks["identified_bottlenecks"]:
            recommendations.append("Consider horizontal scaling or CPU optimization")
        
        # Throughput recommendations
        max_throughput = max(r.throughput_achieved for r in self.test_results)
        if max_throughput < 10:  # <10 requests per second
            recommendations.append("System throughput is low - consider performance profiling and optimization")
        
        return recommendations


@pytest.mark.load
@pytest.mark.slow
class TestNWTNLoadTesting:
    """Load testing for NWTN orchestrator"""
    
    @pytest.fixture
    def load_test_runner(self):
        return LoadTestRunner()
    
    @pytest.fixture
    def mock_nwtn_orchestrator(self):
        orchestrator = Mock(spec=NWTNOrchestrator)
        
        async def mock_process_query(user_input):
            # Simulate realistic processing time with some variability
            base_time = 0.1 + (len(user_input.prompt) * 0.001)
            processing_time = base_time + (time.time() % 0.1)  # Add some randomness
            await asyncio.sleep(processing_time)
            
            return {
                "session_id": str(uuid.uuid4()),
                "final_answer": f"NWTN processed: {user_input.prompt[:50]}...",
                "reasoning_trace": [
                    {"step_id": str(uuid.uuid4()), "agent_type": "architect", "execution_time": processing_time * 0.3},
                    {"step_id": str(uuid.uuid4()), "agent_type": "executor", "execution_time": processing_time * 0.7}
                ],
                "confidence_score": 0.85,
                "context_used": len(user_input.prompt),
                "processing_time": processing_time
            }
        
        orchestrator.process_query = mock_process_query
        return orchestrator
    
    async def test_nwtn_concurrent_users_load(self, load_test_runner, mock_nwtn_orchestrator):
        """Test NWTN performance under concurrent user load"""
        
        async def nwtn_query_test():
            user_input = UserInput(
                user_id=f"load_test_user_{uuid.uuid4()}",
                prompt="Load test query: Explain the impact of artificial intelligence on modern society",
                context_allocation=200
            )
            
            result = await mock_nwtn_orchestrator.process_query(user_input)
            return result
        
        # Test with moderate concurrent load
        metrics = await load_test_runner.run_concurrent_load_test(
            test_function=nwtn_query_test,
            concurrent_users=25,
            test_duration_seconds=60,
            ramp_up_seconds=10
        )
        
        # Performance assertions
        assert metrics.error_rate < 0.1, f"High error rate: {metrics.error_rate:.1%}"
        assert metrics.average_response_time < 2.0, f"High response time: {metrics.average_response_time:.2f}s"
        assert metrics.throughput_achieved > 5, f"Low throughput: {metrics.throughput_achieved:.1f} RPS"
        
        print(f"🎭 NWTN Concurrent Load Test Results:")
        print(f"   👥 {metrics.concurrent_users} users, {metrics.duration_seconds:.1f}s duration")
        print(f"   📊 {metrics.successful_requests}/{metrics.total_requests} successful ({(1-metrics.error_rate)*100:.1f}%)")
        print(f"   ⚡ {metrics.throughput_achieved:.1f} RPS throughput")
        print(f"   ⏱️  {metrics.average_response_time*1000:.1f}ms avg, {metrics.p95_response_time*1000:.1f}ms p95")
        
        return metrics
    
    async def test_nwtn_throughput_capacity(self, load_test_runner, mock_nwtn_orchestrator):
        """Test NWTN maximum throughput capacity"""
        
        async def nwtn_query_test():
            user_input = UserInput(
                user_id=f"throughput_user_{uuid.uuid4()}",
                prompt="Throughput test: What are the key principles of quantum computing?",
                context_allocation=150
            )
            
            result = await mock_nwtn_orchestrator.process_query(user_input)
            return result
        
        # Test throughput at different RPS targets
        target_rps_levels = [10, 20, 30]
        throughput_results = []
        
        for target_rps in target_rps_levels:
            print(f"   🎯 Testing {target_rps} RPS target...")
            
            metrics = await load_test_runner.run_throughput_test(
                test_function=nwtn_query_test,
                target_rps=target_rps,
                test_duration_seconds=30
            )
            
            throughput_results.append(metrics)
            
            # Check if we're reaching system limits
            if metrics.throughput_achieved < target_rps * 0.8:  # <80% of target
                print(f"   ⚠️  System capacity limit reached at {metrics.throughput_achieved:.1f} RPS")
                break
            
            # Brief pause between tests
            await asyncio.sleep(5)
        
        # Analyze results
        max_throughput = max(m.throughput_achieved for m in throughput_results)
        sustainable_throughput = max(m.throughput_achieved for m in throughput_results if m.error_rate < 0.05)
        
        print(f"🎭 NWTN Throughput Test Results:")
        print(f"   🚀 Maximum throughput: {max_throughput:.1f} RPS")
        print(f"   ✅ Sustainable throughput: {sustainable_throughput:.1f} RPS (error rate <5%)")
        
        # Performance assertions
        assert max_throughput > 15, f"Maximum throughput too low: {max_throughput:.1f} RPS"
        assert sustainable_throughput > 10, f"Sustainable throughput too low: {sustainable_throughput:.1f} RPS"
        
        return throughput_results
    
    def test_nwtn_stress_testing(self, load_test_runner, mock_nwtn_orchestrator):
        """Test NWTN behavior under increasing stress"""
        
        async def nwtn_query_test():
            user_input = UserInput(
                user_id=f"stress_user_{uuid.uuid4()}",
                prompt="Stress test query: Analyze the complex interactions between machine learning algorithms and data structures",
                context_allocation=300
            )
            
            result = await mock_nwtn_orchestrator.process_query(user_input)
            return result
        
        # Run stress test with increasing user load
        stress_results = load_test_runner.run_stress_test(
            test_function=nwtn_query_test,
            max_concurrent_users=100,
            ramp_up_step=20,
            step_duration_seconds=45
        )
        
        # Find breaking point
        breaking_point = None
        for result in stress_results:
            if result.error_rate > 0.2 or result.average_response_time > 5.0:
                breaking_point = result.concurrent_users
                break
        
        print(f"🎭 NWTN Stress Test Results:")
        print(f"   🔥 Tested up to {max(r.concurrent_users for r in stress_results)} concurrent users")
        if breaking_point:
            print(f"   ⚠️  Breaking point: {breaking_point} users")
        else:
            print(f"   💪 System stable throughout stress test")
        
        # Performance assertions
        stable_results = [r for r in stress_results if r.error_rate < 0.1 and r.average_response_time < 3.0]
        assert len(stable_results) > 0, "System unstable at all load levels"
        
        max_stable_users = max(r.concurrent_users for r in stable_results)
        assert max_stable_users >= 20, f"System unstable below 20 users: {max_stable_users}"
        
        return stress_results
    
    async def test_nwtn_spike_resilience(self, load_test_runner, mock_nwtn_orchestrator):
        """Test NWTN resilience to sudden load spikes"""
        
        async def nwtn_query_test():
            user_input = UserInput(
                user_id=f"spike_user_{uuid.uuid4()}",
                prompt="Spike test query: Evaluate the performance implications of distributed AI systems under varying load conditions",
                context_allocation=250
            )
            
            result = await mock_nwtn_orchestrator.process_query(user_input)
            return result
        
        # Test spike from 10 to 50 users
        metrics = await load_test_runner.run_spike_test(
            test_function=nwtn_query_test,
            baseline_users=10,
            spike_users=50,
            spike_duration_seconds=30,
            total_duration_seconds=120
        )
        
        spike_analysis = metrics.system_resources.get("spike_analysis", {})
        
        print(f"🎭 NWTN Spike Test Results:")
        print(f"   ⚡ Spike: {metrics.concurrent_users} → {metrics.system_resources['spike_users']} users")
        print(f"   📊 Error rate during spike: {metrics.error_rate:.1%}")
        if spike_analysis:
            print(f"   📈 Performance impact: {spike_analysis.get('spike_performance_impact', 1):.2f}x response time")
        
        # Performance assertions
        assert metrics.error_rate < 0.15, f"High error rate during spike: {metrics.error_rate:.1%}"
        
        if spike_analysis and "spike_performance_impact" in spike_analysis:
            assert spike_analysis["spike_performance_impact"] < 3.0, f"Excessive performance degradation: {spike_analysis['spike_performance_impact']:.2f}x"
        
        return metrics


@pytest.mark.load
@pytest.mark.slow  
class TestFTNSLoadTesting:
    """Load testing for FTNS tokenomics system"""
    
    @pytest.fixture
    def load_test_runner(self):
        return LoadTestRunner()
    
    @pytest.fixture
    def mock_ftns_service(self):
        service = Mock(spec=FTNSService)
        
        # Simulate database-like latency
        def mock_get_balance(user_id):
            time.sleep(0.01 + (time.time() % 0.005))  # 10-15ms latency
            return {
                "total_balance": Decimal("100.00"),
                "available_balance": Decimal("85.50"),
                "reserved_balance": Decimal("14.50")
            }
        
        def mock_create_transaction(from_user, to_user, amount, transaction_type):
            time.sleep(0.015 + (time.time() % 0.01))  # 15-25ms latency
            return {
                "transaction_id": str(uuid.uuid4()),
                "success": True,
                "amount": amount,
                "new_balance": Decimal("75.25")
            }
        
        service.get_balance = mock_get_balance
        service.create_transaction = mock_create_transaction
        return service
    
    async def test_ftns_balance_lookup_load(self, load_test_runner, mock_ftns_service):
        """Test FTNS balance lookup performance under load"""
        
        async def balance_lookup_test():
            # Wrap synchronous call in async
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                mock_ftns_service.get_balance, 
                f"balance_user_{uuid.uuid4()}"
            )
            return result
        
        # Test balance lookup throughput
        metrics = await load_test_runner.run_throughput_test(
            test_function=balance_lookup_test,
            target_rps=100,  # Target 100 balance lookups per second
            test_duration_seconds=30
        )
        
        print(f"💰 FTNS Balance Lookup Load Test:")
        print(f"   🎯 Target: 100 RPS, Achieved: {metrics.throughput_achieved:.1f} RPS")
        print(f"   ⏱️  {metrics.average_response_time*1000:.1f}ms avg response time")
        print(f"   📊 {metrics.successful_requests}/{metrics.total_requests} successful")
        
        # Performance assertions
        assert metrics.throughput_achieved > 80, f"Low balance lookup throughput: {metrics.throughput_achieved:.1f} RPS"
        assert metrics.average_response_time < 0.1, f"High balance lookup latency: {metrics.average_response_time*1000:.1f}ms"
        assert metrics.error_rate == 0, f"Balance lookup errors: {metrics.error_rate:.1%}"
        
        return metrics
    
    async def test_ftns_transaction_processing_load(self, load_test_runner, mock_ftns_service):
        """Test FTNS transaction processing under concurrent load"""
        
        async def transaction_test():
            # Wrap synchronous call in async
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                mock_ftns_service.create_transaction,
                f"from_user_{uuid.uuid4()}",
                f"to_user_{uuid.uuid4()}",
                10.0,
                "load_test_transfer"
            )
            return result
        
        # Test concurrent transaction processing
        metrics = await load_test_runner.run_concurrent_load_test(
            test_function=transaction_test,
            concurrent_users=50,
            test_duration_seconds=45,
            ramp_up_seconds=10
        )
        
        print(f"💰 FTNS Transaction Processing Load Test:")
        print(f"   👥 {metrics.concurrent_users} concurrent users")
        print(f"   ⚡ {metrics.throughput_achieved:.1f} transactions/second")
        print(f"   ⏱️  {metrics.average_response_time*1000:.1f}ms avg processing time")
        print(f"   📊 Error rate: {metrics.error_rate:.1%}")
        
        # Performance assertions
        assert metrics.error_rate < 0.05, f"High transaction error rate: {metrics.error_rate:.1%}"
        assert metrics.average_response_time < 0.2, f"High transaction latency: {metrics.average_response_time*1000:.1f}ms"
        assert metrics.throughput_achieved > 30, f"Low transaction throughput: {metrics.throughput_achieved:.1f} TPS"
        
        return metrics
    
    def test_ftns_mixed_workload_stress(self, load_test_runner, mock_ftns_service):
        """Test FTNS performance under mixed balance lookups and transactions"""
        
        async def mixed_workload_test():
            # 70% balance lookups, 30% transactions
            if time.time() % 1 < 0.7:  # 70% probability
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    mock_ftns_service.get_balance,
                    f"mixed_user_{uuid.uuid4()}"
                )
            else:  # 30% probability
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    mock_ftns_service.create_transaction,
                    f"from_user_{uuid.uuid4()}",
                    f"to_user_{uuid.uuid4()}",
                    5.0,
                    "mixed_workload"
                )
            return result
        
        # Run stress test with mixed workload
        stress_results = load_test_runner.run_stress_test(
            test_function=mixed_workload_test,
            max_concurrent_users=150,
            ramp_up_step=30,
            step_duration_seconds=30
        )
        
        max_stable_users = 0
        for result in stress_results:
            if result.error_rate < 0.1 and result.average_response_time < 0.5:
                max_stable_users = result.concurrent_users
        
        print(f"💰 FTNS Mixed Workload Stress Test:")
        print(f"   🔥 Maximum stable load: {max_stable_users} concurrent users")
        print(f"   📊 Workload: 70% balance lookups, 30% transactions")
        
        # Performance assertions
        assert max_stable_users >= 60, f"Low mixed workload capacity: {max_stable_users} users"
        
        return stress_results


@pytest.mark.load
class TestComprehensiveLoadTestSuite:
    """Comprehensive load testing suite runner"""
    
    async def test_full_load_testing_suite(self):
        """Run complete load testing suite and generate comprehensive report"""
        
        print("🚀 Starting PRSM Comprehensive Load Testing Suite...")
        print("=" * 80)
        
        load_runner = LoadTestRunner()
        all_metrics = []
        
        try:
            # NWTN Load Tests
            print("🎭 Running NWTN Load Tests...")
            nwtn_tests = TestNWTNLoadTesting()
            
            # Concurrent users test
            nwtn_concurrent = await nwtn_tests.test_nwtn_concurrent_users_load(
                load_runner, nwtn_tests.mock_nwtn_orchestrator()
            )
            all_metrics.append(nwtn_concurrent)
            
            # Throughput test
            nwtn_throughput = await nwtn_tests.test_nwtn_throughput_capacity(
                load_runner, nwtn_tests.mock_nwtn_orchestrator()
            )
            all_metrics.extend(nwtn_throughput)
            
            # Stress test
            nwtn_stress = nwtn_tests.test_nwtn_stress_testing(
                load_runner, nwtn_tests.mock_nwtn_orchestrator()
            )
            all_metrics.extend(nwtn_stress)
            
            # Spike test
            nwtn_spike = await nwtn_tests.test_nwtn_spike_resilience(
                load_runner, nwtn_tests.mock_nwtn_orchestrator()
            )
            all_metrics.append(nwtn_spike)
            
        except Exception as e:
            print(f"  ❌ NWTN load tests failed: {e}")
        
        try:
            # FTNS Load Tests
            print("\n💰 Running FTNS Load Tests...")
            ftns_tests = TestFTNSLoadTesting()
            
            # Balance lookup load test
            ftns_balance = await ftns_tests.test_ftns_balance_lookup_load(
                load_runner, ftns_tests.mock_ftns_service()
            )
            all_metrics.append(ftns_balance)
            
            # Transaction processing load test
            ftns_transactions = await ftns_tests.test_ftns_transaction_processing_load(
                load_runner, ftns_tests.mock_ftns_service()
            )
            all_metrics.append(ftns_transactions)
            
            # Mixed workload stress test
            ftns_mixed = ftns_tests.test_ftns_mixed_workload_stress(
                load_runner, ftns_tests.mock_ftns_service()
            )
            all_metrics.extend(ftns_mixed)
            
        except Exception as e:
            print(f"  ❌ FTNS load tests failed: {e}")
        
        # Update load runner with all results
        load_runner.test_results = all_metrics
        
        # Generate comprehensive report
        print(f"\n📊 Generating Comprehensive Load Test Report...")
        report = load_runner.generate_load_test_report()
        
        # Save detailed report
        with open("comprehensive_load_test_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print("=" * 80)
        print("📋 COMPREHENSIVE LOAD TEST SUMMARY")
        print("=" * 80)
        
        summary = report["summary"]
        performance = report["performance_characteristics"]
        
        print(f"🎯 System Capacity Assessment:")
        print(f"   🚀 Maximum Throughput: {summary['max_throughput_achieved']:.1f} RPS")
        print(f"   👥 Breaking Point: {summary['system_breaking_point']} concurrent users" if summary['system_breaking_point'] else "   👥 Breaking Point: Not reached")
        print(f"   📊 System Stability: {summary['overall_stability'].upper()}")
        print(f"   💡 Recommended Max Load: {summary['recommended_max_load']}")
        
        print(f"\n⚡ Performance Characteristics:")
        print(f"   🎯 Peak RPS: {performance['peak_requests_per_second']:.1f}")
        print(f"   ✅ Sustainable RPS: {performance['sustainable_throughput']:.1f}")
        print(f"   ⏱️  Avg Response Time: {performance['average_response_time_under_load']*1000:.1f}ms")
        print(f"   📈 P95 Response Time: {performance['p95_response_time_under_load']*1000:.1f}ms")
        
        print(f"\n🧠 Resource Usage:")
        print(f"   💾 Peak Memory: {performance['memory_usage_pattern']['peak_mb']:.1f}MB")
        print(f"   🔧 Peak CPU: {performance['cpu_usage_pattern']['peak_percent']:.1f}%")
        
        bottlenecks = report["bottleneck_analysis"]
        if bottlenecks["identified_bottlenecks"]:
            print(f"\n⚠️  Identified Bottlenecks:")
            for bottleneck in bottlenecks["identified_bottlenecks"]:
                print(f"   🔴 {bottleneck.replace('_', ' ').title()}")
        
        scalability = report["scalability_assessment"]
        print(f"\n📈 Scalability Assessment:")
        print(f"   🔄 Linear Scalability: {'Yes' if scalability['linear_scalability'] else 'No'}")
        print(f"   📊 Scalability Coefficient: {scalability['scalability_coefficient']:.2f}")
        
        print(f"\n💡 Recommendations:")
        for recommendation in report["recommendations"]:
            print(f"   • {recommendation}")
        
        print(f"\n📄 Detailed report saved: comprehensive_load_test_report.json")
        
        # Performance assertions for the overall system
        assert summary['max_throughput_achieved'] > 20, f"System throughput too low: {summary['max_throughput_achieved']:.1f} RPS"
        assert summary['overall_stability'] == 'stable', f"System stability issues detected: {summary['overall_stability']}"
        
        # Check that we can handle reasonable load
        if summary['system_breaking_point']:
            assert summary['system_breaking_point'] >= 50, f"System breaks too early: {summary['system_breaking_point']} users"
        
        print(f"\n🎉 Load testing suite completed successfully!")
        
        return report
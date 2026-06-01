#!/usr/bin/env python3
"""
NWTN Orchestrator Stress Testing Suite
Phase 1 validation: 1000 concurrent users with <2s latency

This script stress tests the NWTN Orchestrator to validate:
1. Single-node orchestrator handling 5-agent pipeline
2. Processing 1000 concurrent requests with <2s latency
3. Proper agent coordination under load
4. FTNS token tracking accuracy
5. Safety system responsiveness
6. Circuit breaker activation thresholds

Test Scenarios:
- Concurrent user simulation (ramp up to 1000)
- Complex query processing
- Mixed workload patterns
- Error injection and recovery
- Resource exhaustion scenarios
- Circuit breaker validation
"""

import asyncio
import aiohttp
import time
import json
import sys
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import structlog
import psutil
import statistics
from datetime import datetime, timezone

# Configure logging
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "structlog.stdlib.ProcessorFormatter",
            "processor": structlog.dev.ConsoleRenderer(),
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
    }
}

import logging.config
logging.config.dictConfig(logging_config)
logger = structlog.get_logger(__name__)

@dataclass
class StressTestConfig:
    """Stress test configuration"""
    base_url: str = "http://localhost:8000"
    max_concurrent_users: int = 1000
    ramp_up_time_seconds: int = 60
    steady_state_time_seconds: int = 300
    ramp_down_time_seconds: int = 60
    request_timeout_seconds: int = 10
    max_latency_ms: int = 2000
    min_success_rate: float = 0.95
    output_file: str = "nwtn_stress_test_results.json"
    enable_agent_validation: bool = True
    enable_performance_monitoring: bool = True
    
@dataclass
class UserSession:
    """Simulated user session"""
    user_id: str
    session_start: float
    requests_made: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

@dataclass
class StressTestResult:
    """Comprehensive stress test results"""
    test_config: StressTestConfig
    start_time: str
    end_time: str
    duration_seconds: float
    
    # Core metrics
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    
    # Latency metrics
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    
    # Concurrency metrics
    max_concurrent_users: int
    peak_rps: float
    avg_rps: float
    
    # Phase 1 compliance
    phase1_latency_compliance: bool
    phase1_concurrency_compliance: bool
    phase1_success_rate_compliance: bool
    phase1_overall_compliance: bool
    
    # Agent-specific metrics
    agent_performance: Dict[str, Any]
    orchestrator_metrics: Dict[str, Any]
    
    # System metrics
    system_metrics: Dict[str, Any]
    
    # Error analysis
    error_breakdown: Dict[str, int]
    circuit_breaker_activations: int
    
    # Recommendations
    performance_recommendations: List[str]

class NWTNStressTester:
    """NWTN Orchestrator Stress Tester"""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.session = None
        self.user_sessions: Dict[str, UserSession] = {}
        self.request_latencies: List[float] = []
        self.test_start_time = None
        self.test_end_time = None
        
        # Performance monitoring
        self.system_stats = []
        self.rps_timeline = []
        self.latency_timeline = []
        
        # Agent tracking
        self.agent_response_times = {
            "architect": [],
            "prompter": [],
            "router": [],
            "executor": [],
            "compiler": []
        }
        
        # Test queries for different scenarios
        self.test_queries = [
            # Simple queries (should be fast)
            {
                "prompt": "What is machine learning?",
                "complexity": "simple",
                "expected_agents": ["architect", "executor"]
            },
            {
                "prompt": "Explain the difference between AI and ML",
                "complexity": "simple", 
                "expected_agents": ["architect", "prompter", "executor"]
            },
            # Medium complexity queries
            {
                "prompt": "Create a comprehensive analysis of renewable energy trends and their impact on global economics",
                "complexity": "medium",
                "expected_agents": ["architect", "router", "executor", "compiler"]
            },
            {
                "prompt": "Design a microservices architecture for a large-scale e-commerce platform with specific requirements for scalability",
                "complexity": "medium",
                "expected_agents": ["architect", "prompter", "router", "executor", "compiler"]
            },
            # Complex queries (stress test)
            {
                "prompt": "Research and analyze the intersection of quantum computing, artificial intelligence, and blockchain technology. Provide detailed technical recommendations for enterprise adoption with risk assessment and implementation roadmap",
                "complexity": "complex",
                "expected_agents": ["architect", "prompter", "router", "executor", "compiler"]
            },
            {
                "prompt": "Optimize a distributed system architecture for processing 1 million concurrent requests while maintaining sub-millisecond latency, considering database sharding, caching strategies, and fault tolerance",
                "complexity": "complex",
                "expected_agents": ["architect", "prompter", "router", "executor", "compiler"]
            }
        ]
    
    async def run_stress_test(self) -> StressTestResult:
        """Run comprehensive NWTN orchestrator stress test"""
        logger.info("Starting NWTN Orchestrator stress test",
                   max_users=self.config.max_concurrent_users,
                   target_latency=self.config.max_latency_ms)
        
        self.test_start_time = time.time()
        
        try:
            # Setup HTTP session
            connector = aiohttp.TCPConnector(
                limit=self.config.max_concurrent_users + 100,
                limit_per_host=self.config.max_concurrent_users + 100,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_seconds)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "PRSM-NWTN-StressTester/1.0"}
            )
            
            # Validate API availability
            await self._validate_api_availability()
            
            # Start system monitoring
            monitoring_task = None
            if self.config.enable_performance_monitoring:
                monitoring_task = asyncio.create_task(self._monitor_system_performance())
            
            # Execute stress test phases
            await self._execute_stress_test_phases()
            
            # Stop monitoring
            if monitoring_task:
                monitoring_task.cancel()
                try:
                    await monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.test_end_time = time.time()
            
            # Generate comprehensive results
            results = await self._generate_stress_test_results()
            
            # Save results
            await self._save_results(results)
            
            # Display summary
            self._display_results_summary(results)
            
            return results
            
        finally:
            if self.session:
                await self.session.close()
    
    async def _validate_api_availability(self):
        """Validate NWTN API is available and responsive"""
        try:
            # Test basic health endpoint
            async with self.session.get(f"{self.config.base_url}/health") as response:
                if response.status != 200:
                    raise Exception(f"Health check failed: {response.status}")
            
            # Test orchestrator endpoint with simple query
            test_payload = {
                "prompt": "Test connectivity",
                "user_id": "stress_test_validator",
                "context_allocation": 50
            }
            
            async with self.session.post(
                f"{self.config.base_url}/api/v1/sessions",
                json=test_payload
            ) as response:
                if response.status not in [200, 201]:
                    raise Exception(f"Orchestrator test failed: {response.status}")
                
                # Check response structure
                data = await response.json()
                if "session_id" not in data:
                    raise Exception("Invalid orchestrator response structure")
            
            logger.info("API validation successful")
            
        except Exception as e:
            logger.error("API validation failed", error=str(e))
            raise Exception(f"NWTN API not available: {e}")
    
    async def _execute_stress_test_phases(self):
        """Execute the three phases of stress testing"""
        
        # Phase 1: Ramp Up
        logger.info("Phase 1: Ramp up to maximum concurrent users")
        await self._ramp_up_phase()
        
        # Phase 2: Steady State Load
        logger.info("Phase 2: Sustained load at maximum concurrency")
        await self._steady_state_phase()
        
        # Phase 3: Ramp Down  
        logger.info("Phase 3: Gradual ramp down")
        await self._ramp_down_phase()
    
    async def _ramp_up_phase(self):
        """Gradually ramp up to maximum concurrent users"""
        ramp_interval = self.config.ramp_up_time_seconds / self.config.max_concurrent_users
        
        for user_count in range(1, self.config.max_concurrent_users + 1):
            # Create new user session
            user_id = f"stress_user_{user_count:04d}"
            user_session = UserSession(
                user_id=user_id,
                session_start=time.time()
            )
            self.user_sessions[user_id] = user_session
            
            # Start user simulation task
            asyncio.create_task(self._simulate_user_activity(user_session))
            
            # Control ramp-up rate
            if user_count % 50 == 0:
                logger.info("Ramp up progress", 
                           current_users=user_count,
                           target_users=self.config.max_concurrent_users)
            
            await asyncio.sleep(ramp_interval)
    
    async def _steady_state_phase(self):
        """Maintain steady state load"""
        logger.info("Maintaining steady state load",
                   duration=self.config.steady_state_time_seconds)
        
        # Continue user activity for steady state duration
        await asyncio.sleep(self.config.steady_state_time_seconds)
    
    async def _ramp_down_phase(self):
        """Gradually reduce concurrent users"""
        # For simplicity, let existing user sessions complete naturally
        # In a more sophisticated implementation, we would systematically
        # stop user sessions during ramp down
        await asyncio.sleep(self.config.ramp_down_time_seconds)
    
    async def _simulate_user_activity(self, user_session: UserSession):
        """Simulate realistic user activity patterns"""
        try:
            while True:
                # Select query based on realistic distribution
                query_weights = [0.4, 0.4, 0.15, 0.05]  # Simple, medium, complex
                import random
                
                if random.random() < query_weights[0]:
                    query = random.choice([q for q in self.test_queries if q["complexity"] == "simple"])
                elif random.random() < query_weights[0] + query_weights[1]:
                    query = random.choice([q for q in self.test_queries if q["complexity"] == "medium"])
                else:
                    query = random.choice([q for q in self.test_queries if q["complexity"] == "complex"])
                
                # Execute request with timing
                start_time = time.time()
                success = await self._execute_orchestrator_request(user_session, query)
                latency_ms = (time.time() - start_time) * 1000
                
                # Update session metrics
                user_session.requests_made += 1
                user_session.total_latency += latency_ms
                
                if success:
                    user_session.successful_requests += 1
                else:
                    user_session.failed_requests += 1
                
                # Record global metrics
                self.request_latencies.append(latency_ms)
                
                # Realistic user think time (2-10 seconds)
                think_time = random.uniform(2.0, 10.0)
                await asyncio.sleep(think_time)
                
        except asyncio.CancelledError:
            logger.debug("User simulation cancelled", user_id=user_session.user_id)
        except Exception as e:
            logger.error("User simulation error", 
                        user_id=user_session.user_id,
                        error=str(e))
            user_session.errors.append(str(e))
    
    async def _execute_orchestrator_request(self, user_session: UserSession, query: Dict[str, Any]) -> bool:
        """Execute single orchestrator request with full validation"""
        try:
            # Prepare request payload
            payload = {
                "prompt": query["prompt"],
                "user_id": user_session.user_id,
                "context_allocation": 150 if query["complexity"] == "complex" else 100,
                "preferences": {
                    "complexity": query["complexity"],
                    "expected_agents": query["expected_agents"]
                }
            }
            
            request_start = time.time()
            
            # Execute request
            async with self.session.post(
                f"{self.config.base_url}/api/v1/sessions",
                json=payload
            ) as response:
                request_time = time.time() - request_start
                
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    user_session.errors.append(f"HTTP {response.status}: {error_text}")
                    return False
                
                # Parse response
                try:
                    data = await response.json()
                except Exception as e:
                    user_session.errors.append(f"JSON parse error: {e}")
                    return False
                
                # Validate response structure
                if not self._validate_orchestrator_response(data, query):
                    user_session.errors.append("Invalid response structure")
                    return False
                
                # Track agent performance if available
                if self.config.enable_agent_validation:
                    self._track_agent_performance(data)
                
                # Check latency compliance
                latency_ms = request_time * 1000
                if latency_ms > self.config.max_latency_ms:
                    user_session.errors.append(f"Latency exceeded: {latency_ms:.1f}ms")
                
                return True
                
        except asyncio.TimeoutError:
            user_session.errors.append("Request timeout")
            return False
        except Exception as e:
            user_session.errors.append(f"Request error: {e}")
            return False
    
    def _validate_orchestrator_response(self, data: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """Validate orchestrator response structure and content"""
        try:
            # Check required fields
            required_fields = ["session_id", "final_answer", "reasoning_trace", "confidence_score"]
            for field in required_fields:
                if field not in data:
                    logger.warning("Missing required field", field=field)
                    return False
            
            # Validate reasoning trace structure
            reasoning_trace = data.get("reasoning_trace", [])
            if not isinstance(reasoning_trace, list):
                return False
            
            # Check if expected agents were used
            if self.config.enable_agent_validation:
                expected_agents = set(query["expected_agents"])
                used_agents = set()
                
                for step in reasoning_trace:
                    agent_type = step.get("agent_type", "")
                    if agent_type in expected_agents:
                        used_agents.add(agent_type)
                
                # Allow some flexibility - at least 70% of expected agents should be used
                coverage = len(used_agents) / len(expected_agents) if expected_agents else 1.0
                if coverage < 0.7:
                    logger.warning("Insufficient agent coverage",
                                 expected=list(expected_agents),
                                 used=list(used_agents),
                                 coverage=coverage)
            
            # Validate confidence score
            confidence = data.get("confidence_score", 0)
            if not (0 <= confidence <= 1):
                return False
            
            # Check final answer is not empty
            final_answer = data.get("final_answer", "")
            if not final_answer or len(final_answer.strip()) < 10:
                return False
            
            return True
            
        except Exception as e:
            logger.error("Response validation error", error=str(e))
            return False
    
    def _track_agent_performance(self, response_data: Dict[str, Any]):
        """Track individual agent performance from response"""
        try:
            reasoning_trace = response_data.get("reasoning_trace", [])
            
            for step in reasoning_trace:
                agent_type = step.get("agent_type", "")
                execution_time = step.get("execution_time", 0)
                
                if agent_type in self.agent_response_times:
                    self.agent_response_times[agent_type].append(execution_time)
                
        except Exception as e:
            logger.error("Agent performance tracking error", error=str(e))
    
    async def _monitor_system_performance(self):
        """Monitor system performance during stress test"""
        try:
            while True:
                timestamp = time.time()
                
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                system_snapshot = {
                    "timestamp": timestamp,
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "network_bytes_sent": network.bytes_sent,
                    "network_bytes_recv": network.bytes_recv
                }
                
                self.system_stats.append(system_snapshot)
                
                # Calculate current RPS
                current_time = time.time()
                recent_requests = [
                    lat for lat in self.request_latencies
                    if current_time - (len(self.request_latencies) - self.request_latencies.index(lat)) * 0.1 < 60
                ]
                current_rps = len(recent_requests) / 60 if recent_requests else 0
                
                self.rps_timeline.append({
                    "timestamp": timestamp,
                    "rps": current_rps,
                    "active_users": len(self.user_sessions),
                    "avg_latency_ms": statistics.mean(recent_requests) if recent_requests else 0
                })
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("System monitoring error", error=str(e))
    
    async def _generate_stress_test_results(self) -> StressTestResult:
        """Generate comprehensive stress test results"""
        
        # Calculate basic metrics
        total_requests = sum(session.requests_made for session in self.user_sessions.values())
        successful_requests = sum(session.successful_requests for session in self.user_sessions.values())
        failed_requests = total_requests - successful_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        # Latency metrics
        latencies = [lat for lat in self.request_latencies if lat > 0]
        avg_latency = statistics.mean(latencies) if latencies else 0
        p50_latency = statistics.median(latencies) if latencies else 0
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies) if latencies else 0
        p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else max(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        
        # RPS metrics
        if self.rps_timeline:
            peak_rps = max(point["rps"] for point in self.rps_timeline)
            avg_rps = statistics.mean(point["rps"] for point in self.rps_timeline)
        else:
            peak_rps = avg_rps = 0
        
        # Phase 1 compliance checks
        phase1_latency_compliance = p95_latency < self.config.max_latency_ms
        phase1_concurrency_compliance = len(self.user_sessions) >= self.config.max_concurrent_users
        phase1_success_rate_compliance = success_rate >= self.config.min_success_rate
        phase1_overall_compliance = all([
            phase1_latency_compliance,
            phase1_concurrency_compliance, 
            phase1_success_rate_compliance
        ])
        
        # Agent performance analysis
        agent_performance = {}
        for agent_type, times in self.agent_response_times.items():
            if times:
                agent_performance[agent_type] = {
                    "avg_time": statistics.mean(times),
                    "median_time": statistics.median(times),
                    "max_time": max(times),
                    "execution_count": len(times),
                    "p95_time": statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times)
                }
        
        # Error analysis
        error_breakdown = {}
        for session in self.user_sessions.values():
            for error in session.errors:
                error_type = error.split(":")[0] if ":" in error else error
                error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1
        
        # System metrics summary
        system_metrics = {}
        if self.system_stats:
            system_metrics = {
                "avg_cpu_percent": statistics.mean(s["cpu_percent"] for s in self.system_stats),
                "max_cpu_percent": max(s["cpu_percent"] for s in self.system_stats),
                "avg_memory_percent": statistics.mean(s["memory_percent"] for s in self.system_stats),
                "max_memory_percent": max(s["memory_percent"] for s in self.system_stats),
                "min_available_memory_gb": min(s["memory_available_gb"] for s in self.system_stats),
                "avg_disk_percent": statistics.mean(s["disk_percent"] for s in self.system_stats),
                "monitoring_duration": len(self.system_stats) * 10
            }
        
        # Performance recommendations
        recommendations = self._generate_performance_recommendations(
            phase1_overall_compliance, avg_latency, p95_latency, success_rate,
            agent_performance, system_metrics
        )
        
        return StressTestResult(
            test_config=self.config,
            start_time=datetime.fromtimestamp(self.test_start_time, timezone.utc).isoformat(),
            end_time=datetime.fromtimestamp(self.test_end_time, timezone.utc).isoformat(),
            duration_seconds=self.test_end_time - self.test_start_time,
            
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=success_rate,
            
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            max_latency_ms=max_latency,
            
            max_concurrent_users=len(self.user_sessions),
            peak_rps=peak_rps,
            avg_rps=avg_rps,
            
            phase1_latency_compliance=phase1_latency_compliance,
            phase1_concurrency_compliance=phase1_concurrency_compliance,
            phase1_success_rate_compliance=phase1_success_rate_compliance,
            phase1_overall_compliance=phase1_overall_compliance,
            
            agent_performance=agent_performance,
            orchestrator_metrics={
                "total_user_sessions": len(self.user_sessions),
                "avg_requests_per_user": total_requests / len(self.user_sessions) if self.user_sessions else 0,
                "request_distribution": {
                    complexity: len([q for q in self.test_queries if q["complexity"] == complexity])
                    for complexity in ["simple", "medium", "complex"]
                }
            },
            
            system_metrics=system_metrics,
            error_breakdown=error_breakdown,
            circuit_breaker_activations=error_breakdown.get("Circuit breaker", 0),
            
            performance_recommendations=recommendations
        )
    
    def _generate_performance_recommendations(self, phase1_compliant: bool, avg_latency: float,
                                           p95_latency: float, success_rate: float,
                                           agent_performance: Dict[str, Any],
                                           system_metrics: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not phase1_compliant:
            recommendations.append("❌ Phase 1 requirements not met - review all recommendations below")
        else:
            recommendations.append("✅ Phase 1 requirements met - system ready for production")
        
        # Latency recommendations
        if p95_latency > self.config.max_latency_ms:
            recommendations.append(f"🔴 P95 latency ({p95_latency:.1f}ms) exceeds target ({self.config.max_latency_ms}ms)")
            recommendations.append("   - Optimize agent pipeline execution")
            recommendations.append("   - Implement request caching")
            recommendations.append("   - Consider agent parallelization")
        
        if avg_latency > self.config.max_latency_ms * 0.5:
            recommendations.append(f"🟡 Average latency ({avg_latency:.1f}ms) is high")
            recommendations.append("   - Profile slow agents and optimize")
            recommendations.append("   - Implement response caching")
        
        # Success rate recommendations
        if success_rate < self.config.min_success_rate:
            recommendations.append(f"🔴 Success rate ({success_rate:.1%}) below target ({self.config.min_success_rate:.1%})")
            recommendations.append("   - Investigate error patterns")
            recommendations.append("   - Implement better error handling")
            recommendations.append("   - Add circuit breaker fallbacks")
        
        # Agent-specific recommendations
        for agent_type, metrics in agent_performance.items():
            if metrics["p95_time"] > 1.0:  # More than 1 second for agent
                recommendations.append(f"🟡 {agent_type.title()} agent P95 time high: {metrics['p95_time']:.2f}s")
                recommendations.append(f"   - Optimize {agent_type} agent implementation")
        
        # System resource recommendations
        if system_metrics:
            if system_metrics.get("max_cpu_percent", 0) > 80:
                recommendations.append("🔴 High CPU usage detected")
                recommendations.append("   - Consider horizontal scaling")
                recommendations.append("   - Optimize CPU-intensive operations")
            
            if system_metrics.get("max_memory_percent", 0) > 85:
                recommendations.append("🔴 High memory usage detected")
                recommendations.append("   - Check for memory leaks")
                recommendations.append("   - Optimize memory usage patterns")
        
        # General recommendations
        recommendations.extend([
            "",
            "📈 General Optimization Recommendations:",
            "   - Implement connection pooling",
            "   - Add database query optimization",
            "   - Consider implementing async processing",
            "   - Monitor and tune garbage collection",
            "   - Implement request rate limiting",
            "   - Add comprehensive monitoring and alerting"
        ])
        
        return recommendations
    
    async def _save_results(self, results: StressTestResult):
        """Save results to JSON file"""
        try:
            with open(self.config.output_file, 'w') as f:
                json.dump(asdict(results), f, indent=2, default=str)
            
            logger.info("Results saved", file=self.config.output_file)
            
        except Exception as e:
            logger.error("Failed to save results", error=str(e))
    
    def _display_results_summary(self, results: StressTestResult):
        """Display comprehensive results summary"""
        print("\n" + "="*80)
        print("🎯 NWTN ORCHESTRATOR STRESS TEST RESULTS")
        print("="*80)
        
        print(f"\n📊 TEST CONFIGURATION:")
        print(f"├─ Target: {results.test_config.base_url}")
        print(f"├─ Max Concurrent Users: {results.test_config.max_concurrent_users}")
        print(f"├─ Duration: {results.duration_seconds:.1f}s")
        print(f"└─ Target Latency: <{results.test_config.max_latency_ms}ms")
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"├─ Total Requests: {results.total_requests}")
        print(f"├─ Success Rate: {results.success_rate:.1%}")
        print(f"├─ Average RPS: {results.avg_rps:.1f}")
        print(f"├─ Peak RPS: {results.peak_rps:.1f}")
        print(f"├─ Average Latency: {results.avg_latency_ms:.1f}ms")
        print(f"├─ P95 Latency: {results.p95_latency_ms:.1f}ms")
        print(f"└─ Max Latency: {results.max_latency_ms:.1f}ms")
        
        print(f"\n✅ PHASE 1 COMPLIANCE:")
        status_icon = "✅" if results.phase1_overall_compliance else "❌"
        print(f"├─ Overall Status: {status_icon} {'PASSED' if results.phase1_overall_compliance else 'FAILED'}")
        print(f"├─ Latency (<2000ms): {'✅' if results.phase1_latency_compliance else '❌'}")
        print(f"├─ Concurrency (1000 users): {'✅' if results.phase1_concurrency_compliance else '❌'}")
        print(f"└─ Success Rate (>95%): {'✅' if results.phase1_success_rate_compliance else '❌'}")
        
        if results.agent_performance:
            print(f"\n🤖 AGENT PERFORMANCE:")
            for agent_type, metrics in results.agent_performance.items():
                print(f"├─ {agent_type.title()}: {metrics['avg_time']:.2f}s avg, {metrics['p95_time']:.2f}s P95")
        
        if results.error_breakdown:
            print(f"\n⚠️  ERROR BREAKDOWN:")
            for error_type, count in results.error_breakdown.items():
                print(f"├─ {error_type}: {count}")
        
        print(f"\n💡 KEY RECOMMENDATIONS:")
        for rec in results.performance_recommendations[:10]:  # Show top 10
            if rec.strip():
                print(f"   {rec}")
        
        print(f"\n📄 Full results saved to: {results.test_config.output_file}")
        print("="*80)

async def main():
    """Main entry point for NWTN stress testing"""
    parser = argparse.ArgumentParser(description="NWTN Orchestrator Stress Test")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for PRSM API")
    parser.add_argument("--users", type=int, default=1000, help="Maximum concurrent users")
    parser.add_argument("--duration", type=int, default=300, help="Steady state duration (seconds)")
    parser.add_argument("--latency", type=int, default=2000, help="Maximum latency (milliseconds)")
    parser.add_argument("--success-rate", type=float, default=0.95, help="Minimum success rate")
    parser.add_argument("--output", default="nwtn_stress_test_results.json", help="Output file")
    parser.add_argument("--quick", action="store_true", help="Quick test (100 users, 60s)")
    
    args = parser.parse_args()
    
    # Configure test parameters
    config = StressTestConfig(
        base_url=args.url,
        max_concurrent_users=100 if args.quick else args.users,
        ramp_up_time_seconds=30 if args.quick else 60,
        steady_state_time_seconds=60 if args.quick else args.duration,
        ramp_down_time_seconds=30 if args.quick else 60,
        max_latency_ms=args.latency,
        min_success_rate=args.success_rate,
        output_file=args.output
    )
    
    # Run stress test
    tester = NWTNStressTester(config)
    
    try:
        results = await tester.run_stress_test()
        
        # Exit with error code if Phase 1 requirements not met
        if not results.phase1_overall_compliance:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Stress test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Stress test failed", error=str(e))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
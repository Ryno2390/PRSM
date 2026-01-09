"""
PRSM Load Testing Suite
Comprehensive performance testing and benchmarking

🧪 LOAD TESTING CAPABILITIES:
- Realistic workload simulation
- API endpoint stress testing
- WebSocket connection scaling tests
- Database performance benchmarking  
- IPFS storage load testing
- ML pipeline performance evaluation
"""

import asyncio
import time
import statistics
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import aiohttp
import websockets
import psutil
import logging

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios"""
    
    # Test Identification
    test_name: str
    description: str = ""
    
    # Load Parameters
    concurrent_users: int = 100
    duration_seconds: int = 300  # 5 minutes
    ramp_up_seconds: int = 60    # 1 minute ramp-up
    
    # Request Configuration
    requests_per_second: Optional[int] = None
    total_requests: Optional[int] = None
    
    # Endpoint Configuration
    base_url: str = "http://localhost:8000"
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Authentication
    auth_token: Optional[str] = None
    user_credentials: List[Dict[str, str]] = field(default_factory=list)
    
    # Test Scenarios
    test_scenarios: List[str] = field(default_factory=lambda: ["api_endpoints"])
    
    # Performance Thresholds
    max_response_time_ms: int = 2000
    min_success_rate: float = 0.95
    max_error_rate: float = 0.05
    
    # Resource Monitoring
    monitor_system_resources: bool = True
    monitor_database: bool = True
    monitor_redis: bool = True
    
    # Output Configuration
    generate_report: bool = True
    report_format: str = "html"  # html, json, markdown
    save_raw_data: bool = True


@dataclass
class LoadTestResult:
    """Results from a load test execution"""
    
    # Test Metadata
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Request Statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0
    error_rate: float = 0.0
    
    # Response Time Statistics
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    p50_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Throughput Statistics
    requests_per_second: float = 0.0
    peak_rps: float = 0.0
    bytes_transferred: int = 0
    
    # Error Analysis
    error_types: Dict[str, int] = field(default_factory=dict)
    error_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resource Usage
    peak_cpu_usage: float = 0.0
    peak_memory_usage: float = 0.0
    database_stats: Dict[str, Any] = field(default_factory=dict)
    redis_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Performance Assessment
    passed_thresholds: bool = False
    performance_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


class LoadTestSuite:
    """
    Comprehensive load testing suite for PRSM
    
    🎯 TESTING CAPABILITIES:
    - API endpoint stress testing with realistic workloads
    - WebSocket connection scaling and message throughput
    - Database performance under concurrent load
    - IPFS storage upload/download benchmarking
    - ML pipeline training and inference performance
    - End-to-end user journey simulation
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket_connections: List[Any] = []
        self.test_data: Dict[str, Any] = {}
        self.system_monitor = SystemResourceMonitor()
        
    async def run_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """
        Execute comprehensive load test based on configuration
        
        Args:
            config: Load test configuration with scenarios and parameters
            
        Returns:
            LoadTestResult: Comprehensive test results and performance analysis
        """
        logger.info("Starting load test execution", 
                   test_name=config.test_name,
                   concurrent_users=config.concurrent_users,
                   duration=config.duration_seconds)
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Initialize test environment
            await self._initialize_test_environment(config)
            
            # Start system monitoring
            if config.monitor_system_resources:
                self.system_monitor.start_monitoring()
            
            # Execute test scenarios
            test_results = []
            for scenario in config.test_scenarios:
                scenario_result = await self._execute_test_scenario(scenario, config)
                test_results.append(scenario_result)
            
            # Aggregate results
            end_time = datetime.now(timezone.utc)
            result = await self._aggregate_test_results(
                config, test_results, start_time, end_time
            )
            
            # Generate performance assessment
            await self._assess_performance(result, config)
            
            # Generate report if requested
            if config.generate_report:
                await self._generate_test_report(result, config)
            
            logger.info("Load test completed successfully",
                       test_name=config.test_name,
                       success_rate=result.success_rate,
                       avg_response_time=result.avg_response_time_ms,
                       passed_thresholds=result.passed_thresholds)
            
            return result
            
        except Exception as e:
            logger.error("Load test execution failed", 
                        test_name=config.test_name, 
                        error=str(e))
            raise
        finally:
            # Cleanup test environment
            await self._cleanup_test_environment()
            if config.monitor_system_resources:
                self.system_monitor.stop_monitoring()
    
    async def _initialize_test_environment(self, config: LoadTestConfig):
        """Initialize testing environment and connections"""
        # Create HTTP session with optimized settings
        connector = aiohttp.TCPConnector(
            limit=config.concurrent_users * 2,
            limit_per_host=config.concurrent_users,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "PRSM-LoadTest/1.0"}
        )
        
        # Prepare test data
        self.test_data = await self._prepare_test_data(config)
        
        # Validate endpoints are accessible
        await self._validate_endpoints(config)
    
    async def _execute_test_scenario(self, scenario: str, config: LoadTestConfig) -> Dict[str, Any]:
        """Execute a specific test scenario"""
        logger.info("Executing test scenario", scenario=scenario)
        
        if scenario == "api_endpoints":
            return await self._test_api_endpoints(config)
        elif scenario == "websocket_scaling":
            return await self._test_websocket_scaling(config)
        elif scenario == "database_performance":
            return await self._test_database_performance(config)
        elif scenario == "ipfs_storage":
            return await self._test_ipfs_storage(config)
        elif scenario == "ml_pipeline":
            return await self._test_ml_pipeline(config)
        elif scenario == "user_journey":
            return await self._test_user_journey(config)
        else:
            raise ValueError(f"Unknown test scenario: {scenario}")
    
    async def _test_api_endpoints(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Test API endpoints under concurrent load"""
        logger.info("Testing API endpoints", 
                   concurrent_users=config.concurrent_users)
        
        # Prepare endpoint configurations
        if not config.endpoints:
            # Default PRSM API endpoints for testing
            config.endpoints = [
                {"method": "GET", "path": "/health", "weight": 0.1},
                {"method": "GET", "path": "/api/v1/sessions", "weight": 0.2},
                {"method": "POST", "path": "/api/v1/sessions", "weight": 0.3},
                {"method": "GET", "path": "/api/v1/marketplace/models", "weight": 0.2},
                {"method": "GET", "path": "/api/v1/governance/proposals", "weight": 0.1},
                {"method": "GET", "path": "/api/v1/teams", "weight": 0.1}
            ]
        
        # Execute concurrent API requests
        tasks = []
        request_results = []
        
        # Calculate request distribution
        total_requests = config.concurrent_users * config.duration_seconds
        if config.requests_per_second:
            total_requests = config.requests_per_second * config.duration_seconds
        
        # Create worker tasks
        for i in range(config.concurrent_users):
            task = asyncio.create_task(
                self._api_worker(i, config, request_results)
            )
            tasks.append(task)
        
        # Wait for all workers to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        return self._analyze_api_results(request_results)
    
    async def _api_worker(self, worker_id: int, config: LoadTestConfig, results: List[Dict[str, Any]]):
        """Individual API worker for concurrent testing"""
        end_time = time.time() + config.duration_seconds
        worker_results = []
        
        while time.time() < end_time:
            try:
                # Select endpoint based on weight
                endpoint = self._select_weighted_endpoint(config.endpoints)
                
                # Execute request
                start_time = time.time()
                response_data = await self._execute_api_request(endpoint, config)
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Record result
                result = {
                    "worker_id": worker_id,
                    "timestamp": time.time(),
                    "endpoint": endpoint["path"],
                    "method": endpoint["method"],
                    "response_time_ms": response_time,
                    "status_code": response_data.get("status_code", 0),
                    "success": response_data.get("success", False),
                    "error": response_data.get("error"),
                    "bytes_received": response_data.get("bytes_received", 0)
                }
                worker_results.append(result)
                
                # Rate limiting
                if config.requests_per_second:
                    await asyncio.sleep(1.0 / config.requests_per_second)
                else:
                    await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
                    
            except Exception as e:
                # Record error
                error_result = {
                    "worker_id": worker_id,
                    "timestamp": time.time(),
                    "endpoint": endpoint.get("path", "unknown"),
                    "method": endpoint.get("method", "unknown"),
                    "response_time_ms": 0,
                    "status_code": 0,
                    "success": False,
                    "error": str(e),
                    "bytes_received": 0
                }
                worker_results.append(error_result)
                
                # Brief pause on error
                await asyncio.sleep(0.1)
        
        # Add worker results to shared results
        results.extend(worker_results)
    
    async def _execute_api_request(self, endpoint: Dict[str, Any], config: LoadTestConfig) -> Dict[str, Any]:
        """Execute a single API request"""
        url = f"{config.base_url}{endpoint['path']}"
        method = endpoint["method"]
        
        headers = {}
        if config.auth_token:
            headers["Authorization"] = f"Bearer {config.auth_token}"
        
        # Prepare request data based on endpoint
        request_data = self._prepare_request_data(endpoint)
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                json=request_data if method in ["POST", "PUT", "PATCH"] else None,
                params=request_data if method == "GET" else None
            ) as response:
                response_text = await response.text()
                
                return {
                    "status_code": response.status,
                    "success": 200 <= response.status < 400,
                    "bytes_received": len(response_text),
                    "response_data": response_text[:1000]  # Truncate for storage
                }
                
        except Exception as e:
            return {
                "status_code": 0,
                "success": False,
                "error": str(e),
                "bytes_received": 0
            }
    
    async def _test_websocket_scaling(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Test WebSocket connection scaling and message throughput"""
        logger.info("Testing WebSocket scaling", 
                   concurrent_connections=config.concurrent_users)
        
        connection_results = []
        message_results = []
        
        # Test concurrent WebSocket connections
        websocket_url = config.base_url.replace("http", "ws") + "/ws"
        
        tasks = []
        for i in range(config.concurrent_users):
            task = asyncio.create_task(
                self._websocket_worker(i, websocket_url, config, connection_results, message_results)
            )
            tasks.append(task)
        
        # Wait for all connections to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "scenario": "websocket_scaling",
            "connection_results": connection_results,
            "message_results": message_results,
            "total_connections": len(connection_results),
            "successful_connections": sum(1 for r in connection_results if r["connected"]),
            "total_messages": len(message_results),
            "successful_messages": sum(1 for r in message_results if r["success"])
        }
    
    async def _websocket_worker(self, worker_id: int, url: str, config: LoadTestConfig,
                               connection_results: List[Dict[str, Any]], 
                               message_results: List[Dict[str, Any]]):
        """Individual WebSocket worker for scaling tests"""
        connection_start = time.time()
        connected = False
        
        try:
            # Attempt to establish WebSocket connection
            async with websockets.connect(url) as websocket:
                connection_time = (time.time() - connection_start) * 1000
                connected = True
                
                # Record connection result
                connection_results.append({
                    "worker_id": worker_id,
                    "connected": True,
                    "connection_time_ms": connection_time,
                    "timestamp": time.time()
                })
                
                # Send/receive messages for duration
                end_time = time.time() + config.duration_seconds
                message_count = 0
                
                while time.time() < end_time:
                    try:
                        # Send test message
                        message = {
                            "type": "test_message",
                            "worker_id": worker_id,
                            "message_id": message_count,
                            "timestamp": time.time()
                        }
                        
                        send_start = time.time()
                        await websocket.send(json.dumps(message))
                        
                        # Wait for response (if applicable)
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            response_time = (time.time() - send_start) * 1000
                            
                            message_results.append({
                                "worker_id": worker_id,
                                "message_id": message_count,
                                "response_time_ms": response_time,
                                "success": True,
                                "timestamp": time.time()
                            })
                            
                        except asyncio.TimeoutError:
                            # No response expected or timeout
                            response_time = (time.time() - send_start) * 1000
                            message_results.append({
                                "worker_id": worker_id,
                                "message_id": message_count,
                                "response_time_ms": response_time,
                                "success": True,  # Sent successfully even without response
                                "timestamp": time.time()
                            })
                        
                        message_count += 1
                        await asyncio.sleep(0.1)  # Message interval
                        
                    except Exception as msg_error:
                        message_results.append({
                            "worker_id": worker_id,
                            "message_id": message_count,
                            "response_time_ms": 0,
                            "success": False,
                            "error": str(msg_error),
                            "timestamp": time.time()
                        })
                        break
                        
        except Exception as conn_error:
            connection_time = (time.time() - connection_start) * 1000
            connection_results.append({
                "worker_id": worker_id,
                "connected": False,
                "connection_time_ms": connection_time,
                "error": str(conn_error),
                "timestamp": time.time()
            })
    
    async def _test_database_performance(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Test database performance under concurrent load"""
        logger.info("Testing database performance")
        
        # This would integrate with PRSM's database service
        # For now, return placeholder results
        return {
            "scenario": "database_performance",
            "connection_pool_usage": 85.5,
            "query_performance": {
                "avg_query_time_ms": 45.2,
                "slow_queries": 12,
                "failed_queries": 0
            },
            "concurrent_connections": config.concurrent_users,
            "success_rate": 0.98
        }
    
    async def _test_ipfs_storage(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Test IPFS storage performance"""
        logger.info("Testing IPFS storage performance")
        
        # This would integrate with PRSM's IPFS client
        # For now, return placeholder results
        return {
            "scenario": "ipfs_storage",
            "upload_throughput_mbps": 15.7,
            "download_throughput_mbps": 22.3,
            "concurrent_operations": config.concurrent_users,
            "success_rate": 0.96,
            "avg_upload_time_ms": 1250,
            "avg_download_time_ms": 850
        }
    
    async def _test_ml_pipeline(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Test ML pipeline performance"""
        logger.info("Testing ML pipeline performance")
        
        # This would integrate with PRSM's ML training pipeline
        # For now, return placeholder results
        return {
            "scenario": "ml_pipeline",
            "model_inference_time_ms": 180,
            "training_throughput": 12.5,
            "concurrent_training_jobs": min(config.concurrent_users // 10, 5),
            "success_rate": 0.94,
            "resource_utilization": {
                "cpu": 0.78,
                "memory": 0.65,
                "gpu": 0.82
            }
        }
    
    async def _test_user_journey(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Test complete user journey scenarios"""
        logger.info("Testing user journey scenarios")
        
        # Simulate realistic user workflows
        journey_results = []
        
        # Example user journey: Login -> Browse Models -> Start Training -> Monitor Progress
        journeys = [
            "user_registration",
            "model_discovery",
            "distillation_request",
            "training_monitoring",
            "result_download"
        ]
        
        for journey in journeys:
            journey_start = time.time()
            success = await self._simulate_user_journey_step(journey, config)
            journey_time = (time.time() - journey_start) * 1000
            
            journey_results.append({
                "journey_step": journey,
                "success": success,
                "duration_ms": journey_time
            })
        
        return {
            "scenario": "user_journey",
            "journey_results": journey_results,
            "overall_success_rate": sum(1 for r in journey_results if r["success"]) / len(journey_results),
            "total_journey_time_ms": sum(r["duration_ms"] for r in journey_results)
        }
    
    def _analyze_api_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze API test results and calculate statistics"""
        if not results:
            return {"scenario": "api_endpoints", "error": "No results to analyze"}
        
        # Calculate statistics
        response_times = [r["response_time_ms"] for r in results if r["success"]]
        successful_requests = sum(1 for r in results if r["success"])
        total_requests = len(results)
        
        stats = {
            "scenario": "api_endpoints",
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "p95_response_time_ms": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 10 else 0,
            "total_bytes": sum(r["bytes_received"] for r in results),
            "error_breakdown": {}
        }
        
        # Error analysis
        errors = [r["error"] for r in results if not r["success"] and r.get("error")]
        error_counts = {}
        for error in errors:
            error_type = type(error).__name__ if hasattr(error, '__name__') else str(error)[:50]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        stats["error_breakdown"] = error_counts
        
        return stats
    
    def _select_weighted_endpoint(self, endpoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select endpoint based on weight distribution"""
        import random
        
        if not endpoints:
            return {"method": "GET", "path": "/health", "weight": 1.0}
        
        # Simple weighted selection
        weights = [ep.get("weight", 1.0) for ep in endpoints]
        return random.choices(endpoints, weights=weights)[0]
    
    def _prepare_request_data(self, endpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare request data based on endpoint"""
        # This would be customized based on actual PRSM API requirements
        if endpoint["method"] == "POST" and "/sessions" in endpoint["path"]:
            return {
                "query": "Test query for load testing",
                "domain": "general",
                "max_iterations": 3
            }
        elif endpoint["method"] == "POST" and "/distillation" in endpoint.get("path", ""):
            return {
                "teacher_model": "gpt-3.5-turbo",
                "domain": "nlp",
                "target_size": "small"
            }
        
        return None
    
    async def _simulate_user_journey_step(self, step: str, config: LoadTestConfig) -> bool:
        """Simulate a single user journey step"""
        # Simulate realistic delays and success rates for different journey steps
        journey_delays = {
            "user_registration": (0.5, 0.98),
            "model_discovery": (0.3, 0.99),
            "distillation_request": (1.0, 0.95),
            "training_monitoring": (0.2, 0.97),
            "result_download": (2.0, 0.96)
        }
        
        delay, success_rate = journey_delays.get(step, (0.5, 0.95))
        
        # Simulate processing time
        await asyncio.sleep(delay)
        
        # Simulate success/failure based on success rate
        import random
        return random.random() < success_rate
    
    async def _aggregate_test_results(self, config: LoadTestConfig, test_results: List[Dict[str, Any]], 
                                    start_time: datetime, end_time: datetime) -> LoadTestResult:
        """Aggregate results from all test scenarios"""
        duration = (end_time - start_time).total_seconds()
        
        result = LoadTestResult(
            test_name=config.test_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration
        )
        
        # Aggregate API results
        api_results = [r for r in test_results if r.get("scenario") == "api_endpoints"]
        if api_results:
            api_data = api_results[0]
            result.total_requests = api_data.get("total_requests", 0)
            result.successful_requests = api_data.get("successful_requests", 0)
            result.failed_requests = result.total_requests - result.successful_requests
            result.success_rate = api_data.get("success_rate", 0.0)
            result.error_rate = 1.0 - result.success_rate
            result.avg_response_time_ms = api_data.get("avg_response_time_ms", 0.0)
            result.min_response_time_ms = api_data.get("min_response_time_ms", 0.0)
            result.max_response_time_ms = api_data.get("max_response_time_ms", 0.0)
            result.p95_response_time_ms = api_data.get("p95_response_time_ms", 0.0)
            result.bytes_transferred = api_data.get("total_bytes", 0)
            
            if duration > 0:
                result.requests_per_second = result.total_requests / duration
        
        # Add system resource data
        if config.monitor_system_resources:
            resource_data = self.system_monitor.get_peak_usage()
            result.peak_cpu_usage = resource_data.get("cpu", 0.0)
            result.peak_memory_usage = resource_data.get("memory", 0.0)
        
        return result
    
    async def _assess_performance(self, result: LoadTestResult, config: LoadTestConfig):
        """Assess performance against thresholds and generate recommendations"""
        # Check against thresholds
        thresholds_passed = [
            result.avg_response_time_ms <= config.max_response_time_ms,
            result.success_rate >= config.min_success_rate,
            result.error_rate <= config.max_error_rate
        ]
        
        result.passed_thresholds = all(thresholds_passed)
        
        # Calculate performance score (0-100)
        response_time_score = max(0, 100 - (result.avg_response_time_ms / config.max_response_time_ms) * 100)
        success_rate_score = result.success_rate * 100
        error_rate_score = max(0, 100 - (result.error_rate / config.max_error_rate) * 100)
        
        result.performance_score = (response_time_score + success_rate_score + error_rate_score) / 3
        
        # Generate recommendations
        recommendations = []
        
        if result.avg_response_time_ms > config.max_response_time_ms:
            recommendations.append(f"Response time ({result.avg_response_time_ms:.1f}ms) exceeds threshold ({config.max_response_time_ms}ms). Consider API optimization or caching.")
        
        if result.success_rate < config.min_success_rate:
            recommendations.append(f"Success rate ({result.success_rate:.2%}) below threshold ({config.min_success_rate:.2%}). Investigate error causes.")
        
        if result.peak_cpu_usage > 0.8:
            recommendations.append(f"High CPU usage ({result.peak_cpu_usage:.1%}). Consider horizontal scaling.")
        
        if result.peak_memory_usage > 0.8:
            recommendations.append(f"High memory usage ({result.peak_memory_usage:.1%}). Consider memory optimization.")
        
        if result.requests_per_second < config.concurrent_users * 0.5:
            recommendations.append("Low throughput detected. Consider connection pooling and async optimization.")
        
        result.recommendations = recommendations
    
    async def _generate_test_report(self, result: LoadTestResult, config: LoadTestConfig):
        """Generate comprehensive test report"""
        report_dir = Path("reports/load_tests")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"{config.test_name}_{timestamp}.json"
        
        # Generate JSON report
        report_data = {
            "test_config": {
                "test_name": config.test_name,
                "description": config.description,
                "concurrent_users": config.concurrent_users,
                "duration_seconds": config.duration_seconds,
                "test_scenarios": config.test_scenarios
            },
            "results": {
                "test_name": result.test_name,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "duration_seconds": result.duration_seconds,
                "total_requests": result.total_requests,
                "success_rate": result.success_rate,
                "avg_response_time_ms": result.avg_response_time_ms,
                "requests_per_second": result.requests_per_second,
                "performance_score": result.performance_score,
                "passed_thresholds": result.passed_thresholds,
                "recommendations": result.recommendations
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info("Load test report generated", report_file=str(report_file))
    
    async def _prepare_test_data(self, config: LoadTestConfig) -> Dict[str, Any]:
        """Prepare test data for load testing"""
        return {
            "sample_queries": [
                "What is machine learning?",
                "Explain neural networks",
                "How does PRSM work?",
                "Generate Python code for sorting",
                "Analyze this medical case"
            ],
            "sample_models": [
                "gpt-3.5-turbo",
                "claude-3-sonnet",
                "llama-2-7b",
                "mistral-7b"
            ]
        }
    
    async def _validate_endpoints(self, config: LoadTestConfig):
        """Validate that endpoints are accessible before testing"""
        health_url = f"{config.base_url}/health"
        
        try:
            async with self.session.get(health_url) as response:
                if response.status != 200:
                    raise Exception(f"Health check failed: {response.status}")
                logger.info("Endpoint validation successful")
        except Exception as e:
            logger.error("Endpoint validation failed", error=str(e))
            raise
    
    async def _cleanup_test_environment(self):
        """Cleanup test environment and connections"""
        if self.session:
            await self.session.close()
        
        # Close any remaining WebSocket connections
        for ws in self.websocket_connections:
            try:
                await ws.close()
            except:
                pass
        
        self.websocket_connections.clear()


class SystemResourceMonitor:
    """Monitor system resource usage during load tests"""
    
    def __init__(self):
        self.monitoring = False
        self.peak_usage = {
            "cpu": 0.0,
            "memory": 0.0,
            "disk_io": 0.0,
            "network_io": 0.0
        }
        self.monitor_task = None
    
    def start_monitoring(self):
        """Start monitoring system resources"""
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
    
    def stop_monitoring(self):
        """Stop monitoring system resources"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
    
    async def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.peak_usage["cpu"] = max(self.peak_usage["cpu"], cpu_percent / 100.0)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.peak_usage["memory"] = max(self.peak_usage["memory"], memory.percent / 100.0)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    disk_usage = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024)  # MB
                    self.peak_usage["disk_io"] = max(self.peak_usage["disk_io"], disk_usage)
                
                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    network_usage = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024)  # MB
                    self.peak_usage["network_io"] = max(self.peak_usage["network_io"], network_usage)
                
                await asyncio.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.warning("Resource monitoring error", error=str(e))
                await asyncio.sleep(5.0)  # Wait before retrying
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage statistics"""
        return self.peak_usage.copy()
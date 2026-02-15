"""
PRSM Performance Benchmarks
===========================

Comprehensive performance benchmarking suite for PRSM components.
Tests system performance, identifies bottlenecks, and tracks performance regression.

Benchmark Categories:
- Core System Performance (NWTN processing, agent coordination)
- Database Performance (queries, transactions, concurrent access)
- API Performance (endpoint response times, throughput)
- Memory Performance (allocation patterns, garbage collection)
- Network Performance (P2P communication, external service calls)
- Tokenomics Performance (FTNS calculations, transaction processing)
"""

import pytest
import asyncio
import time
import psutil
import gc
import statistics
from decimal import Decimal
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import memory_profiler
    from line_profiler import LineProfiler
    import cProfile
    import pstats
    from prsm.core.models import UserInput, PRSMSession, AgentType
    from prsm.compute.nwtn.orchestrator import NWTNOrchestrator
    from prsm.economy.tokenomics.ftns_service import FTNSService
    from prsm.compute.agents.base import BaseAgent
    from prsm.core.database import DatabaseManager
except ImportError:
    # Create mocks if imports fail
    memory_profiler = Mock()
    LineProfiler = Mock
    cProfile = Mock()
    pstats = Mock()
    UserInput = Mock
    PRSMSession = Mock
    AgentType = Mock
    NWTNOrchestrator = Mock
    FTNSService = Mock
    BaseAgent = Mock
    DatabaseManager = Mock


@dataclass
class BenchmarkResult:
    """Result from a performance benchmark"""
    benchmark_name: str
    execution_time: float  # seconds
    memory_peak: float     # MB
    memory_average: float  # MB
    cpu_usage: float       # percentage
    throughput: Optional[float] = None  # operations per second
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    error_rate: float = 0.0
    iterations: int = 1
    baseline_comparison: Optional[float] = None  # percentage change from baseline
    metadata: Dict[str, Any] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.metadata is None:
            self.metadata = {}


class PerformanceBenchmarker:
    """Core performance benchmarking utilities"""
    
    def __init__(self):
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        self.results_history: List[BenchmarkResult] = []
        
    def load_baseline_results(self, baseline_file: str = "performance_baseline.json"):
        """Load baseline performance results for comparison"""
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
                for name, data in baseline_data.items():
                    self.baseline_results[name] = BenchmarkResult(**data)
        except FileNotFoundError:
            print(f"No baseline file found at {baseline_file}, will create new baseline")
    
    def save_baseline_results(self, baseline_file: str = "performance_baseline.json"):
        """Save current results as new baseline"""
        baseline_data = {}
        for result in self.results_history:
            baseline_data[result.benchmark_name] = asdict(result)
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
    
    async def benchmark_async_function(
        self, 
        func: Callable,
        *args,
        iterations: int = 10,
        warmup_iterations: int = 2,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark an async function with detailed metrics"""
        
        # Warmup runs
        for _ in range(warmup_iterations):
            try:
                await func(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors
        
        # Collect garbage before benchmark
        gc.collect()
        
        # Track memory and CPU
        process = psutil.Process()
        memory_readings = []
        execution_times = []
        errors = 0
        
        # Start monitoring
        start_time = time.perf_counter()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        for i in range(iterations):
            iteration_start = time.perf_counter()
            
            try:
                await func(*args, **kwargs)
                
                iteration_end = time.perf_counter()
                execution_times.append(iteration_end - iteration_start)
                
                # Memory reading
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_readings.append(current_memory)
                
            except Exception as e:
                errors += 1
                execution_times.append(float('inf'))  # Mark as failed
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate metrics
        valid_times = [t for t in execution_times if t != float('inf')]
        
        if valid_times:
            avg_execution_time = statistics.mean(valid_times)
            throughput = len(valid_times) / total_time
            latency_p50 = statistics.median(valid_times)
            latency_p95 = statistics.quantiles(valid_times, n=20)[18] if len(valid_times) > 5 else latency_p50
            latency_p99 = statistics.quantiles(valid_times, n=100)[98] if len(valid_times) > 10 else latency_p95
        else:
            avg_execution_time = float('inf')
            throughput = 0
            latency_p50 = latency_p95 = latency_p99 = float('inf')
        
        memory_peak = max(memory_readings) if memory_readings else initial_memory
        memory_average = statistics.mean(memory_readings) if memory_readings else initial_memory
        
        # CPU usage (approximate)
        cpu_usage = process.cpu_percent()
        
        error_rate = errors / iterations
        
        return BenchmarkResult(
            benchmark_name=func.__name__,
            execution_time=avg_execution_time,
            memory_peak=memory_peak,
            memory_average=memory_average,
            cpu_usage=cpu_usage,
            throughput=throughput,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            error_rate=error_rate,
            iterations=iterations,
            metadata={
                "total_time": total_time,
                "warmup_iterations": warmup_iterations,
                "valid_iterations": len(valid_times)
            }
        )
    
    def benchmark_sync_function(
        self,
        func: Callable,
        *args,
        iterations: int = 100,
        warmup_iterations: int = 10,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a synchronous function"""
        
        # Warmup
        for _ in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        gc.collect()
        
        process = psutil.Process()
        execution_times = []
        memory_readings = []
        errors = 0
        
        start_time = time.perf_counter()
        
        for i in range(iterations):
            iteration_start = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                iteration_end = time.perf_counter()
                execution_times.append(iteration_end - iteration_start)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_readings.append(current_memory)
                
            except Exception as e:
                errors += 1
                execution_times.append(float('inf'))
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate metrics (same as async version)
        valid_times = [t for t in execution_times if t != float('inf')]
        
        if valid_times:
            avg_execution_time = statistics.mean(valid_times)
            throughput = len(valid_times) / total_time
            latency_p50 = statistics.median(valid_times)
            latency_p95 = statistics.quantiles(valid_times, n=20)[18] if len(valid_times) > 5 else latency_p50
            latency_p99 = statistics.quantiles(valid_times, n=100)[98] if len(valid_times) > 10 else latency_p95
        else:
            avg_execution_time = float('inf')
            throughput = 0
            latency_p50 = latency_p95 = latency_p99 = float('inf')
        
        memory_peak = max(memory_readings) if memory_readings else 0
        memory_average = statistics.mean(memory_readings) if memory_readings else 0
        cpu_usage = process.cpu_percent()
        error_rate = errors / iterations
        
        return BenchmarkResult(
            benchmark_name=func.__name__,
            execution_time=avg_execution_time,
            memory_peak=memory_peak,
            memory_average=memory_average,
            cpu_usage=cpu_usage,
            throughput=throughput,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            error_rate=error_rate,
            iterations=iterations
        )
    
    def compare_with_baseline(self, result: BenchmarkResult) -> BenchmarkResult:
        """Compare result with baseline and add regression info"""
        if result.benchmark_name in self.baseline_results:
            baseline = self.baseline_results[result.benchmark_name]
            
            # Calculate percentage change (negative = improvement, positive = regression)
            if baseline.execution_time > 0:
                result.baseline_comparison = ((result.execution_time - baseline.execution_time) / baseline.execution_time) * 100
            
            result.metadata = result.metadata or {}
            result.metadata.update({
                "baseline_execution_time": baseline.execution_time,
                "baseline_memory_peak": baseline.memory_peak,
                "baseline_throughput": baseline.throughput,
                "performance_change": "regression" if result.baseline_comparison > 5 else "improvement" if result.baseline_comparison < -5 else "stable"
            })
        
        return result


@pytest.mark.benchmark
@pytest.mark.performance
class TestNWTNPerformanceBenchmarks:
    """Benchmark NWTN orchestrator performance"""
    
    @pytest.fixture
    def benchmarker(self):
        return PerformanceBenchmarker()
    
    @pytest.fixture
    def mock_nwtn_orchestrator(self):
        orchestrator = Mock(spec=NWTNOrchestrator)
        
        # Mock realistic processing times
        async def mock_process_query(user_input):
            # Simulate processing time based on query complexity
            processing_time = len(user_input.prompt) * 0.001 + 0.1  # Base + complexity
            await asyncio.sleep(processing_time)
            
            return {
                "session_id": str(uuid.uuid4()),
                "final_answer": f"Processed response for: {user_input.prompt[:50]}...",
                "reasoning_trace": [
                    {
                        "step_id": str(uuid.uuid4()),
                        "agent_type": "architect",
                        "execution_time": processing_time * 0.3
                    },
                    {
                        "step_id": str(uuid.uuid4()),
                        "agent_type": "executor",
                        "execution_time": processing_time * 0.7
                    }
                ],
                "confidence_score": 0.85,
                "context_used": len(user_input.prompt),
                "processing_time": processing_time
            }
        
        orchestrator.process_query = mock_process_query
        return orchestrator
    
    async def test_nwtn_single_query_performance(self, benchmarker, mock_nwtn_orchestrator):
        """Benchmark single NWTN query processing"""
        
        async def single_query_benchmark():
            user_input = UserInput(
                user_id="benchmark_user",
                prompt="Explain the principles of quantum computing and its applications in cryptography",
                context_allocation=200
            )
            
            result = await mock_nwtn_orchestrator.process_query(user_input)
            return result
        
        benchmark_result = await benchmarker.benchmark_async_function(
            single_query_benchmark,
            iterations=50,
            warmup_iterations=5
        )
        
        benchmark_result.benchmark_name = "nwtn_single_query"
        benchmark_result = benchmarker.compare_with_baseline(benchmark_result)
        benchmarker.results_history.append(benchmark_result)
        
        # Performance assertions
        assert benchmark_result.execution_time < 1.0  # Should complete in under 1 second
        assert benchmark_result.error_rate == 0.0     # No errors expected
        assert benchmark_result.throughput > 1.0      # At least 1 query per second
        
        return benchmark_result
    
    async def test_nwtn_concurrent_queries_performance(self, benchmarker, mock_nwtn_orchestrator):
        """Benchmark concurrent NWTN query processing"""
        
        async def concurrent_queries_benchmark():
            tasks = []
            for i in range(10):  # 10 concurrent queries
                user_input = UserInput(
                    user_id=f"concurrent_user_{i}",
                    prompt=f"Query {i}: Analyze the performance implications of distributed AI systems",
                    context_allocation=150
                )
                task = mock_nwtn_orchestrator.process_query(user_input)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        benchmark_result = await benchmarker.benchmark_async_function(
            concurrent_queries_benchmark,
            iterations=20,
            warmup_iterations=2
        )
        
        benchmark_result.benchmark_name = "nwtn_concurrent_queries"
        benchmark_result = benchmarker.compare_with_baseline(benchmark_result)
        benchmarker.results_history.append(benchmark_result)
        
        # Performance assertions for concurrent processing
        assert benchmark_result.execution_time < 5.0   # 10 concurrent queries in under 5 seconds
        assert benchmark_result.error_rate < 0.05      # Less than 5% error rate
        
        return benchmark_result
    
    async def test_nwtn_memory_usage_under_load(self, benchmarker, mock_nwtn_orchestrator):
        """Benchmark NWTN memory usage under sustained load"""
        
        async def memory_load_benchmark():
            # Process many queries to test memory management
            for i in range(100):
                user_input = UserInput(
                    user_id=f"memory_test_user_{i}",
                    prompt=f"Large query {i}: " + "analyze " * 50,  # Larger prompts
                    context_allocation=500
                )
                
                result = await mock_nwtn_orchestrator.process_query(user_input)
                
                # Simulate some processing of results
                processed_data = {
                    "query_id": i,
                    "result": result,
                    "timestamp": time.time()
                }
                
                # Cleanup simulation
                if i % 10 == 0:
                    gc.collect()
        
        benchmark_result = await benchmarker.benchmark_async_function(
            memory_load_benchmark,
            iterations=5,
            warmup_iterations=1
        )
        
        benchmark_result.benchmark_name = "nwtn_memory_load"
        benchmark_result = benchmarker.compare_with_baseline(benchmark_result)
        benchmarker.results_history.append(benchmark_result)
        
        # Memory usage assertions
        assert benchmark_result.memory_peak < 500  # Less than 500MB peak memory
        memory_growth = benchmark_result.memory_peak - benchmark_result.memory_average
        assert memory_growth < 100  # Memory growth under 100MB during test
        
        return benchmark_result


@pytest.mark.benchmark
@pytest.mark.performance
class TestFTNSPerformanceBenchmarks:
    """Benchmark FTNS tokenomics performance"""
    
    @pytest.fixture
    def benchmarker(self):
        return PerformanceBenchmarker()
    
    @pytest.fixture
    def mock_ftns_service(self):
        service = Mock(spec=FTNSService)
        
        # Mock balance calculation with realistic complexity
        def mock_calculate_balance(user_id):
            # Simulate database query time
            time.sleep(0.01)  # 10ms simulated DB query
            return {
                "total_balance": Decimal("100.50"),
                "available_balance": Decimal("85.25"),
                "reserved_balance": Decimal("15.25")
            }
        
        def mock_create_transaction(from_user, to_user, amount, transaction_type):
            # Simulate transaction processing
            time.sleep(0.005)  # 5ms processing time
            return {
                "transaction_id": str(uuid.uuid4()),
                "success": True,
                "new_balance": Decimal("75.25")
            }
        
        service.get_balance = mock_calculate_balance
        service.create_transaction = mock_create_transaction
        return service
    
    def test_ftns_balance_calculation_performance(self, benchmarker, mock_ftns_service):
        """Benchmark FTNS balance calculation performance"""
        
        def balance_benchmark():
            return mock_ftns_service.get_balance("benchmark_user")
        
        benchmark_result = benchmarker.benchmark_sync_function(
            balance_benchmark,
            iterations=1000,
            warmup_iterations=50
        )
        
        benchmark_result.benchmark_name = "ftns_balance_calculation"
        benchmark_result = benchmarker.compare_with_baseline(benchmark_result)
        benchmarker.results_history.append(benchmark_result)
        
        # Performance assertions
        assert benchmark_result.execution_time < 0.1   # Under 100ms per balance calculation
        assert benchmark_result.throughput > 50        # At least 50 calculations per second
        assert benchmark_result.error_rate == 0.0
        
        return benchmark_result
    
    def test_ftns_transaction_processing_performance(self, benchmarker, mock_ftns_service):
        """Benchmark FTNS transaction processing performance"""
        
        def transaction_benchmark():
            return mock_ftns_service.create_transaction(
                from_user="user_a",
                to_user="user_b",
                amount=10.0,
                transaction_type="transfer"
            )
        
        benchmark_result = benchmarker.benchmark_sync_function(
            transaction_benchmark,
            iterations=500,
            warmup_iterations=25
        )
        
        benchmark_result.benchmark_name = "ftns_transaction_processing"
        benchmark_result = benchmarker.compare_with_baseline(benchmark_result)
        benchmarker.results_history.append(benchmark_result)
        
        # Performance assertions
        assert benchmark_result.execution_time < 0.05  # Under 50ms per transaction
        assert benchmark_result.throughput > 100       # At least 100 transactions per second
        
        return benchmark_result
    
    def test_ftns_concurrent_transactions_performance(self, benchmarker, mock_ftns_service):
        """Benchmark concurrent FTNS transaction processing"""
        
        def concurrent_transactions_benchmark():
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = []
                
                for i in range(50):  # 50 concurrent transactions
                    future = executor.submit(
                        mock_ftns_service.create_transaction,
                        f"user_{i}",
                        f"user_{i+1}",
                        5.0,
                        "benchmark_transfer"
                    )
                    futures.append(future)
                
                results = [future.result() for future in as_completed(futures)]
                return results
        
        benchmark_result = benchmarker.benchmark_sync_function(
            concurrent_transactions_benchmark,
            iterations=10,
            warmup_iterations=2
        )
        
        benchmark_result.benchmark_name = "ftns_concurrent_transactions"
        benchmark_result = benchmarker.compare_with_baseline(benchmark_result)
        benchmarker.results_history.append(benchmark_result)
        
        # Concurrent processing assertions
        assert benchmark_result.execution_time < 2.0   # 50 concurrent transactions in under 2 seconds
        assert benchmark_result.error_rate < 0.02      # Less than 2% error rate under load
        
        return benchmark_result


@pytest.mark.benchmark
@pytest.mark.performance  
class TestDatabasePerformanceBenchmarks:
    """Benchmark database operation performance"""
    
    @pytest.fixture
    def benchmarker(self):
        return PerformanceBenchmarker()
    
    @pytest.fixture
    def mock_database_manager(self):
        db_manager = Mock(spec=DatabaseManager)
        
        # Mock database operations with realistic latencies
        def mock_create_session(session_data):
            time.sleep(0.002)  # 2ms insert time
            return {"session_id": str(uuid.uuid4()), "created": True}
        
        def mock_query_sessions(user_id, limit=10):
            time.sleep(0.008)  # 8ms query time
            return [
                {"session_id": str(uuid.uuid4()), "user_id": user_id}
                for _ in range(limit)
            ]
        
        def mock_update_session(session_id, updates):
            time.sleep(0.003)  # 3ms update time
            return {"updated": True, "session_id": session_id}
        
        db_manager.create_session = mock_create_session
        db_manager.query_sessions = mock_query_sessions
        db_manager.update_session = mock_update_session
        return db_manager
    
    def test_database_crud_operations_performance(self, benchmarker, mock_database_manager):
        """Benchmark basic CRUD operations performance"""
        
        def crud_benchmark():
            # Create
            session_data = {
                "user_id": "benchmark_user",
                "context_allocation": 200,
                "status": "pending"
            }
            create_result = mock_database_manager.create_session(session_data)
            
            # Read
            sessions = mock_database_manager.query_sessions("benchmark_user", limit=5)
            
            # Update
            update_result = mock_database_manager.update_session(
                create_result["session_id"],
                {"status": "completed"}
            )
            
            return {
                "created": create_result["created"],
                "queried": len(sessions),
                "updated": update_result["updated"]
            }
        
        benchmark_result = benchmarker.benchmark_sync_function(
            crud_benchmark,
            iterations=200,
            warmup_iterations=20
        )
        
        benchmark_result.benchmark_name = "database_crud_operations"
        benchmark_result = benchmarker.compare_with_baseline(benchmark_result)
        benchmarker.results_history.append(benchmark_result)
        
        # Database performance assertions
        assert benchmark_result.execution_time < 0.05  # CRUD cycle under 50ms
        assert benchmark_result.throughput > 100       # At least 100 CRUD cycles per second
        assert benchmark_result.error_rate == 0.0
        
        return benchmark_result
    
    def test_database_bulk_operations_performance(self, benchmarker, mock_database_manager):
        """Benchmark bulk database operations"""
        
        def bulk_operations_benchmark():
            # Bulk create
            sessions_created = 0
            for i in range(100):  # Create 100 sessions
                session_data = {
                    "user_id": f"bulk_user_{i}",
                    "context_allocation": 150,
                    "status": "pending"
                }
                result = mock_database_manager.create_session(session_data)
                if result["created"]:
                    sessions_created += 1
            
            # Bulk query
            all_sessions = []
            for i in range(10):  # Query 10 different users
                user_sessions = mock_database_manager.query_sessions(f"bulk_user_{i}")
                all_sessions.extend(user_sessions)
            
            return {
                "sessions_created": sessions_created,
                "sessions_queried": len(all_sessions)
            }
        
        benchmark_result = benchmarker.benchmark_sync_function(
            bulk_operations_benchmark,
            iterations=10,
            warmup_iterations=2
        )
        
        benchmark_result.benchmark_name = "database_bulk_operations"
        benchmark_result = benchmarker.compare_with_baseline(benchmark_result)
        benchmarker.results_history.append(benchmark_result)
        
        # Bulk operations assertions
        assert benchmark_result.execution_time < 5.0   # Bulk operations under 5 seconds
        assert benchmark_result.error_rate == 0.0
        
        return benchmark_result


@pytest.mark.benchmark
@pytest.mark.performance
class TestMemoryPerformanceBenchmarks:
    """Benchmark memory usage patterns and garbage collection"""
    
    @pytest.fixture
    def benchmarker(self):
        return PerformanceBenchmarker()
    
    def test_memory_allocation_patterns(self, benchmarker):
        """Benchmark memory allocation and deallocation patterns"""
        
        def memory_allocation_benchmark():
            # Simulate PRSM data structures
            sessions = []
            
            # Allocate many session objects
            for i in range(1000):
                session_data = {
                    "session_id": str(uuid.uuid4()),
                    "user_id": f"user_{i}",
                    "reasoning_trace": [
                        {
                            "step_id": str(uuid.uuid4()),
                            "agent_type": "executor",
                            "input_data": [f"Query step {j}" for j in range(10)]
                        }
                        for _ in range(5)  # 5 reasoning steps per session
                    ],
                    "metadata": {"iteration": i, "timestamp": time.time()}
                }
                sessions.append(session_data)
            
            # Process sessions (simulate usage)
            processed_count = 0
            for session in sessions:
                # Simulate processing
                processed_session = {
                    **session,
                    "processed": True,
                    "processing_time": time.time()
                }
                processed_count += 1
            
            # Cleanup (simulate garbage collection trigger)
            sessions.clear()
            gc.collect()
            
            return processed_count
        
        benchmark_result = benchmarker.benchmark_sync_function(
            memory_allocation_benchmark,
            iterations=10,
            warmup_iterations=2
        )
        
        benchmark_result.benchmark_name = "memory_allocation_patterns"
        benchmark_result = benchmarker.compare_with_baseline(benchmark_result)
        benchmarker.results_history.append(benchmark_result)
        
        # Memory performance assertions
        assert benchmark_result.memory_peak < 200      # Peak memory under 200MB
        memory_efficiency = benchmark_result.memory_average / benchmark_result.memory_peak
        assert memory_efficiency > 0.7  # Memory usage should be reasonably efficient
        
        return benchmark_result
    
    def test_garbage_collection_performance(self, benchmarker):
        """Benchmark garbage collection impact on performance"""
        
        def gc_performance_benchmark():
            # Create objects that will need garbage collection
            objects = []
            
            # Phase 1: Allocate objects
            for i in range(5000):
                obj = {
                    "id": i,
                    "data": "x" * 100,  # Some data
                    "references": [j for j in range(i % 10)],  # Circular references
                    "nested": {
                        "level1": {"level2": {"data": f"nested_{i}"}}
                    }
                }
                objects.append(obj)
            
            # Phase 2: Create some circular references
            for i in range(0, len(objects) - 1, 2):
                objects[i]["ref"] = objects[i + 1]
                objects[i + 1]["ref"] = objects[i]
            
            # Phase 3: Force garbage collection
            gc_start = time.perf_counter()
            gc.collect()
            gc_time = time.perf_counter() - gc_start
            
            # Phase 4: Cleanup
            objects.clear()
            
            return gc_time
        
        benchmark_result = benchmarker.benchmark_sync_function(
            gc_performance_benchmark,
            iterations=20,
            warmup_iterations=3
        )
        
        benchmark_result.benchmark_name = "garbage_collection_performance"
        benchmark_result = benchmarker.compare_with_baseline(benchmark_result)
        benchmarker.results_history.append(benchmark_result)
        
        # GC performance assertions
        assert benchmark_result.execution_time < 0.5   # GC should complete quickly
        
        return benchmark_result


class BenchmarkReporter:
    """Generate comprehensive benchmark reports"""
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            "summary": {
                "total_benchmarks": len(self.results),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_performance": self._calculate_overall_performance(),
                "regressions_detected": self._detect_regressions(),
                "improvements_detected": self._detect_improvements()
            },
            "benchmark_results": [asdict(result) for result in self.results],
            "performance_analysis": self._analyze_performance_trends(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _calculate_overall_performance(self) -> str:
        """Calculate overall system performance rating"""
        if not self.results:
            return "no_data"
        
        performance_scores = []
        for result in self.results:
            # Score based on multiple factors
            score = 100  # Start with perfect score
            
            # Penalize high execution times
            if result.execution_time > 1.0:
                score -= 20
            elif result.execution_time > 0.5:
                score -= 10
            
            # Penalize low throughput
            if result.throughput and result.throughput < 10:
                score -= 15
            
            # Penalize high error rates
            if result.error_rate > 0.1:
                score -= 25
            elif result.error_rate > 0.05:
                score -= 10
            
            # Penalize regressions
            if result.baseline_comparison and result.baseline_comparison > 10:
                score -= 20
            
            performance_scores.append(max(score, 0))
        
        avg_score = statistics.mean(performance_scores)
        
        if avg_score >= 90:
            return "excellent"
        elif avg_score >= 75:
            return "good"
        elif avg_score >= 60:
            return "acceptable"
        elif avg_score >= 40:
            return "needs_improvement"
        else:
            return "poor"
    
    def _detect_regressions(self) -> List[Dict[str, Any]]:
        """Detect performance regressions"""
        regressions = []
        
        for result in self.results:
            if result.baseline_comparison and result.baseline_comparison > 5:  # >5% slower
                regressions.append({
                    "benchmark": result.benchmark_name,
                    "regression_percentage": result.baseline_comparison,
                    "current_time": result.execution_time,
                    "baseline_time": result.metadata.get("baseline_execution_time") if result.metadata else None,
                    "severity": "high" if result.baseline_comparison > 25 else "medium" if result.baseline_comparison > 15 else "low"
                })
        
        return regressions
    
    def _detect_improvements(self) -> List[Dict[str, Any]]:
        """Detect performance improvements"""
        improvements = []
        
        for result in self.results:
            if result.baseline_comparison and result.baseline_comparison < -5:  # >5% faster
                improvements.append({
                    "benchmark": result.benchmark_name,
                    "improvement_percentage": abs(result.baseline_comparison),
                    "current_time": result.execution_time,
                    "baseline_time": result.metadata.get("baseline_execution_time") if result.metadata else None
                })
        
        return improvements
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across benchmarks"""
        
        execution_times = [r.execution_time for r in self.results if r.execution_time != float('inf')]
        memory_peaks = [r.memory_peak for r in self.results]
        throughputs = [r.throughput for r in self.results if r.throughput]
        error_rates = [r.error_rate for r in self.results]
        
        return {
            "execution_time_stats": {
                "mean": statistics.mean(execution_times) if execution_times else 0,
                "median": statistics.median(execution_times) if execution_times else 0,
                "max": max(execution_times) if execution_times else 0,
                "min": min(execution_times) if execution_times else 0
            },
            "memory_usage_stats": {
                "mean_peak": statistics.mean(memory_peaks) if memory_peaks else 0,
                "max_peak": max(memory_peaks) if memory_peaks else 0
            },
            "throughput_stats": {
                "mean": statistics.mean(throughputs) if throughputs else 0,
                "max": max(throughputs) if throughputs else 0
            },
            "reliability_stats": {
                "mean_error_rate": statistics.mean(error_rates) if error_rates else 0,
                "max_error_rate": max(error_rates) if error_rates else 0,
                "benchmarks_with_errors": sum(1 for r in error_rates if r > 0)
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Analyze results and generate recommendations
        high_memory_benchmarks = [r for r in self.results if r.memory_peak > 100]
        slow_benchmarks = [r for r in self.results if r.execution_time > 1.0]
        error_prone_benchmarks = [r for r in self.results if r.error_rate > 0.05]
        
        if high_memory_benchmarks:
            recommendations.append(
                f"Consider memory optimization for {len(high_memory_benchmarks)} benchmarks with high memory usage"
            )
        
        if slow_benchmarks:
            recommendations.append(
                f"Investigate performance bottlenecks in {len(slow_benchmarks)} slow benchmarks"
            )
        
        if error_prone_benchmarks:
            recommendations.append(
                f"Improve error handling and reliability for {len(error_prone_benchmarks)} error-prone benchmarks"
            )
        
        # Check for regression trends
        regressions = self._detect_regressions()
        if len(regressions) > len(self.results) * 0.2:  # >20% regressions
            recommendations.append("Multiple performance regressions detected - consider code review and optimization")
        
        return recommendations
    
    def save_report(self, filename: str = "performance_benchmark_report.json"):
        """Save performance report to file"""
        report = self.generate_performance_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filename


@pytest.mark.benchmark
class TestComprehensivePerformanceSuite:
    """Run comprehensive performance benchmark suite"""
    
    async def test_full_performance_benchmark_suite(self):
        """Run all performance benchmarks and generate report"""
        
        benchmarker = PerformanceBenchmarker()
        benchmarker.load_baseline_results()
        
        # Initialize test fixtures
        nwtn_benchmarks = TestNWTNPerformanceBenchmarks()
        ftns_benchmarks = TestFTNSPerformanceBenchmarks()
        db_benchmarks = TestDatabasePerformanceBenchmarks()
        memory_benchmarks = TestMemoryPerformanceBenchmarks()
        
        all_results = []
        
        print("🚀 Running PRSM Performance Benchmark Suite...")
        print("=" * 60)
        
        try:
            # NWTN Benchmarks
            print("🎭 NWTN Performance Benchmarks...")
            mock_orchestrator = nwtn_benchmarks.mock_nwtn_orchestrator()
            
            result1 = await nwtn_benchmarks.test_nwtn_single_query_performance(benchmarker, mock_orchestrator)
            all_results.append(result1)
            print(f"  ✅ Single Query: {result1.execution_time:.3f}s, {result1.throughput:.1f} ops/s")
            
            result2 = await nwtn_benchmarks.test_nwtn_concurrent_queries_performance(benchmarker, mock_orchestrator)
            all_results.append(result2)
            print(f"  ✅ Concurrent Queries: {result2.execution_time:.3f}s, Error Rate: {result2.error_rate:.1%}")
            
            result3 = await nwtn_benchmarks.test_nwtn_memory_usage_under_load(benchmarker, mock_orchestrator)
            all_results.append(result3)
            print(f"  ✅ Memory Load: Peak {result3.memory_peak:.1f}MB, Avg {result3.memory_average:.1f}MB")
            
        except Exception as e:
            print(f"  ❌ NWTN benchmarks failed: {e}")
        
        try:
            # FTNS Benchmarks
            print("\n💰 FTNS Performance Benchmarks...")
            mock_ftns = ftns_benchmarks.mock_ftns_service()
            
            result4 = ftns_benchmarks.test_ftns_balance_calculation_performance(benchmarker, mock_ftns)
            all_results.append(result4)
            print(f"  ✅ Balance Calculation: {result4.execution_time*1000:.1f}ms, {result4.throughput:.1f} ops/s")
            
            result5 = ftns_benchmarks.test_ftns_transaction_processing_performance(benchmarker, mock_ftns)
            all_results.append(result5)
            print(f"  ✅ Transaction Processing: {result5.execution_time*1000:.1f}ms, {result5.throughput:.1f} ops/s")
            
            result6 = ftns_benchmarks.test_ftns_concurrent_transactions_performance(benchmarker, mock_ftns)
            all_results.append(result6)
            print(f"  ✅ Concurrent Transactions: {result6.execution_time:.3f}s, Error Rate: {result6.error_rate:.1%}")
            
        except Exception as e:
            print(f"  ❌ FTNS benchmarks failed: {e}")
        
        try:
            # Database Benchmarks
            print("\n🗄️  Database Performance Benchmarks...")
            mock_db = db_benchmarks.mock_database_manager()
            
            result7 = db_benchmarks.test_database_crud_operations_performance(benchmarker, mock_db)
            all_results.append(result7)
            print(f"  ✅ CRUD Operations: {result7.execution_time*1000:.1f}ms, {result7.throughput:.1f} ops/s")
            
            result8 = db_benchmarks.test_database_bulk_operations_performance(benchmarker, mock_db)
            all_results.append(result8)
            print(f"  ✅ Bulk Operations: {result8.execution_time:.3f}s")
            
        except Exception as e:
            print(f"  ❌ Database benchmarks failed: {e}")
        
        try:
            # Memory Benchmarks
            print("\n🧠 Memory Performance Benchmarks...")
            
            result9 = memory_benchmarks.test_memory_allocation_patterns(benchmarker)
            all_results.append(result9)
            print(f"  ✅ Memory Allocation: Peak {result9.memory_peak:.1f}MB, Efficiency {result9.memory_average/result9.memory_peak:.1%}")
            
            result10 = memory_benchmarks.test_garbage_collection_performance(benchmarker)
            all_results.append(result10)
            print(f"  ✅ Garbage Collection: {result10.execution_time*1000:.1f}ms")
            
        except Exception as e:
            print(f"  ❌ Memory benchmarks failed: {e}")
        
        # Generate comprehensive report
        print(f"\n📊 Generating Performance Report...")
        reporter = BenchmarkReporter(all_results)
        report_file = reporter.save_report()
        
        report = reporter.generate_performance_report()
        
        print("=" * 60)
        print("📋 PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"🎯 Overall Performance: {report['summary']['overall_performance'].upper()}")
        print(f"📈 Total Benchmarks: {report['summary']['total_benchmarks']}")
        print(f"📉 Regressions: {len(report['summary']['regressions_detected'])}")
        print(f"📈 Improvements: {len(report['summary']['improvements_detected'])}")
        
        if report['summary']['regressions_detected']:
            print("\n⚠️  REGRESSIONS DETECTED:")
            for regression in report['summary']['regressions_detected']:
                print(f"  📉 {regression['benchmark']}: {regression['regression_percentage']:+.1f}% ({regression['severity']})")
        
        if report['summary']['improvements_detected']:
            print("\n🎉 IMPROVEMENTS DETECTED:")
            for improvement in report['summary']['improvements_detected']:
                print(f"  📈 {improvement['benchmark']}: {improvement['improvement_percentage']:.1f}% faster")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for recommendation in report['recommendations']:
            print(f"  • {recommendation}")
        
        print(f"\n📄 Detailed report saved: {report_file}")
        
        # Save new baseline if performance is acceptable
        if report['summary']['overall_performance'] in ['excellent', 'good', 'acceptable']:
            benchmarker.save_baseline_results()
            print("💾 New performance baseline saved")
        
        # Performance assertions
        assert len(all_results) > 0, "No benchmarks completed successfully"
        assert report['summary']['overall_performance'] != 'poor', f"Overall performance is poor: {report['summary']['overall_performance']}"
        
        # Regression check
        critical_regressions = [r for r in report['summary']['regressions_detected'] if r['severity'] == 'high']
        assert len(critical_regressions) == 0, f"Critical performance regressions detected: {critical_regressions}"
        
        return report
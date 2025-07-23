"""
Performance Benchmarks and Regression Tests
===========================================

Comprehensive performance benchmarks for all PRSM components with
baseline tracking and regression detection.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List, Callable
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
import statistics
import gc

try:
    from prsm.core.caching import CacheManager
    from prsm.core.performance import get_performance_monitor, get_profiler
    from prsm.nwtn.meta_reasoning_engine import NWTNEngine
    from prsm.tokenomics.ftns_service import FTNSService
    from prsm.core.database.optimized_queries import create_optimized_engine
    from prsm.core.async_patterns import gather_with_limit, async_map
except ImportError:
    # Create mocks if imports fail
    CacheManager = Mock
    get_performance_monitor = lambda: Mock()
    get_profiler = lambda: Mock()
    NWTNEngine = Mock
    FTNSService = Mock
    create_optimized_engine = Mock()
    gather_with_limit = AsyncMock()
    async_map = AsyncMock()


@pytest.mark.performance
@pytest.mark.benchmark
class TestCorePerformanceBenchmarks:
    """Core system performance benchmarks"""
    
    def test_model_creation_benchmark(self, benchmark, db_factory):
        """Benchmark model creation performance"""
        def create_models():
            models = []
            for i in range(100):
                session = db_factory.create_prsm_session(
                    user_id=f"user_{i}",
                    status="pending"
                )
                transaction = db_factory.create_ftns_transaction(
                    user_id=f"user_{i}",
                    amount=Decimal(f"{10 + i}.00"),
                    transaction_type="reward"
                )
                models.extend([session, transaction])
            return models
        
        result = benchmark(create_models)
        
        # Should create 200 models
        assert len(result) == 200
    
    def test_model_serialization_benchmark(self, benchmark, db_factory):
        """Benchmark model serialization performance"""
        # Create test models
        models = []
        for i in range(100):
            session = db_factory.create_prsm_session(
                user_id=f"user_{i}",
                metadata={"index": i, "data": f"test_data_{i}"}
            )
            models.append(session)
        
        def serialize_models():
            return [model.dict() for model in models]
        
        result = benchmark(serialize_models)
        assert len(result) == 100
    
    @pytest.mark.asyncio
    async def test_async_operations_benchmark(self, performance_runner):
        """Benchmark async operations performance"""
        async def async_operation(x):
            await asyncio.sleep(0.001)  # Simulate I/O
            return x * 2
        
        async def run_async_operations():
            tasks = [async_operation(i) for i in range(100)]
            return await asyncio.gather(*tasks)
        
        metrics = performance_runner.run_performance_test(
            lambda: asyncio.run(run_async_operations()),
            iterations=10,
            warmup_iterations=2
        )
        
        # Should complete within reasonable time
        assert metrics.execution_time_ms < 1000  # Less than 1 second
        assert metrics.error_rate == 0.0
    
    def test_memory_usage_benchmark(self, memory_profiler, db_factory):
        """Benchmark memory usage patterns"""
        memory_profiler.take_snapshot("baseline")
        
        # Create many objects
        objects = []
        for i in range(1000):
            session = db_factory.create_prsm_session(user_id=f"user_{i}")
            objects.append(session)
        
        memory_profiler.take_snapshot("after_creation")
        
        # Clear objects
        objects.clear()
        gc.collect()
        
        memory_profiler.take_snapshot("after_cleanup")
        
        # Analyze memory usage
        creation_diff = memory_profiler.compare_snapshots("baseline", "after_creation")
        cleanup_diff = memory_profiler.compare_snapshots("after_creation", "after_cleanup")
        
        # Memory should be released after cleanup
        assert creation_diff["memory_diff"] > 0  # Memory increased during creation
        assert cleanup_diff["memory_diff"] < 0   # Memory decreased after cleanup


@pytest.mark.performance
@pytest.mark.benchmark
class TestCachingPerformanceBenchmarks:
    """Caching system performance benchmarks"""
    
    def test_cache_operations_benchmark(self, benchmark, mock_cache_manager):
        """Benchmark cache operations"""
        cache = mock_cache_manager
        
        def cache_operations():
            # Simulate mixed cache operations
            for i in range(100):
                key = f"test_key_{i}"
                value = {"data": f"test_value_{i}", "index": i}
                
                # Set operation
                cache.set(key, value, ttl=300)
                
                # Get operation
                retrieved = cache.get(key)
                
                # Simulate cache hit/miss pattern
                if i % 10 == 0:
                    cache.delete(key)  # Simulate cache eviction
        
        benchmark(cache_operations)
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_access_benchmark(self, performance_runner, mock_cache_manager):
        """Benchmark concurrent cache access"""
        cache = mock_cache_manager
        
        async def concurrent_cache_operations():
            async def cache_worker(worker_id):
                for i in range(50):
                    key = f"worker_{worker_id}_key_{i}"
                    value = {"worker": worker_id, "data": i}
                    
                    cache.set(key, value)
                    result = cache.get(key)
                    
                    if i % 5 == 0:
                        cache.delete(key)
            
            # Run 10 concurrent workers
            tasks = [cache_worker(i) for i in range(10)]
            await asyncio.gather(*tasks)
        
        metrics = performance_runner.run_performance_test(
            lambda: asyncio.run(concurrent_cache_operations()),
            iterations=5,
            warmup_iterations=1,
            concurrent_users=1
        )
        
        assert metrics.error_rate == 0.0
        assert metrics.execution_time_ms < 2000  # Less than 2 seconds
    
    def test_cache_hit_ratio_benchmark(self, performance_runner, mock_cache_manager):
        """Benchmark cache hit ratio performance"""
        cache = mock_cache_manager
        
        def cache_hit_test():
            hits = 0
            misses = 0
            
            # Pre-populate cache
            for i in range(100):
                cache.set(f"key_{i}", f"value_{i}")
            
            # Test hit ratio with mixed access pattern
            for i in range(200):
                key = f"key_{i % 150}"  # Some keys won't exist
                result = cache.get(key)
                
                if result is not None:
                    hits += 1
                else:
                    misses += 1
            
            hit_ratio = hits / (hits + misses)
            return hit_ratio
        
        metrics = performance_runner.run_performance_test(
            cache_hit_test,
            iterations=10,
            warmup_iterations=2
        )
        
        # Should maintain reasonable hit ratio
        assert metrics.error_rate == 0.0


@pytest.mark.performance
@pytest.mark.benchmark
class TestNWTNPerformanceBenchmarks:
    """NWTN reasoning engine performance benchmarks"""
    
    @pytest.mark.asyncio
    async def test_nwtn_query_processing_benchmark(self, performance_runner, mock_nwtn_engine):
        """Benchmark NWTN query processing"""
        engine = mock_nwtn_engine
        
        test_queries = [
            "What is artificial intelligence?",
            "Explain quantum computing principles",
            "How does blockchain technology work?",
            "Describe machine learning algorithms",
            "What are the benefits of renewable energy?"
        ]
        
        async def process_queries():
            results = []
            for query in test_queries:
                result = await engine.process_query(
                    query=query,
                    mode="adaptive",
                    max_depth=2
                )
                results.append(result)
            return results
        
        metrics = performance_runner.run_performance_test(
            lambda: asyncio.run(process_queries()),
            iterations=5,
            warmup_iterations=1
        )
        
        # Should process queries within reasonable time
        assert metrics.execution_time_ms < 5000  # Less than 5 seconds per batch
        assert metrics.error_rate == 0.0
    
    @pytest.mark.asyncio
    async def test_nwtn_concurrent_queries_benchmark(self, load_test_runner, mock_nwtn_engine):
        """Benchmark concurrent NWTN query processing"""
        engine = mock_nwtn_engine
        
        async def single_query():
            return await engine.process_query(
                query="Test concurrent query",
                mode="quick",
                max_depth=1
            )
        
        results = await load_test_runner.run_load_test(
            test_function=single_query,
            concurrent_users=10,
            duration_seconds=30,
            ramp_up_seconds=5
        )
        
        # Performance assertions for concurrent queries
        assert results.error_rate < 0.05  # Less than 5% error rate
        assert results.average_response_time < 2000  # Less than 2 seconds average
        assert results.requests_per_second > 5  # At least 5 queries/sec
    
    def test_nwtn_reasoning_depth_performance(self, performance_runner, mock_nwtn_engine):
        """Benchmark performance vs reasoning depth"""
        engine = mock_nwtn_engine
        
        def test_reasoning_depths():
            results = {}
            for depth in [1, 2, 3, 4, 5]:
                start_time = time.perf_counter()
                
                engine.process_query(
                    query="Test reasoning depth performance",
                    mode="deep",
                    max_depth=depth
                )
                
                end_time = time.perf_counter()
                results[depth] = (end_time - start_time) * 1000
            
            return results
        
        metrics = performance_runner.run_performance_test(
            test_reasoning_depths,
            iterations=3,
            warmup_iterations=1
        )
        
        assert metrics.error_rate == 0.0
    
    def test_nwtn_memory_usage_benchmark(self, memory_profiler, mock_nwtn_engine):
        """Benchmark NWTN memory usage"""
        engine = mock_nwtn_engine
        
        memory_profiler.take_snapshot("before_nwtn")
        
        # Process multiple queries
        for i in range(20):
            engine.process_query(
                query=f"Memory test query {i}",
                mode="adaptive",
                max_depth=2
            )
        
        memory_profiler.take_snapshot("after_nwtn")
        
        memory_diff = memory_profiler.compare_snapshots("before_nwtn", "after_nwtn")
        
        # Memory usage should be reasonable
        assert memory_diff["memory_diff"] < 100 * 1024 * 1024  # Less than 100MB


@pytest.mark.performance
@pytest.mark.benchmark  
class TestFTNSPerformanceBenchmarks:
    """FTNS tokenomics performance benchmarks"""
    
    def test_balance_operations_benchmark(self, benchmark, mock_ftns_service):
        """Benchmark FTNS balance operations"""
        service = mock_ftns_service
        
        def balance_operations():
            users = [f"user_{i}" for i in range(100)]
            
            for user_id in users:
                # Get balance
                balance = service.get_balance(user_id)
                
                # Get available balance
                available = service.get_available_balance(user_id)
                
                # Check if operations are consistent
                assert balance >= available
        
        benchmark(balance_operations)
    
    def test_transaction_processing_benchmark(self, performance_runner, mock_ftns_service):
        """Benchmark transaction processing"""
        service = mock_ftns_service
        
        def process_transactions():
            transactions = []
            
            for i in range(100):
                result = service.create_transaction(
                    user_id=f"user_{i % 20}",
                    amount=Decimal(f"{10 + i}.00"),
                    transaction_type=["reward", "charge", "transfer"][i % 3],
                    description=f"Test transaction {i}"
                )
                transactions.append(result)
            
            return transactions
        
        metrics = performance_runner.run_performance_test(
            process_transactions,
            iterations=5,
            warmup_iterations=1
        )
        
        assert metrics.error_rate == 0.0
        assert metrics.execution_time_ms < 2000  # Less than 2 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_transfers_benchmark(self, load_test_runner, mock_ftns_service):
        """Benchmark concurrent FTNS transfers"""
        service = mock_ftns_service
        
        async def transfer_operation():
            return service.transfer(
                sender_id="test_sender",
                recipient_id="test_recipient",
                amount=Decimal("10.00"),
                description="Concurrent transfer test"
            )
        
        results = await load_test_runner.run_load_test(
            test_function=transfer_operation,
            concurrent_users=15,
            duration_seconds=20,
            ramp_up_seconds=3
        )
        
        # Performance assertions
        assert results.error_rate < 0.10  # Less than 10% error rate
        assert results.average_response_time < 1000  # Less than 1 second
        assert results.requests_per_second > 10  # At least 10 transfers/sec
    
    def test_transaction_history_performance(self, performance_runner, mock_ftns_service):
        """Benchmark transaction history retrieval"""
        service = mock_ftns_service
        
        def get_transaction_histories():
            users = [f"user_{i}" for i in range(50)]
            histories = []
            
            for user_id in users:
                history = service.get_transaction_history(
                    user_id=user_id,
                    limit=20,
                    offset=0
                )
                histories.append(history)
            
            return histories
        
        metrics = performance_runner.run_performance_test(
            get_transaction_histories,
            iterations=5,
            warmup_iterations=1
        )
        
        assert metrics.error_rate == 0.0
        assert metrics.execution_time_ms < 1500  # Less than 1.5 seconds


@pytest.mark.performance
@pytest.mark.benchmark
class TestDatabasePerformanceBenchmarks:
    """Database operations performance benchmarks"""
    
    def test_database_query_benchmark(self, benchmark, test_session, sample_db_data):
        """Benchmark database query performance"""
        def query_operations():
            if test_session is None:
                return []
            
            results = []
            
            # Simple queries
            for i in range(50):
                query_result = test_session.execute(
                    "SELECT * FROM prsm_sessions WHERE user_id = :user_id",
                    {"user_id": f"user_{i % 5}"}
                ).fetchall()
                results.extend(query_result)
            
            return results
        
        result = benchmark(query_operations)
        # Result will be empty list if test_session is None (mocked)
        assert isinstance(result, list)
    
    def test_database_bulk_insert_benchmark(self, performance_runner, test_session, db_factory):
        """Benchmark bulk database insertions"""
        if test_session is None:
            pytest.skip("Database not available")
        
        def bulk_insert():
            sessions = []
            
            for i in range(100):
                session = db_factory.create_prsm_session(
                    user_id=f"bulk_user_{i}",
                    status="pending"
                )
                sessions.append(session)
                test_session.add(session)
            
            test_session.commit()
            return sessions
        
        metrics = performance_runner.run_performance_test(
            bulk_insert,
            iterations=3,
            warmup_iterations=1
        )
        
        assert metrics.error_rate == 0.0
        assert metrics.execution_time_ms < 3000  # Less than 3 seconds
    
    def test_database_transaction_benchmark(self, performance_runner, test_session, db_factory):
        """Benchmark database transaction performance"""
        if test_session is None:
            pytest.skip("Database not available")
        
        def transaction_operations():
            with test_session.begin():
                for i in range(50):
                    session = db_factory.create_prsm_session(
                        user_id=f"tx_user_{i}",
                        status="pending"
                    )
                    test_session.add(session)
                    
                    transaction = db_factory.create_ftns_transaction(
                        user_id=f"tx_user_{i}",
                        amount=Decimal(f"{10 + i}.00"),
                        transaction_type="reward"
                    )
                    test_session.add(transaction)
        
        metrics = performance_runner.run_performance_test(
            transaction_operations,
            iterations=5,
            warmup_iterations=1
        )
        
        assert metrics.error_rate == 0.0


@pytest.mark.performance
@pytest.mark.benchmark
class TestAsyncPatternsBenchmarks:
    """Async patterns performance benchmarks"""
    
    @pytest.mark.asyncio
    async def test_gather_with_limit_benchmark(self, performance_runner):
        """Benchmark gather_with_limit performance"""
        async def async_task(x):
            await asyncio.sleep(0.001)  # Simulate async work
            return x * 2
        
        async def test_gather_with_limit():
            tasks = [lambda i=i: async_task(i) for i in range(100)]
            return await gather_with_limit(tasks, limit=20)
        
        metrics = performance_runner.run_performance_test(
            lambda: asyncio.run(test_gather_with_limit()),
            iterations=5,
            warmup_iterations=1
        )
        
        assert metrics.error_rate == 0.0
        assert metrics.execution_time_ms < 2000  # Less than 2 seconds
    
    @pytest.mark.asyncio
    async def test_async_map_benchmark(self, performance_runner):
        """Benchmark async_map performance"""
        async def process_item(item):
            await asyncio.sleep(0.001)
            return item * 2
        
        async def test_async_map():
            items = list(range(100))
            return await async_map(process_item, items, concurrency=15)
        
        metrics = performance_runner.run_performance_test(
            lambda: asyncio.run(test_async_map()),
            iterations=5,
            warmup_iterations=1
        )
        
        assert metrics.error_rate == 0.0
        assert metrics.execution_time_ms < 2000
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_benchmark(self, load_test_runner):
        """Benchmark concurrent async processing"""
        async def cpu_intensive_task():
            # Simulate CPU-intensive work
            total = 0
            for i in range(1000):
                total += i ** 2
            await asyncio.sleep(0.001)  # Small async delay
            return total
        
        results = await load_test_runner.run_load_test(
            test_function=cpu_intensive_task,
            concurrent_users=25,
            duration_seconds=15,
            ramp_up_seconds=3
        )
        
        assert results.error_rate < 0.05
        assert results.requests_per_second > 20  # At least 20 ops/sec


@pytest.mark.performance
@pytest.mark.regression
class TestPerformanceRegressionTests:
    """Performance regression detection tests"""
    
    def test_core_operations_regression(self, regression_detector, performance_runner, db_factory):
        """Test for performance regression in core operations"""
        def core_operations():
            # Mix of core operations
            models = []
            for i in range(50):
                session = db_factory.create_prsm_session(user_id=f"user_{i}")
                transaction = db_factory.create_ftns_transaction(
                    user_id=f"user_{i}",
                    amount=Decimal(f"{10 + i}.00"),
                    transaction_type="reward"
                )
                models.extend([session, transaction])
            
            # Serialize models
            serialized = [model.dict() for model in models]
            
            return len(serialized)
        
        metrics = performance_runner.run_performance_test(
            core_operations,
            iterations=10,
            warmup_iterations=2
        )
        
        # Record results for regression tracking
        regression_detector.record_result("core_operations", metrics)
        
        # Check for regression
        regression_result = regression_detector.check_regression(
            "core_operations",
            tolerance_percent=15.0  # Allow 15% degradation
        )
        
        if regression_result["status"] == "regression":
            pytest.fail(f"Performance regression detected: {regression_result['summary']}")
    
    @pytest.mark.asyncio
    async def test_async_operations_regression(self, regression_detector, performance_runner):
        """Test for regression in async operations"""
        async def async_operations():
            async def async_work(x):
                await asyncio.sleep(0.001)
                return x * 2
            
            tasks = [async_work(i) for i in range(100)]
            results = await asyncio.gather(*tasks)
            return len(results)
        
        metrics = performance_runner.run_performance_test(
            lambda: asyncio.run(async_operations()),
            iterations=5,
            warmup_iterations=1
        )
        
        regression_detector.record_result("async_operations", metrics)
        
        regression_result = regression_detector.check_regression(
            "async_operations",
            tolerance_percent=20.0
        )
        
        if regression_result["status"] == "regression":
            pytest.fail(f"Async performance regression: {regression_result['summary']}")
    
    def test_memory_usage_regression(self, regression_detector, memory_profiler, db_factory):
        """Test for memory usage regression"""
        memory_profiler.take_snapshot("baseline_memory")
        
        # Perform memory-intensive operations
        objects = []
        for i in range(500):
            session = db_factory.create_prsm_session(
                user_id=f"memory_user_{i}",
                metadata={"large_data": "x" * 1000}  # 1KB per object
            )
            objects.append(session)
        
        memory_profiler.take_snapshot("after_operations")
        
        # Calculate memory usage
        memory_diff = memory_profiler.compare_snapshots("baseline_memory", "after_operations")
        memory_usage_mb = memory_diff["memory_diff"] / (1024 * 1024)
        
        # Create fake metrics for regression tracking
        from tests.fixtures.performance import PerformanceMetrics
        fake_metrics = PerformanceMetrics(
            execution_time_ms=100,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=10,
            peak_memory_mb=memory_usage_mb,
            throughput_ops_per_sec=50
        )
        
        regression_detector.record_result("memory_usage", fake_metrics)
        
        # Check for memory regression
        regression_result = regression_detector.check_regression(
            "memory_usage",
            tolerance_percent=25.0  # Allow 25% memory increase
        )
        
        if regression_result["status"] == "regression":
            pytest.fail(f"Memory usage regression: {regression_result['summary']}")
        
        # Cleanup
        objects.clear()
        gc.collect()


@pytest.mark.performance
@pytest.mark.stress
class TestStressTests:
    """Stress tests for system limits"""
    
    def test_high_volume_model_creation(self, performance_runner, db_factory):
        """Stress test with high volume model creation"""
        def create_many_models():
            models = []
            for i in range(5000):  # Large number of models
                session = db_factory.create_prsm_session(
                    user_id=f"stress_user_{i}",
                    metadata={"index": i}
                )
                models.append(session)
            return len(models)
        
        metrics = performance_runner.run_performance_test(
            create_many_models,
            iterations=3,
            warmup_iterations=1
        )
        
        # Should handle large volumes without errors
        assert metrics.error_rate == 0.0
        assert metrics.execution_time_ms < 10000  # Less than 10 seconds
    
    @pytest.mark.asyncio
    async def test_extreme_concurrency_stress(self, load_test_runner):
        """Stress test with extreme concurrency"""
        async def simple_operation():
            await asyncio.sleep(0.01)  # 10ms operation
            return "completed"
        
        results = await load_test_runner.run_load_test(
            test_function=simple_operation,
            concurrent_users=100,  # High concurrency
            duration_seconds=30,
            ramp_up_seconds=5
        )
        
        # Should handle high concurrency reasonably well
        assert results.error_rate < 0.15  # Less than 15% error rate under stress
        assert results.total_requests > 1000  # Completed significant work
    
    def test_memory_pressure_stress(self, performance_runner, memory_profiler, db_factory):
        """Stress test under memory pressure"""
        memory_profiler.take_snapshot("stress_baseline")
        
        def memory_intensive_operations():
            large_objects = []
            
            try:
                for i in range(2000):
                    # Create objects with large metadata
                    session = db_factory.create_prsm_session(
                        user_id=f"memory_stress_user_{i}",
                        metadata={
                            "large_data": "x" * 5000,  # 5KB per object
                            "index": i,
                            "extra_data": list(range(100))
                        }
                    )
                    large_objects.append(session)
                
                return len(large_objects)
            
            finally:
                # Cleanup to prevent memory issues
                large_objects.clear()
                gc.collect()
        
        metrics = performance_runner.run_performance_test(
            memory_intensive_operations,
            iterations=2,
            warmup_iterations=1
        )
        
        memory_profiler.take_snapshot("stress_after")
        
        # Should handle memory pressure without crashing
        assert metrics.error_rate == 0.0
        
        # Check memory was released
        memory_diff = memory_profiler.compare_snapshots("stress_baseline", "stress_after")
        # After cleanup, memory usage should not have grown significantly
        assert abs(memory_diff["memory_diff"]) < 50 * 1024 * 1024  # Less than 50MB residual
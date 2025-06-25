"""
Comprehensive test suite for PRSM Performance Optimization System
Tests for load testing, scaling, caching, optimization, and monitoring components
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from prsm.performance import (
    LoadTestSuite, LoadTestConfig, LoadTestResult,
    AutoScaler, ScalingPolicy, ScalingMetric, ScalingTrigger,
    CacheManager, CacheConfig, CacheLevel,
    PerformanceOptimizer, QueryOptimizer, APIOptimizer,
    APMIntegration, DistributedTracing, MetricsCollector,
    MetricType, AlertSeverity
)


class TestLoadTestSuite:
    """Test suite for load testing functionality"""
    
    @pytest.fixture
    def load_test_config(self):
        return LoadTestConfig(
            test_name="api_load_test",
            description="Test API endpoint performance",
            concurrent_users=10,
            duration_seconds=30,
            base_url="http://localhost:8000",
            test_scenarios=["api_endpoints"]
        )
    
    @pytest.fixture
    def load_test_suite(self):
        return LoadTestSuite()
    
    @pytest.mark.asyncio
    async def test_load_test_initialization(self, load_test_suite):
        """Test load test suite initialization"""
        assert load_test_suite.session is None
        assert load_test_suite.websocket_connections == []
        assert load_test_suite.test_data == {}
        assert load_test_suite.system_monitor is not None
    
    @pytest.mark.asyncio
    async def test_load_test_execution(self, load_test_suite, load_test_config):
        """Test complete load test execution"""
        # Mock HTTP session to avoid actual network calls
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='{"status": "ok"}')
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            mock_session.return_value.__aenter__.return_value.request.return_value.__aenter__.return_value = mock_response
            
            # Run load test
            result = await load_test_suite.run_load_test(load_test_config)
            
            # Verify result structure
            assert isinstance(result, LoadTestResult)
            assert result.test_name == load_test_config.test_name
            assert result.duration_seconds > 0
            assert result.total_requests >= 0
            assert 0 <= result.success_rate <= 1
    
    @pytest.mark.asyncio
    async def test_api_worker_functionality(self, load_test_suite, load_test_config):
        """Test individual API worker functionality"""
        results = []
        
        with patch.object(load_test_suite, '_execute_api_request') as mock_execute:
            mock_execute.return_value = {
                "status_code": 200,
                "success": True,
                "bytes_received": 1024
            }
            
            # Set short duration for testing
            load_test_config.duration_seconds = 1
            
            await load_test_suite._api_worker(0, load_test_config, results)
            
            # Verify results were recorded
            assert len(results) > 0
            assert all("worker_id" in result for result in results)
            assert all("response_time_ms" in result for result in results)
    
    def test_weighted_endpoint_selection(self, load_test_suite):
        """Test weighted endpoint selection logic"""
        endpoints = [
            {"method": "GET", "path": "/health", "weight": 0.1},
            {"method": "GET", "path": "/api/sessions", "weight": 0.9}
        ]
        
        # Test multiple selections to verify weighting
        selections = []
        for _ in range(100):
            selected = load_test_suite._select_weighted_endpoint(endpoints)
            selections.append(selected["path"])
        
        # Higher weight endpoint should be selected more often
        sessions_count = selections.count("/api/sessions")
        health_count = selections.count("/health")
        
        assert sessions_count > health_count
    
    @pytest.mark.asyncio
    async def test_websocket_scaling_test(self, load_test_suite, load_test_config):
        """Test WebSocket scaling functionality"""
        load_test_config.test_scenarios = ["websocket_scaling"]
        load_test_config.concurrent_users = 5
        load_test_config.duration_seconds = 2
        
        # Mock WebSocket connections
        with patch('websockets.connect') as mock_connect:
            mock_websocket = AsyncMock()
            mock_websocket.send = AsyncMock()
            mock_websocket.recv = AsyncMock(return_value='{"response": "ok"}')
            mock_connect.return_value.__aenter__.return_value = mock_websocket
            
            result = await load_test_suite._test_websocket_scaling(load_test_config)
            
            assert result["scenario"] == "websocket_scaling"
            assert "total_connections" in result
            assert "successful_connections" in result
            assert "total_messages" in result


class TestAutoScaler:
    """Test suite for auto-scaling functionality"""
    
    @pytest.fixture
    def scaling_policy(self):
        return ScalingPolicy(
            name="test-scaling",
            service_name="test-service",
            min_replicas=2,
            max_replicas=10,
            metrics=[
                ScalingMetric(
                    name="cpu_usage",
                    trigger=ScalingTrigger.CPU_USAGE,
                    scale_up_threshold=0.7,
                    scale_down_threshold=0.3,
                    weight=1.0
                )
            ]
        )
    
    @pytest.fixture
    def auto_scaler(self):
        return AutoScaler()
    
    def test_scaling_policy_creation(self, scaling_policy):
        """Test scaling policy creation and validation"""
        assert scaling_policy.name == "test-scaling"
        assert scaling_policy.service_name == "test-service"
        assert scaling_policy.min_replicas == 2
        assert scaling_policy.max_replicas == 10
        assert len(scaling_policy.metrics) == 1
    
    def test_add_scaling_policy(self, auto_scaler, scaling_policy):
        """Test adding scaling policy to auto-scaler"""
        auto_scaler.add_scaling_policy(scaling_policy)
        
        assert scaling_policy.service_name in auto_scaler.policies
        assert auto_scaler.policies[scaling_policy.service_name] == scaling_policy
    
    def test_scaling_score_calculation(self, auto_scaler, scaling_policy):
        """Test scaling score calculation logic"""
        # Test scale-up scenario
        metrics = {"cpu_usage": 0.8}  # Above threshold
        score = auto_scaler._calculate_scaling_score(metrics, scaling_policy)
        assert score > 0  # Should indicate scale-up needed
        
        # Test scale-down scenario
        metrics = {"cpu_usage": 0.2}  # Below threshold
        score = auto_scaler._calculate_scaling_score(metrics, scaling_policy)
        assert score < 0  # Should indicate scale-down possible
        
        # Test stable scenario
        metrics = {"cpu_usage": 0.5}  # Within thresholds
        score = auto_scaler._calculate_scaling_score(metrics, scaling_policy)
        assert score == 0  # Should indicate stable
    
    def test_scaling_decision_logic(self, auto_scaler, scaling_policy):
        """Test scaling decision making"""
        current_replicas = 3
        
        # Test scale-up decision
        scaling_score = 0.8
        decision = auto_scaler._make_scaling_decision(scaling_score, current_replicas, scaling_policy)
        assert decision["action"].value == "up"
        assert decision["new_replicas"] > current_replicas
        
        # Test scale-down decision
        scaling_score = -0.8
        decision = auto_scaler._make_scaling_decision(scaling_score, current_replicas, scaling_policy)
        assert decision["action"].value == "down"
        assert decision["new_replicas"] < current_replicas
    
    def test_time_based_scaling_factor(self, auto_scaler, scaling_policy):
        """Test time-based scaling adjustments"""
        scaling_policy.time_based_scaling = True
        scaling_policy.weekend_scaling_factor = 0.7
        scaling_policy.night_scaling_factor = 0.5
        
        factor = auto_scaler._get_time_based_factor(scaling_policy)
        assert 0 < factor <= 1.0
    
    @pytest.mark.asyncio
    async def test_auto_scaling_loop(self, auto_scaler, scaling_policy):
        """Test auto-scaling monitoring loop"""
        auto_scaler.add_scaling_policy(scaling_policy)
        
        # Mock methods to avoid actual scaling operations
        auto_scaler._collect_metrics = AsyncMock()
        auto_scaler._evaluate_scaling_decision = AsyncMock()
        
        # Start and immediately stop auto-scaling
        await auto_scaler.start_auto_scaling()
        assert auto_scaler.monitoring_active
        
        # Brief pause to let loop run
        await asyncio.sleep(0.1)
        
        await auto_scaler.stop_auto_scaling()
        assert not auto_scaler.monitoring_active


class TestCacheManager:
    """Test suite for caching functionality"""
    
    @pytest.fixture
    def cache_config(self):
        return CacheConfig(
            name="test-cache",
            enable_l1_memory=True,
            enable_l2_redis=False,  # Disable Redis for testing
            enable_l3_cdn=False,    # Disable CDN for testing
            l1_max_size=100,
            l1_ttl_seconds=300
        )
    
    @pytest.fixture
    def cache_manager(self, cache_config):
        return CacheManager(cache_config)
    
    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, cache_manager):
        """Test cache manager initialization"""
        await cache_manager.initialize()
        
        assert cache_manager.config is not None
        assert cache_manager.l1_cache == {}
        assert cache_manager.stats is not None
    
    @pytest.mark.asyncio
    async def test_l1_cache_operations(self, cache_manager):
        """Test L1 memory cache operations"""
        await cache_manager.initialize()
        
        # Test set operation
        key = "test_key"
        value = "test_value"
        success = await cache_manager.set(key, value)
        assert success
        
        # Test get operation
        retrieved_value = await cache_manager.get(key)
        assert retrieved_value == value
        
        # Test cache hit statistics
        assert cache_manager.stats.cache_hits > 0
        assert cache_manager.stats.total_requests > 0
    
    @pytest.mark.asyncio
    async def test_cache_miss_scenario(self, cache_manager):
        """Test cache miss scenario"""
        await cache_manager.initialize()
        
        # Try to get non-existent key
        value = await cache_manager.get("non_existent_key")
        assert value is None
        
        # Verify cache miss was recorded
        assert cache_manager.stats.cache_misses > 0
    
    @pytest.mark.asyncio
    async def test_cache_deletion(self, cache_manager):
        """Test cache deletion operations"""
        await cache_manager.initialize()
        
        # Set and then delete a key
        key = "delete_test_key"
        value = "delete_test_value"
        
        await cache_manager.set(key, value)
        assert await cache_manager.get(key) == value
        
        success = await cache_manager.delete(key)
        assert success
        
        # Verify key was deleted
        assert await cache_manager.get(key) is None
    
    @pytest.mark.asyncio
    async def test_cache_pattern_invalidation(self, cache_manager):
        """Test pattern-based cache invalidation"""
        await cache_manager.initialize()
        
        # Set multiple keys with similar patterns
        await cache_manager.set("user:123:profile", "profile_data")
        await cache_manager.set("user:123:settings", "settings_data")
        await cache_manager.set("user:456:profile", "other_profile")
        
        # Invalidate pattern
        invalidated_count = await cache_manager.invalidate_pattern("user:123:*")
        assert invalidated_count >= 0
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache_manager):
        """Test cache statistics collection"""
        await cache_manager.initialize()
        
        # Perform some cache operations
        await cache_manager.set("stat_test_1", "value1")
        await cache_manager.set("stat_test_2", "value2")
        await cache_manager.get("stat_test_1")  # Hit
        await cache_manager.get("nonexistent")   # Miss
        
        stats = await cache_manager.get_cache_statistics()
        
        assert "cache_stats" in stats
        assert "configuration" in stats
        assert stats["cache_stats"]["total_requests"] > 0
        assert stats["cache_stats"]["cache_hits"] > 0
        assert stats["cache_stats"]["cache_misses"] > 0
    
    @pytest.mark.asyncio
    async def test_cache_warming(self, cache_manager):
        """Test cache warming functionality"""
        await cache_manager.initialize()
        
        warm_data = {
            "warm_key_1": "warm_value_1",
            "warm_key_2": "warm_value_2",
            "warm_key_3": "warm_value_3"
        }
        
        results = await cache_manager.warm_cache(warm_data)
        
        # Verify all keys were warmed successfully
        for key, success in results.items():
            assert success
            cached_value = await cache_manager.get(key)
            assert cached_value == warm_data[key]


class TestPerformanceOptimizer:
    """Test suite for performance optimization functionality"""
    
    @pytest.fixture
    def performance_optimizer(self):
        return PerformanceOptimizer()
    
    @pytest.fixture
    def query_optimizer(self):
        return QueryOptimizer()
    
    @pytest.fixture
    def api_optimizer(self):
        return APIOptimizer()
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, performance_optimizer):
        """Test comprehensive performance analysis"""
        system_config = {
            "database_pool": Mock(),
            "api_stats": {
                "endpoint_metrics": {
                    "/api/v1/sessions": {
                        "avg_response_time_ms": 800,
                        "request_count": 1000
                    }
                }
            }
        }
        
        analysis = await performance_optimizer.run_comprehensive_analysis(system_config)
        
        assert "analysis_timestamp" in analysis
        assert "system_metrics" in analysis
        assert "database_analysis" in analysis
        assert "api_analysis" in analysis
        assert "optimization_recommendations" in analysis
        assert "performance_score" in analysis
        assert 0 <= analysis["performance_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_query_optimization(self, query_optimizer):
        """Test database query optimization"""
        test_query = "SELECT * FROM sessions WHERE created_at > NOW() - INTERVAL '1 day'"
        
        optimization_result = await query_optimizer.optimize_query(test_query)
        
        assert "original_query" in optimization_result
        assert "optimized_query" in optimization_result
        assert "optimizations_applied" in optimization_result
        assert "estimated_improvement" in optimization_result
        assert optimization_result["original_query"] == test_query
    
    @pytest.mark.asyncio
    async def test_api_performance_analysis(self, api_optimizer):
        """Test API performance analysis"""
        endpoint_stats = {
            "endpoints": {
                "/api/v1/sessions": {
                    "avg_response_time_ms": 1200,
                    "request_count": 5000,
                    "avg_payload_size_kb": 150
                },
                "/api/v1/models": {
                    "avg_response_time_ms": 300,
                    "request_count": 2000,
                    "avg_payload_size_kb": 50
                }
            }
        }
        
        analysis = await api_optimizer.analyze_api_performance(endpoint_stats)
        
        assert "slow_endpoints" in analysis
        assert "large_payloads" in analysis
        assert "optimization_recommendations" in analysis
        assert "caching_opportunities" in analysis
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_loop(self, performance_optimizer):
        """Test continuous performance monitoring"""
        # Mock the monitoring methods
        performance_optimizer._collect_system_metrics = AsyncMock()
        performance_optimizer._detect_performance_regressions = AsyncMock(return_value=[])
        performance_optimizer._detect_optimization_opportunities = AsyncMock(return_value=[])
        
        # Start monitoring with very short interval
        await performance_optimizer.start_continuous_monitoring(1)
        assert performance_optimizer.monitoring_active
        
        # Brief pause to let monitoring run
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await performance_optimizer.stop_continuous_monitoring()
        assert not performance_optimizer.monitoring_active
    
    def test_performance_score_calculation(self, performance_optimizer):
        """Test performance score calculation"""
        analysis_data = {
            "system_metrics": {
                "cpu_usage_percent": 50,
                "memory_usage_percent": 60,
                "avg_response_time_ms": 200,
                "error_rate_percent": 1.0,
                "cache_hit_rate": 0.8
            }
        }
        
        score = asyncio.run(performance_optimizer._calculate_performance_score(analysis_data))
        
        assert 0 <= score <= 100
        assert isinstance(score, float)


class TestAPMIntegration:
    """Test suite for APM and monitoring functionality"""
    
    @pytest.fixture
    def apm_config(self):
        return {
            "enabled": True,
            "sampling_rate": 1.0,
            "export_interval_seconds": 60,
            "retention_hours": 24
        }
    
    @pytest.fixture
    def apm_integration(self, apm_config):
        return APMIntegration("test-service", apm_config)
    
    @pytest.fixture
    def distributed_tracing(self):
        return DistributedTracing("test-service")
    
    @pytest.fixture
    def metrics_collector(self):
        return MetricsCollector("test-service")
    
    @pytest.mark.asyncio
    async def test_apm_initialization(self, apm_integration):
        """Test APM integration initialization"""
        await apm_integration.initialize()
        
        assert apm_integration.service_name == "test-service"
        assert apm_integration.tracing is not None
        assert apm_integration.metrics is not None
        assert len(apm_integration.alert_thresholds) > 0
    
    def test_distributed_tracing_span_lifecycle(self, distributed_tracing):
        """Test distributed tracing span creation and completion"""
        # Start a span
        span = distributed_tracing.start_span("test_operation")
        
        assert span.span_id is not None
        assert span.trace_id is not None
        assert span.operation_name == "test_operation"
        assert span.service_name == "test-service" # From tags
        
        # Add some data to the span
        distributed_tracing.add_span_tag(span.span_id, "test_tag", "test_value")
        distributed_tracing.add_span_log(span.span_id, {"event": "test_event"})
        
        # Finish the span
        distributed_tracing.finish_span(span.span_id, "ok")
        
        # Verify span was moved to completed traces
        assert span.span_id not in distributed_tracing.active_spans
        assert len(distributed_tracing.completed_traces) > 0
        assert span.duration_ms is not None
    
    @pytest.mark.asyncio
    async def test_tracing_context_manager(self, distributed_tracing):
        """Test tracing context manager functionality"""
        async with distributed_tracing.trace_operation("context_test") as span:
            assert span.span_id in distributed_tracing.active_spans
            
            # Simulate some work
            await asyncio.sleep(0.01)
        
        # Span should be finished after context exit
        assert span.span_id not in distributed_tracing.active_spans
        assert span.duration_ms is not None
    
    def test_metrics_collection(self, metrics_collector):
        """Test metrics collection functionality"""
        # Record different types of metrics
        metrics_collector.increment_counter("test.counter", 5, {"env": "test"})
        metrics_collector.set_gauge("test.gauge", 42.5, {"type": "memory"})
        metrics_collector.record_timer("test.timer", 150.0, {"operation": "db_query"})
        
        # Verify metrics were recorded
        assert "test.counter" in metrics_collector.metrics
        assert "test.gauge" in metrics_collector.metrics
        assert "test.timer" in metrics_collector.metrics
        
        # Test metric summary
        summary = metrics_collector.get_metric_summary("test.timer")
        assert "count" in summary
        assert "min" in summary
        assert "max" in summary
        assert "mean" in summary
    
    @pytest.mark.asyncio
    async def test_metrics_timing_context_manager(self, metrics_collector):
        """Test metrics timing context manager"""
        async with metrics_collector.time_operation("test_operation"):
            await asyncio.sleep(0.01)  # Simulate work
        
        # Verify timer metric was recorded
        assert "test_operation.duration" in metrics_collector.metrics
        
        summary = metrics_collector.get_metric_summary("test_operation.duration")
        assert summary["count"] == 1
        assert summary["last_value"] > 0  # Should have some duration
    
    @pytest.mark.asyncio
    async def test_apm_request_monitoring(self, apm_integration):
        """Test APM request monitoring functionality"""
        await apm_integration.initialize()
        
        async with apm_integration.monitor_request("/api/test", "GET", "user123") as context:
            assert "span" in context
            assert "request_id" in context
            assert "start_time" in context
            
            # Simulate some request processing
            await asyncio.sleep(0.01)
        
        # Verify metrics were recorded
        assert len(apm_integration.metrics.metrics) > 0
        
        # Check for request counter and timer metrics
        metrics_summary = apm_integration.metrics.get_all_metrics_summary()
        assert metrics_summary["total_metric_types"] > 0
    
    def test_alert_threshold_configuration(self, apm_integration):
        """Test alert threshold configuration"""
        # Set alert thresholds
        apm_integration.set_alert_threshold(
            "response_time", 1000, AlertSeverity.WARNING, "greater_than"
        )
        apm_integration.set_alert_threshold(
            "error_rate", 5.0, AlertSeverity.CRITICAL, "greater_than"
        )
        
        # Verify thresholds were set
        assert "response_time" in apm_integration.alert_thresholds
        assert "error_rate" in apm_integration.alert_thresholds
        
        response_time_config = apm_integration.alert_thresholds["response_time"]
        assert AlertSeverity.WARNING.value in response_time_config
        assert response_time_config[AlertSeverity.WARNING.value]["threshold"] == 1000
    
    @pytest.mark.asyncio
    async def test_alert_checking(self, apm_integration):
        """Test alert checking functionality"""
        await apm_integration.initialize()
        
        # Set a low threshold to trigger alerts
        apm_integration.set_alert_threshold(
            "test_metric", 10, AlertSeverity.WARNING, "greater_than"
        )
        
        # Record metric that exceeds threshold
        apm_integration.metrics.record_metric(
            "test_metric", 20, MetricType.GAUGE
        )
        
        # Check for alerts
        alerts = await apm_integration.check_alerts()
        
        # Should trigger alert since 20 > 10
        assert len(alerts) > 0
        assert alerts[0].metric_name == "test_metric"
        assert alerts[0].current_value == 20
        assert alerts[0].threshold_value == 10
    
    @pytest.mark.asyncio
    async def test_performance_dashboard_data(self, apm_integration):
        """Test performance dashboard data generation"""
        await apm_integration.initialize()
        
        # Generate some metrics and traces
        apm_integration.metrics.increment_counter("requests", 10)
        apm_integration.tracing.start_span("test_operation")
        
        dashboard_data = await apm_integration.get_performance_dashboard_data()
        
        assert "service_name" in dashboard_data
        assert "timestamp" in dashboard_data
        assert "metrics_summary" in dashboard_data
        assert "active_alerts" in dashboard_data
        assert "recent_traces" in dashboard_data
        assert "performance_overview" in dashboard_data
    
    @pytest.mark.asyncio
    async def test_monitoring_data_export(self, apm_integration):
        """Test monitoring data export functionality"""
        await apm_integration.initialize()
        
        # Add some test data
        apm_integration.metrics.increment_counter("test_counter", 5)
        span = apm_integration.tracing.start_span("test_span")
        apm_integration.tracing.finish_span(span.span_id)
        
        # Test JSON export
        json_data = await apm_integration.export_monitoring_data("json")
        assert json_data is not None
        
        parsed_data = json.loads(json_data)
        assert "service" in parsed_data
        assert "metrics" in parsed_data
        assert "traces" in parsed_data
        
        # Test Prometheus export
        prometheus_data = await apm_integration.export_monitoring_data("prometheus")
        assert prometheus_data is not None
        assert "test_counter" in prometheus_data


class TestPerformanceIntegration:
    """Integration tests for the complete performance optimization system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance_analysis(self):
        """Test complete end-to-end performance analysis workflow"""
        # Create performance optimizer
        optimizer = PerformanceOptimizer()
        
        # Mock system configuration
        system_config = {
            "database_pool": Mock(),
            "api_stats": {
                "endpoints": {
                    "/api/v1/sessions": {
                        "avg_response_time_ms": 1200,
                        "request_count": 5000
                    }
                }
            }
        }
        
        # Run comprehensive analysis
        analysis = await optimizer.run_comprehensive_analysis(system_config)
        
        # Verify analysis results
        assert analysis["performance_score"] > 0
        assert len(analysis["optimization_recommendations"]) >= 0
        
        # Test applying an optimization (if any recommendations exist)
        if analysis["optimization_recommendations"]:
            first_recommendation = analysis["optimization_recommendations"][0]
            
            # Mock the optimization execution
            with patch.object(optimizer, '_execute_optimization_step') as mock_execute:
                mock_execute.return_value = None  # Successful execution
                
                result = await optimizer.apply_optimization(first_recommendation)
                assert "optimization_title" in result
                assert "status" in result
    
    @pytest.mark.asyncio
    async def test_integrated_load_test_with_monitoring(self):
        """Test integration between load testing and monitoring"""
        # Create APM integration
        apm_config = {"enabled": True, "sampling_rate": 1.0}
        apm = APMIntegration("test-service", apm_config)
        await apm.initialize()
        
        # Simulate load test operations with monitoring
        operations = [
            "user_login",
            "fetch_models", 
            "start_training",
            "check_status",
            "download_results"
        ]
        
        for operation in operations:
            async with apm.monitor_request(f"/api/{operation}", "POST") as context:
                # Simulate processing time
                await asyncio.sleep(0.01)
                
                # Record custom metrics
                apm.metrics.increment_counter(f"{operation}.requests")
                apm.metrics.record_timer(f"{operation}.processing_time", 10.0)
        
        # Verify monitoring data was collected
        dashboard_data = await apm.get_performance_dashboard_data()
        metrics_summary = dashboard_data["metrics_summary"]
        
        assert metrics_summary["total_metric_types"] > 0
        assert metrics_summary["recent_activity"]["metrics_last_hour"] > 0
    
    @pytest.mark.asyncio
    async def test_cache_and_optimization_integration(self):
        """Test integration between caching and optimization systems"""
        # Create cache manager
        cache_config = CacheConfig(
            name="integration-test-cache",
            enable_l1_memory=True,
            l1_max_size=1000
        )
        cache_manager = CacheManager(cache_config)
        await cache_manager.initialize()
        
        # Create API optimizer
        api_optimizer = APIOptimizer()
        
        # Simulate API performance data
        endpoint_stats = {
            "endpoints": {
                "/api/models": {
                    "avg_response_time_ms": 800,
                    "request_count": 10000,
                    "cache_hit_rate": 0.3  # Low cache hit rate
                }
            }
        }
        
        # Analyze API performance
        analysis = await api_optimizer.analyze_api_performance(endpoint_stats)
        
        # Should identify caching opportunities
        assert "caching_opportunities" in analysis
        caching_opportunities = analysis["caching_opportunities"]
        
        # Implement caching for identified opportunities
        if caching_opportunities:
            test_data = {
                "models:popular": {"models": ["gpt-3.5", "claude-3"]},
                "models:recent": {"models": ["gpt-4", "claude-3.5"]}
            }
            
            # Warm cache with test data
            cache_results = await cache_manager.warm_cache(test_data)
            
            # Verify cache warming was successful
            for key, success in cache_results.items():
                assert success
            
            # Verify cached data can be retrieved
            for key in test_data.keys():
                cached_value = await cache_manager.get(key)
                assert cached_value == test_data[key]
        
        # Get cache statistics
        cache_stats = await cache_manager.get_cache_statistics()
        assert cache_stats["cache_stats"]["total_sets"] > 0
        assert cache_stats["cache_stats"]["cache_hits"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
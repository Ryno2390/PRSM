"""
Integration tests for the PRSM metrics system.

Tests cover initialization, metric recording, middleware wiring,
and the /health/metrics endpoint registry fix.
"""
import os
import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from decimal import Decimal


class TestMetricsInitialization:
    """Tests for metrics system initialization"""

    def setup_method(self):
        """Reset global metrics collector before each test"""
        import prsm.compute.performance.metrics as metrics_module
        self._original_collector = metrics_module.metrics_collector
        metrics_module.metrics_collector = None

    def teardown_method(self):
        """Restore global metrics collector after each test"""
        import prsm.compute.performance.metrics as metrics_module
        # Stop any running collection task
        if metrics_module.metrics_collector and metrics_module.metrics_collector.collection_active:
            metrics_module.metrics_collector.collection_active = False
        metrics_module.metrics_collector = self._original_collector

    def test_initialize_metrics_creates_collector(self):
        """initialize_metrics() should create a MetricsCollector"""
        from prsm.compute.performance.metrics import (
            initialize_metrics, get_metrics_collector, MetricsConfig
        )
        config = MetricsConfig(
            service_name="test-service",
            enable_prometheus=True,
            prometheus_port=None,
            store_in_redis=False,
            collect_system_metrics=False,
            collect_process_metrics=False,
            collect_runtime_metrics=False,
        )
        initialize_metrics(config, redis_client=None)
        collector = get_metrics_collector()
        assert collector is not None
        assert collector.config.service_name == "test-service"

    def test_get_metrics_collector_raises_when_not_initialized(self):
        """get_metrics_collector() should raise RuntimeError if not initialized"""
        from prsm.compute.performance.metrics import get_metrics_collector
        with pytest.raises(RuntimeError, match="not initialized"):
            get_metrics_collector()

    def test_increment_counter_is_noop_when_not_initialized(self):
        """increment_counter() should not raise when metrics_collector is None"""
        from prsm.compute.performance.metrics import increment_counter
        increment_counter("some_counter", amount=1)

    def test_set_gauge_is_noop_when_not_initialized(self):
        """set_gauge() should not raise when metrics_collector is None"""
        from prsm.compute.performance.metrics import set_gauge
        set_gauge("some_gauge", value=42.0)

    def test_observe_histogram_is_noop_when_not_initialized(self):
        """observe_histogram() should not raise when metrics_collector is None"""
        from prsm.compute.performance.metrics import observe_histogram
        observe_histogram("some_histogram", value=0.123)


class TestMetricsRecording:
    """Tests for metric recording after initialization"""

    def setup_method(self):
        """Initialize metrics with a test config"""
        import prsm.compute.performance.metrics as metrics_module
        self._original_collector = metrics_module.metrics_collector

        from prsm.compute.performance.metrics import initialize_metrics, MetricsConfig
        config = MetricsConfig(
            service_name="test-recording",
            enable_prometheus=True,
            prometheus_port=None,
            store_in_redis=False,
            collect_system_metrics=False,
            collect_process_metrics=False,
            collect_runtime_metrics=False,
        )
        initialize_metrics(config, redis_client=None)

    def teardown_method(self):
        import prsm.compute.performance.metrics as metrics_module
        metrics_module.metrics_collector = self._original_collector

    def test_record_custom_metric_stores_value(self):
        """record_metric() should store values in metric history"""
        from prsm.compute.performance.metrics import get_metrics_collector, record_metric
        record_metric("test_custom_metric", 99.5)
        collector = get_metrics_collector()
        assert "test_custom_metric" in collector.metric_history
        assert collector.metric_history["test_custom_metric"][-1].value == 99.5

    def test_http_requests_counter_registered_on_init(self):
        """Default Prometheus metrics should include http_requests_total"""
        from prsm.compute.performance.metrics import get_metrics_collector
        collector = get_metrics_collector()
        assert "http_requests_total" in collector.prometheus_metrics

    def test_http_duration_histogram_registered_on_init(self):
        """Default Prometheus metrics should include http_request_duration_seconds"""
        from prsm.compute.performance.metrics import get_metrics_collector
        collector = get_metrics_collector()
        assert "http_request_duration_seconds" in collector.prometheus_metrics


class TestHealthMetricsEndpoint:
    """Tests for the /health/metrics endpoint registry fix"""

    def test_metrics_endpoint_uses_collector_registry_when_initialized(self):
        """Endpoint should return MetricsCollector output, not default registry"""
        mock_collector = MagicMock()
        mock_collector.get_prometheus_metrics.return_value = "# PRSM custom metrics\n"

        with patch(
            "prsm.compute.performance.metrics.get_metrics_collector",
            return_value=mock_collector
        ):
            from prsm.interface.api.health_api import prometheus_metrics
            import asyncio
            response = asyncio.run(prometheus_metrics())
            assert "PRSM custom metrics" in response.body.decode()

    def test_metrics_endpoint_falls_back_to_default_registry_when_not_initialized(self):
        """Endpoint should not raise 500 when MetricsCollector is not initialized"""
        with patch(
            "prsm.compute.performance.metrics.get_metrics_collector",
            side_effect=RuntimeError("Metrics system not initialized")
        ):
            from prsm.interface.api.health_api import prometheus_metrics
            import asyncio
            response = asyncio.run(prometheus_metrics())
            assert response.status_code == 200

"""Tests for Tier 3: Python SDK client and dashboard metrics."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from prsm.sdk import PRSMClient
from prsm.observability.dashboard_metrics import DashboardMetrics, RingStatus


class TestPRSMClient:
    def test_client_creation(self):
        client = PRSMClient("http://localhost:8000")
        assert client.base_url == "http://localhost:8000"
        assert client.api_key == ""

    def test_client_with_api_key(self):
        client = PRSMClient("http://localhost:8000", api_key="prsm_test123")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer prsm_test123"

    def test_client_strips_trailing_slash(self):
        client = PRSMClient("http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_quote_works_locally(self):
        """Quote uses local pricing engine, no network needed."""
        import asyncio
        client = PRSMClient()

        async def _test():
            quote = await client.quote("test query", shards=3, tier="t2")
            assert "total" in quote
            assert float(quote["total"]) > 0

        asyncio.run(_test())


class TestDashboardMetrics:
    def test_metrics_without_node(self):
        metrics = DashboardMetrics(node=None)
        summary = metrics.get_summary()
        assert summary["rings_total"] == 10
        assert summary["rings_initialized"] == 0

    def test_metrics_with_mock_node(self):
        node = MagicMock()
        node.agent_dispatcher = MagicMock()
        node.agent_executor = MagicMock()
        node.swarm_coordinator = MagicMock()
        node.pricing_engine = MagicMock()
        node.pricing_engine.spot.multiplier = "1.0"
        node.pricing_engine.spot.network_utilization = 0.5
        node.prosumer_manager = MagicMock()
        node.agent_forge = MagicMock()
        node.agent_forge.traces = [1, 2, 3]  # 3 traces
        node.confidential_executor = MagicMock()
        node.tensor_executor = MagicMock()
        node.nwtn_model_service = MagicMock()
        node.integrity_verifier = MagicMock()
        node.privacy_budget = MagicMock()
        node.privacy_budget.get_audit_report.return_value = {"total_spent": 16.0}

        metrics = DashboardMetrics(node=node)
        summary = metrics.get_summary()

        assert summary["rings_initialized"] == 10
        assert summary["forge"]["traces_collected"] == 3
        assert len(summary["rings"]) == 10

    def test_ring_status_dataclass(self):
        rs = RingStatus(ring_number=1, name="The Sandbox", initialized=True, healthy=True)
        assert rs.ring_number == 1
        assert rs.initialized

    def test_collect_ring_status_returns_10(self):
        node = MagicMock()
        # Set all components to None to test partial initialization
        for attr in ['agent_dispatcher', 'agent_executor', 'swarm_coordinator',
                     'pricing_engine', 'prosumer_manager', 'agent_forge',
                     'confidential_executor', 'tensor_executor',
                     'nwtn_model_service', 'integrity_verifier', 'privacy_budget']:
            setattr(node, attr, None)

        metrics = DashboardMetrics(node=node)
        rings = metrics.collect_ring_status()
        assert len(rings) == 10

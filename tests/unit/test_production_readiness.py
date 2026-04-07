"""Tests for production readiness items."""

import pytest
import json

from prsm.data.embedding_sharder import EmbeddingSharder, _simple_embedding
from prsm.observability.health_monitor import HealthMonitor


class TestEmbeddingSharder:
    def test_simple_embedding(self):
        emb = _simple_embedding("hello world", dim=16)
        assert len(emb) == 16
        # Normalized
        norm = sum(v*v for v in emb) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_shard_json_records(self):
        records = [
            {"state": "NC", "type": "EV", "count": 100},
            {"state": "NC", "type": "ICE", "count": 200},
            {"state": "CA", "type": "EV", "count": 300},
            {"state": "CA", "type": "ICE", "count": 400},
        ]
        sharder = EmbeddingSharder()
        manifest = sharder.shard_records(records, "test-ds", n_shards=2)
        assert manifest.dataset_id == "test-ds"
        assert manifest.total_records == 4
        assert len(manifest.shards) == 2
        assert all(len(s.centroid) == 32 for s in manifest.shards)

    def test_shard_file_csv(self):
        csv_data = b"state,type,count\nNC,EV,100\nNC,ICE,200\nCA,EV,300\nCA,ICE,400"
        sharder = EmbeddingSharder()
        manifest = sharder.shard_file(csv_data, "csv-test", n_shards=2)
        assert manifest.total_records == 4
        assert len(manifest.shards) >= 1

    def test_shard_empty_data(self):
        sharder = EmbeddingSharder()
        manifest = sharder.shard_records([], "empty", n_shards=4)
        assert manifest.total_records == 0

    def test_shards_have_keywords(self):
        records = [{"city": "Raleigh", "state": "NC"} for _ in range(10)]
        sharder = EmbeddingSharder()
        manifest = sharder.shard_records(records, "kw-test", n_shards=2)
        all_keywords = []
        for s in manifest.shards:
            all_keywords.extend(s.keywords)
        assert len(all_keywords) > 0


class TestHealthMonitor:
    def test_creation(self):
        monitor = HealthMonitor()
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_check_health_no_node(self):
        monitor = HealthMonitor(node=None)
        checks = await monitor.check_health()
        assert len(checks["alerts"]) > 0

    @pytest.mark.asyncio
    async def test_check_health_with_mock_node(self):
        from unittest.mock import MagicMock
        node = MagicMock()
        node.transport.peer_count = 5
        node.privacy_budget.remaining = 50.0
        # Mock the rings
        node.agent_dispatcher = MagicMock()
        node.agent_executor = MagicMock()
        node.swarm_coordinator = MagicMock()
        node.pricing_engine = MagicMock()
        node.pricing_engine.spot.multiplier = "1.0"
        node.pricing_engine.spot.network_utilization = 0.5
        node.prosumer_manager = MagicMock()
        node.agent_forge = MagicMock()
        node.agent_forge.traces = []
        node.confidential_executor = MagicMock()
        node.tensor_executor = MagicMock()
        node.nwtn_model_service = MagicMock()
        node.integrity_verifier = MagicMock()
        node.privacy_budget = MagicMock()
        node.privacy_budget.remaining = 50.0
        node.privacy_budget.get_audit_report.return_value = {"total_spent": 0}
        node.pipeline_audit_log = MagicMock()

        monitor = HealthMonitor(node=node)
        checks = await monitor.check_health()
        assert checks["peers_ok"]
        assert checks["peer_count"] == 5

    @pytest.mark.asyncio
    async def test_alert_callback(self):
        alerts_received = []
        monitor = HealthMonitor()
        monitor.on_alert(lambda checks: alerts_received.append(checks))
        await monitor.check_health()
        assert len(alerts_received) > 0  # No node = alert


class TestTLSConfig:
    def test_tls_fields_exist(self):
        from prsm.node.config import NodeConfig
        config = NodeConfig()
        assert hasattr(config, 'tls_enabled')
        assert hasattr(config, 'tls_cert_path')
        assert config.tls_enabled is False

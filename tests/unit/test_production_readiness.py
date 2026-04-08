"""Tests for production readiness items."""

import pytest
import json

from prsm.data.embedding_sharder import EmbeddingSharder, _simple_embedding
from prsm.observability.health_monitor import HealthMonitor


class TestEmbeddingSharder:
    def test_simple_embedding(self):
        emb = _simple_embedding("hello world")
        assert len(emb) > 0
        # Normalized
        norm = sum(v*v for v in emb) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_simple_embedding_custom_dim(self):
        emb = _simple_embedding("hello world", dim=16)
        assert len(emb) == 16
        norm = sum(v*v for v in emb) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_embedding_provider_loads(self):
        from prsm.data.embedding_sharder import EmbeddingProvider
        provider = EmbeddingProvider()
        vec = provider.embed("test text")
        assert vec.shape[0] > 0
        # Should be unit-normalized
        norm = float((vec ** 2).sum() ** 0.5)
        assert abs(norm - 1.0) < 0.01
        assert provider.backend in ("sentence_transformers", "transformers", "hash")

    def test_embedding_provider_batch(self):
        from prsm.data.embedding_sharder import EmbeddingProvider
        provider = EmbeddingProvider()
        vecs = provider.embed_batch(["hello", "world", "test"])
        assert vecs.shape[0] == 3
        assert vecs.shape[1] == provider.dimension

    def test_different_texts_different_embeddings(self):
        from prsm.data.embedding_sharder import EmbeddingProvider
        provider = EmbeddingProvider()
        v1 = provider.embed("quantum physics research paper")
        v2 = provider.embed("chocolate cake recipe")
        # Different texts should produce different embeddings
        cos_sim = float(v1 @ v2)
        assert cos_sim < 0.99  # Not identical

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
        assert len(manifest.shards) >= 1
        assert all(len(s.centroid) == sharder.provider.dimension for s in manifest.shards)

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

    def test_kmeans_clusters_semantically(self):
        """Records about similar topics should land in the same shard."""
        from prsm.data.embedding_sharder import _kmeans, EmbeddingProvider
        import numpy as np

        provider = EmbeddingProvider()
        texts = [
            "machine learning neural network",
            "deep learning transformer model",
            "chocolate cake recipe baking",
            "cookie dessert sugar flour",
        ]
        vecs = provider.embed_batch(texts)
        labels, centroids = _kmeans(vecs, k=2)

        # Verify k-means produces valid output regardless of backend
        assert len(labels) == 4
        assert centroids.shape[0] == 2
        assert set(labels.tolist()).issubset({0, 1})

        # Semantic assertions only hold with real embeddings (not hash fallback)
        if provider.backend != "hash":
            assert labels[0] == labels[1], "ML texts should cluster together"
            assert labels[2] == labels[3], "Food texts should cluster together"
            assert labels[0] != labels[2], "ML and food should be in different clusters"


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

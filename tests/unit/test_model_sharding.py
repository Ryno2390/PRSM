"""Tests for Ring 8 model sharding data models and sharder."""

import hashlib
import time

import numpy as np
import pytest

from prsm.compute.model_sharding.models import (
    ModelShard,
    PipelineConfig,
    PipelineStakeTier,
    ShardedModel,
)
from prsm.compute.model_sharding.sharder import ModelSharder


# ── PipelineStakeTier ──────────────────────────────────────────────


class TestPipelineStakeTier:
    def test_open_tier(self):
        tier = PipelineStakeTier.OPEN
        assert tier.label == "open"
        assert tier.stake_required == 0
        assert tier.slash_rate == 0.0

    def test_standard_tier(self):
        tier = PipelineStakeTier.STANDARD
        assert tier.label == "standard"
        assert tier.stake_required == 5000
        assert tier.slash_rate == 0.5

    def test_premium_tier(self):
        tier = PipelineStakeTier.PREMIUM
        assert tier.label == "premium"
        assert tier.stake_required == 25000
        assert tier.slash_rate == 1.0

    def test_critical_tier(self):
        tier = PipelineStakeTier.CRITICAL
        assert tier.label == "critical"
        assert tier.stake_required == 50000
        assert tier.slash_rate == 1.0

    def test_four_members(self):
        assert len(PipelineStakeTier) == 4


# ── ModelShard ─────────────────────────────────────────────────────


class TestModelShard:
    def _make_shard(self) -> ModelShard:
        return ModelShard(
            shard_id="s-001",
            model_id="m-abc",
            shard_index=0,
            total_shards=4,
            tensor_data=b"\x01\x02\x03",
            tensor_shape=(3,),
            layer_range=(0, 12),
            size_bytes=3,
            checksum="abc123",
        )

    def test_creation(self):
        shard = self._make_shard()
        assert shard.shard_id == "s-001"
        assert shard.model_id == "m-abc"
        assert shard.shard_index == 0
        assert shard.total_shards == 4
        assert shard.tensor_data == b"\x01\x02\x03"
        assert shard.tensor_shape == (3,)
        assert shard.layer_range == (0, 12)
        assert shard.size_bytes == 3
        assert shard.checksum == "abc123"

    def test_to_dict_roundtrip(self):
        original = self._make_shard()
        d = original.to_dict()
        restored = ModelShard.from_dict(d)
        assert restored.shard_id == original.shard_id
        assert restored.model_id == original.model_id
        assert restored.shard_index == original.shard_index
        assert restored.total_shards == original.total_shards
        assert restored.tensor_data == original.tensor_data
        assert restored.tensor_shape == original.tensor_shape
        assert restored.layer_range == original.layer_range
        assert restored.size_bytes == original.size_bytes
        assert restored.checksum == original.checksum

    def test_defaults(self):
        shard = ModelShard(
            shard_id="s-002",
            model_id="m-xyz",
            shard_index=1,
            total_shards=2,
            tensor_data=b"",
            tensor_shape=(0,),
        )
        assert shard.layer_range == (0, 0)
        assert shard.size_bytes == 0
        assert shard.checksum == ""


# ── ShardedModel ───────────────────────────────────────────────────


class TestShardedModel:
    def test_creation(self):
        model = ShardedModel(
            model_id="m-abc",
            model_name="test-model",
            total_shards=2,
        )
        assert model.model_id == "m-abc"
        assert model.model_name == "test-model"
        assert model.total_shards == 2
        assert model.shards == []
        assert model.stake_tier == PipelineStakeTier.STANDARD
        assert isinstance(model.created_at, float)

    def test_get_shard_by_index(self):
        shard_0 = ModelShard("s0", "m1", 0, 2, b"\x00", (1,))
        shard_1 = ModelShard("s1", "m1", 1, 2, b"\x01", (1,))
        model = ShardedModel("m1", "test", 2, shards=[shard_0, shard_1])
        assert model.get_shard_by_index(0) is shard_0
        assert model.get_shard_by_index(1) is shard_1
        assert model.get_shard_by_index(99) is None

    def test_total_size_bytes(self):
        shard_0 = ModelShard("s0", "m1", 0, 2, b"", (1,), size_bytes=100)
        shard_1 = ModelShard("s1", "m1", 1, 2, b"", (1,), size_bytes=200)
        model = ShardedModel("m1", "test", 2, shards=[shard_0, shard_1])
        assert model.total_size_bytes == 300


# ── PipelineConfig ─────────────────────────────────────────────────


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.parallelism_degree == 4
        assert config.min_pool_size == 20
        assert config.require_tee is False
        assert config.privacy_level == "standard"
        assert config.stake_tier == PipelineStakeTier.STANDARD
        assert config.enable_diversified_pipeline is False
        assert config.max_latency_ms == 5000

    def test_custom_values(self):
        config = PipelineConfig(
            parallelism_degree=8,
            min_pool_size=50,
            require_tee=True,
            privacy_level="high",
            stake_tier=PipelineStakeTier.CRITICAL,
            enable_diversified_pipeline=True,
            max_latency_ms=2000,
        )
        assert config.parallelism_degree == 8
        assert config.min_pool_size == 50
        assert config.require_tee is True
        assert config.privacy_level == "high"
        assert config.stake_tier == PipelineStakeTier.CRITICAL
        assert config.enable_diversified_pipeline is True
        assert config.max_latency_ms == 2000


# ── ModelSharder ───────────────────────────────────────────────────


class TestModelSharder:
    def test_shard_tensor_splits_correctly(self):
        tensor = np.arange(12).reshape(4, 3)
        shards = ModelSharder.shard_tensor(tensor, 2)
        assert len(shards) == 2
        # Largest dim is axis 0 (size 4), so each shard has 2 rows
        assert shards[0].shape == (2, 3)
        assert shards[1].shape == (2, 3)

    def test_reassemble_tensor_recovers_original(self):
        tensor = np.random.rand(8, 4)
        shards = ModelSharder.shard_tensor(tensor, 4)
        reassembled = ModelSharder.reassemble_tensor(shards, axis=0)
        np.testing.assert_array_equal(reassembled, tensor)

    def test_shard_tensor_uneven_split(self):
        tensor = np.arange(10)
        shards = ModelSharder.shard_tensor(tensor, 3)
        assert len(shards) == 3
        total_elements = sum(s.size for s in shards)
        assert total_elements == 10

    def test_shard_model_creates_correct_shards(self):
        weights = {
            "layer1": np.random.rand(8, 4),
            "layer2": np.random.rand(6, 3),
        }
        model = ModelSharder.shard_model("m1", "test-model", weights, 4)
        assert model.model_id == "m1"
        assert model.model_name == "test-model"
        assert model.total_shards == 4
        # 2 layers x 4 shards each = 8 shard objects
        assert len(model.shards) == 8

    def test_each_shard_has_checksum(self):
        weights = {"w": np.random.rand(4, 4)}
        model = ModelSharder.shard_model("m1", "test", weights, 2)
        for shard in model.shards:
            assert shard.checksum != ""
            assert len(shard.checksum) == 64  # SHA-256 hex length
            # Verify checksum matches data
            expected = hashlib.sha256(shard.tensor_data).hexdigest()
            assert shard.checksum == expected

    def test_shard_model_size_bytes(self):
        weights = {"w": np.zeros((4, 4), dtype=np.float64)}
        model = ModelSharder.shard_model("m1", "test", weights, 2)
        for shard in model.shards:
            assert shard.size_bytes > 0
            assert shard.size_bytes == len(shard.tensor_data)

"""Tests for Swarm Job data models."""

import pytest
import math
from prsm.compute.swarm.models import (
    SwarmJob,
    SwarmResult,
    SwarmStatus,
    MapReduceStrategy,
    ReduceLocation,
    ShardAssignment,
)
from prsm.compute.agents.models import AgentManifest


class TestMapReduceStrategy:
    def test_default_strategy(self):
        strategy = MapReduceStrategy()
        assert strategy.reduce_location == ReduceLocation.ORIGIN
        assert strategy.quorum_pct == 0.8
        assert strategy.per_shard_timeout == 60

    def test_custom_strategy(self):
        strategy = MapReduceStrategy(
            reduce_location=ReduceLocation.AGGREGATOR,
            quorum_pct=0.5,
            per_shard_timeout=120,
            global_timeout=600,
        )
        assert strategy.reduce_location == ReduceLocation.AGGREGATOR
        assert strategy.quorum_pct == 0.5


class TestSwarmJob:
    def test_job_creation(self):
        job = SwarmJob(
            job_id="swarm-001",
            query="EV adoption trends",
            shard_cids=["QmA", "QmB", "QmC"],
            wasm_binary=b"\x00asm\x01\x00\x00\x00",
            agent_manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=10.0,
        )
        assert job.job_id == "swarm-001"
        assert len(job.shard_cids) == 3
        assert job.status == SwarmStatus.PENDING

    def test_job_budget_per_shard(self):
        job = SwarmJob(
            job_id="swarm-002",
            query="test",
            shard_cids=["QmA", "QmB", "QmC", "QmD"],
            wasm_binary=b"\x00asm\x01\x00\x00\x00",
            agent_manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=20.0,
        )
        assert job.budget_per_shard == 5.0

    def test_job_quorum_count(self):
        job = SwarmJob(
            job_id="swarm-003",
            query="test",
            shard_cids=[f"Qm{i}" for i in range(10)],
            wasm_binary=b"\x00asm\x01\x00\x00\x00",
            agent_manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=50.0,
            strategy=MapReduceStrategy(quorum_pct=0.8),
        )
        assert job.quorum_count == 8

    def test_job_is_quorum_met(self):
        job = SwarmJob(
            job_id="swarm-004",
            query="test",
            shard_cids=["QmA", "QmB"],
            wasm_binary=b"\x00asm\x01\x00\x00\x00",
            agent_manifest=AgentManifest(required_cids=[], min_hardware_tier="t1"),
            budget_ftns=10.0,
            strategy=MapReduceStrategy(quorum_pct=0.5),
        )
        job.completed_shards.append("QmA")
        assert job.is_quorum_met()


class TestSwarmResult:
    def test_result_creation(self):
        result = SwarmResult(
            job_id="swarm-001",
            shard_results={"QmA": {"count": 42}, "QmB": {"count": 58}},
            total_pcu=1.5,
            total_ftns_spent=5.0,
        )
        assert len(result.shard_results) == 2
        assert result.shards_completed == 2

    def test_result_success_rate(self):
        result = SwarmResult(
            job_id="swarm-002",
            shard_results={"QmA": {"v": 1}, "QmB": {"v": 2}},
            total_shards=5,
            total_pcu=2.0,
            total_ftns_spent=8.0,
        )
        assert result.success_rate == 0.4


class TestShardAssignment:
    def test_assignment_creation(self):
        assignment = ShardAssignment(
            shard_cid="QmTestShard",
            agent_id="agent-001",
            provider_id="provider-abc",
        )
        assert assignment.shard_cid == "QmTestShard"
        assert assignment.status == "dispatched"

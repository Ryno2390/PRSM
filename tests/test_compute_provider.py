"""
Tests for prsm.node.compute_provider — job acceptance, execution, and result signing.
"""

import asyncio
import json

import pytest

from prsm.node.compute_provider import (
    ComputeJob,
    ComputeProvider,
    JobStatus,
    JobType,
    SystemResources,
    _cpu_benchmark,
    detect_resources,
)
from prsm.node.gossip import GOSSIP_JOB_OFFER, GOSSIP_JOB_RESULT, GossipProtocol
from prsm.node.identity import generate_node_identity, verify_signature
from prsm.node.local_ledger import LocalLedger, TransactionType
from prsm.node.transport import WebSocketTransport


class TestDetectResources:
    def test_detect_returns_resources(self):
        resources = detect_resources()
        assert resources.cpu_count >= 1
        assert resources.memory_total_gb >= 0

    def test_cpu_benchmark(self):
        count = _cpu_benchmark(100)
        assert count == 25  # primes below 100


class TestComputeProvider:
    @pytest.fixture
    async def setup(self):
        identity = generate_node_identity("provider")
        transport = WebSocketTransport(identity, host="127.0.0.1", port=19300)
        gossip = GossipProtocol(transport, fanout=3, heartbeat_interval=9999)
        ledger = LocalLedger(":memory:")
        await ledger.initialize()
        await ledger.create_wallet(identity.node_id, "provider")
        await ledger.create_wallet("system")

        provider = ComputeProvider(
            identity=identity,
            transport=transport,
            gossip=gossip,
            ledger=ledger,
            max_concurrent_jobs=3,
        )

        yield provider, identity, ledger
        await ledger.close()

    @pytest.mark.asyncio
    async def test_available_capacity(self, setup):
        provider, identity, ledger = setup
        capacity = provider.available_capacity
        assert capacity["concurrent_slots"] == 3
        assert capacity["active_jobs"] == 0
        assert capacity["cpu_cores_allocated"] >= 1

    @pytest.mark.asyncio
    async def test_execute_benchmark_job(self, setup):
        provider, identity, ledger = setup
        await provider.start()

        job = ComputeJob(
            job_id="test-job-1",
            job_type=JobType.BENCHMARK,
            requester_id="other-node",
            payload={"iterations": 1000},
            ftns_budget=5.0,
            status=JobStatus.RUNNING,
        )
        provider.active_jobs[job.job_id] = job

        await provider._execute_job(job)

        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        assert "primes_found" in job.result
        assert "elapsed_seconds" in job.result
        assert job.result_signature is not None

        # Verify result signature
        result_bytes = json.dumps(job.result, sort_keys=True).encode()
        assert verify_signature(identity.public_key_b64, result_bytes, job.result_signature)

        # Check earnings were recorded
        balance = await ledger.get_balance(identity.node_id)
        assert balance == 5.0

    @pytest.mark.asyncio
    async def test_execute_inference_job(self, setup):
        provider, identity, ledger = setup
        await provider.start()

        job = ComputeJob(
            job_id="test-job-2",
            job_type=JobType.INFERENCE,
            requester_id="other-node",
            payload={"prompt": "What is PRSM?", "model": "local"},
            ftns_budget=2.0,
            status=JobStatus.RUNNING,
        )
        provider.active_jobs[job.job_id] = job

        await provider._execute_job(job)

        assert job.status == JobStatus.COMPLETED
        assert "response" in job.result
        assert identity.node_id[:8] in job.result["response"]

    @pytest.mark.asyncio
    async def test_execute_embedding_job(self, setup):
        provider, identity, ledger = setup
        await provider.start()

        job = ComputeJob(
            job_id="test-job-3",
            job_type=JobType.EMBEDDING,
            requester_id="other-node",
            payload={"text": "hello world", "dimensions": 64},
            ftns_budget=1.0,
            status=JobStatus.RUNNING,
        )
        provider.active_jobs[job.job_id] = job

        await provider._execute_job(job)

        assert job.status == JobStatus.COMPLETED
        assert "embedding" in job.result
        assert len(job.result["embedding"]) == 64

    @pytest.mark.asyncio
    async def test_job_offer_from_self_ignored(self, setup):
        provider, identity, ledger = setup
        await provider.start()

        # Simulate receiving our own job offer
        await provider._on_job_offer(
            GOSSIP_JOB_OFFER,
            {
                "job_id": "self-job",
                "job_type": "benchmark",
                "requester_id": identity.node_id,  # our own ID
                "payload": {},
                "ftns_budget": 1.0,
            },
            identity.node_id,
        )

        assert len(provider.active_jobs) == 0

    @pytest.mark.asyncio
    async def test_max_concurrent_jobs_respected(self, setup):
        provider, identity, ledger = setup
        provider.max_concurrent_jobs = 1
        await provider.start()

        # Fill up the slot
        provider.active_jobs["existing"] = ComputeJob(
            job_id="existing",
            job_type=JobType.BENCHMARK,
            requester_id="other",
            payload={},
            ftns_budget=1.0,
            status=JobStatus.RUNNING,
        )

        # New job should be rejected (capacity full)
        await provider._on_job_offer(
            GOSSIP_JOB_OFFER,
            {
                "job_id": "new-job",
                "job_type": "benchmark",
                "requester_id": "another-node",
                "payload": {},
                "ftns_budget": 1.0,
            },
            "another-node",
        )

        assert "new-job" not in provider.active_jobs

    @pytest.mark.asyncio
    async def test_get_stats(self, setup):
        provider, identity, ledger = setup
        stats = provider.get_stats()
        assert "resources" in stats
        assert "allocation" in stats
        assert "capacity" in stats
        assert "active_jobs" in stats

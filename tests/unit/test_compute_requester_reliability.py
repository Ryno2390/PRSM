"""Tests for ComputeRequester reliability recording."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from prsm.node.compute_requester import ComputeRequester, JobType, SubmittedJob


def _make_requester():
    identity = MagicMock()
    identity.node_id = "requester_node"
    transport = MagicMock()
    gossip = MagicMock()
    gossip.subscribe = MagicMock()
    gossip.publish = AsyncMock(return_value=1)
    ledger = MagicMock()
    ledger.get_balance = AsyncMock(return_value=1000.0)
    ledger.transfer = AsyncMock()

    discovery = MagicMock()
    discovery.record_job_success = MagicMock()
    discovery.record_job_failure = MagicMock()

    req = ComputeRequester(
        identity=identity,
        transport=transport,
        gossip=gossip,
        ledger=ledger,
        discovery=discovery,
    )
    req.escrow = None
    req.ledger_sync = None
    return req


class TestReliabilityRecording:

    @pytest.mark.asyncio
    async def test_successful_result_records_success(self):
        req = _make_requester()
        req._running = True

        job = SubmittedJob(
            job_id="job_001",
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=0.0,
        )
        req.submitted_jobs["job_001"] = job

        await req._on_job_result("job_result", {
            "job_id": "job_001",
            "provider_id": "requester_node",
            "status": "completed",
            "result": {"output": "hello"},
        }, "requester_node")

        req.discovery.record_job_success.assert_called_once_with("requester_node")

    @pytest.mark.asyncio
    async def test_failed_result_records_failure(self):
        req = _make_requester()
        req._running = True

        job = SubmittedJob(
            job_id="job_002",
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=0.0,
        )
        req.submitted_jobs["job_002"] = job

        await req._on_job_result("job_result", {
            "job_id": "job_002",
            "provider_id": "provider_node",
            "status": "failed",
            "error": "GPU OOM",
        }, "provider_node")

        req.discovery.record_job_failure.assert_called_once_with("provider_node")

    @pytest.mark.asyncio
    async def test_no_discovery_is_safe(self):
        req = _make_requester()
        req.discovery = None
        req._running = True

        job = SubmittedJob(
            job_id="job_003",
            job_type=JobType.INFERENCE,
            payload={"prompt": "test"},
            ftns_budget=0.0,
        )
        req.submitted_jobs["job_003"] = job

        await req._on_job_result("job_result", {
            "job_id": "job_003",
            "provider_id": "requester_node",
            "status": "completed",
            "result": {"output": "ok"},
        }, "requester_node")

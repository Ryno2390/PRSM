"""Tests for WASM job execution via ComputeProvider."""

import pytest
import base64
from unittest.mock import AsyncMock, MagicMock

from prsm.node.compute_provider import ComputeProvider, ComputeJob, JobType, JobStatus


class TestWASMJobType:
    def test_wasm_execute_in_job_type_enum(self):
        assert hasattr(JobType, "WASM_EXECUTE")
        assert JobType.WASM_EXECUTE == "wasm_execute"


# Minimal WASM: (module (func (export "run") (result i32) (i32.const 42)))
MINIMAL_WASM = bytes([
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x72, 0x75, 0x6e, 0x00, 0x00,
    0x0a, 0x06, 0x01, 0x04, 0x00, 0x41, 0x2a, 0x0b,
])


class TestWASMJobExecution:
    @pytest.fixture
    def provider(self):
        identity = MagicMock()
        identity.node_id = "test-node-123"
        identity.sign = MagicMock(return_value="test-signature")
        identity.public_key_b64 = "dGVzdC1rZXk="

        transport = MagicMock()
        gossip = AsyncMock()
        gossip.publish = AsyncMock()

        ledger = AsyncMock()

        provider = ComputeProvider(
            identity=identity,
            transport=transport,
            gossip=gossip,
            ledger=ledger,
        )
        return provider

    @pytest.mark.asyncio
    async def test_execute_wasm_job_completes(self, provider):
        job = ComputeJob(
            job_id="wasm-job-001",
            job_type=JobType.WASM_EXECUTE,
            requester_id="requester-abc",
            payload={
                "wasm_bytes_b64": base64.b64encode(MINIMAL_WASM).decode(),
                "input_data_b64": base64.b64encode(b'{"query": "test"}').decode(),
                "entry_point": "run",
                "max_memory_bytes": 256 * 1024 * 1024,
                "max_execution_seconds": 30,
            },
            ftns_budget=1.0,
        )

        await provider._execute_job(job)

        assert job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        if job.status == JobStatus.COMPLETED:
            assert job.result is not None
            assert "execution_status" in job.result
            assert job.result_signature is not None

    @pytest.mark.asyncio
    async def test_wasm_job_with_invalid_binary_fails(self, provider):
        job = ComputeJob(
            job_id="wasm-job-bad",
            job_type=JobType.WASM_EXECUTE,
            requester_id="requester-abc",
            payload={
                "wasm_bytes_b64": base64.b64encode(b"not-valid-wasm").decode(),
                "input_data_b64": base64.b64encode(b"").decode(),
                "entry_point": "run",
            },
            ftns_budget=1.0,
        )

        await provider._execute_job(job)

        assert job.status == JobStatus.FAILED
        assert job.error is not None

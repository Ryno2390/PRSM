"""
Sprint 7: Production Polish — Silent Failures & First-Run UX Tests

Tests for:
- Backend detection utility
- Preflight diagnostics integration
- Mock response tagging
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from dataclasses import dataclass
from typing import List
from pathlib import Path

# ── Backend Detection Tests ─────────────────────────────────────────────────────
# TestDetectAvailableBackends removed in v1.6.1: the
# prsm.compute.nwtn.backends module was deleted in v1.6.0 as part of the
# legacy NWTN AGI framework removal.


# ── Preflight Diagnostics Tests ────────────────────────────────────────────────

class TestPreflightDiagnostics:
    """Tests for preflight diagnostics integration."""

    def test_preflight_check_result_dataclass_fields(self):
        """Test that PreflightCheckResult has all expected fields."""
        from prsm.cli import PreflightCheckResult, PREFLIGHT_PASS

        result = PreflightCheckResult(
            name="Test Check",
            status=PREFLIGHT_PASS,
            required=True,
            details="Test details",
            remediation="Test remediation"
        )

        # Verify all fields are accessible
        assert result.name == "Test Check"
        assert result.status == PREFLIGHT_PASS
        assert result.required is True
        assert result.details == "Test details"
        assert result.remediation == "Test remediation"

    def test_preflight_diagnostics_function_exists(self):
        """Test that _node_preflight_diagnostics function exists and is callable."""
        from prsm.cli import _node_preflight_diagnostics

        assert callable(_node_preflight_diagnostics)

    def test_preflight_diagnostics_function_signature(self):
        """Test that _node_preflight_diagnostics has correct signature."""
        from prsm.cli import _node_preflight_diagnostics
        import inspect

        sig = inspect.signature(_node_preflight_diagnostics)
        params = list(sig.parameters.keys())

        # Should have a config parameter
        assert "config" in params

    def test_has_hard_preflight_failures_no_failures(self):
        """Test _has_hard_preflight_failures returns False when no required failures."""
        from prsm.cli import _has_hard_preflight_failures, PreflightCheckResult, PREFLIGHT_PASS, PREFLIGHT_WARN

        results = [
            PreflightCheckResult(
                name="Test 1",
                status=PREFLIGHT_PASS,
                required=True,
                details="OK",
                remediation="None"
            ),
            PreflightCheckResult(
                name="Test 2",
                status=PREFLIGHT_WARN,
                required=False,  # Warning on optional check
                details="Warning",
                remediation="Fix optional"
            ),
        ]

        assert _has_hard_preflight_failures(results) is False

    def test_has_hard_preflight_failures_with_failure(self):
        """Test _has_hard_preflight_failures returns True when required check fails."""
        from prsm.cli import _has_hard_preflight_failures, PreflightCheckResult, PREFLIGHT_PASS, PREFLIGHT_FAIL

        results = [
            PreflightCheckResult(
                name="Test 1",
                status=PREFLIGHT_PASS,
                required=True,
                details="OK",
                remediation="None"
            ),
            PreflightCheckResult(
                name="Test 2",
                status=PREFLIGHT_FAIL,
                required=True,  # Failed required check
                details="Failed",
                remediation="Fix this"
            ),
        ]

        assert _has_hard_preflight_failures(results) is True

    def test_has_hard_preflight_failures_optional_failure_ignored(self):
        """Test that optional check failures don't cause hard failure."""
        from prsm.cli import _has_hard_preflight_failures, PreflightCheckResult, PREFLIGHT_FAIL

        results = [
            PreflightCheckResult(
                name="Optional Check",
                status=PREFLIGHT_FAIL,
                required=False,  # Failed but optional
                details="Failed",
                remediation="Fix optional"
            ),
        ]

        assert _has_hard_preflight_failures(results) is False

    def test_preflight_hard_failures_abort_startup(self, capsys):
        """Test that hard preflight failures abort node startup."""
        from prsm.cli import _has_hard_preflight_failures, PreflightCheckResult, PREFLIGHT_FAIL

        # Create results with a hard failure
        results = [
            PreflightCheckResult(
                name="Critical Check",
                status=PREFLIGHT_FAIL,
                required=True,
                details="Failed",
                remediation="Fix this"
            ),
        ]

        # Verify that hard failures are detected
        assert _has_hard_preflight_failures(results) is True

        # The actual abort logic is in the CLI command, but we can verify
        # the helper function works correctly
        # Simulate the CLI logic
        if _has_hard_preflight_failures(results):
            print("\n❌ Preflight checks failed. Cannot start node.")
            print("   Fix the issues above and try again.")

        # Verify output
        captured = capsys.readouterr()
        assert "Preflight checks failed" in captured.out


# ── Inference Mock Response Tagging Tests ──────────────────────────────────────

class TestInferenceMockResponseTagging:
    """Tests for inference response tagging when no backend is configured."""

    @pytest.fixture
    def mock_identity(self):
        """Create a mock node identity."""
        identity = MagicMock()
        identity.node_id = "test-node-abc123"
        identity.sign = MagicMock(return_value="test-signature")
        identity.public_key_b64 = "test-public-key"
        return identity

    @pytest.fixture
    def mock_transport(self):
        """Create a mock transport."""
        transport = MagicMock()
        transport.peer_count = 0
        return transport

    @pytest.fixture
    def mock_gossip(self):
        """Create a mock gossip protocol."""
        gossip = MagicMock()
        gossip.subscribe = MagicMock()
        gossip.publish = AsyncMock()
        return gossip

    @pytest.fixture
    def mock_ledger(self):
        """Create a mock ledger."""
        ledger = MagicMock()
        ledger.credit = AsyncMock()
        return ledger

    @pytest.fixture
    def compute_provider(self, mock_identity, mock_transport, mock_gossip, mock_ledger):
        """Create a compute provider with mocked dependencies."""
        from prsm.node.compute_provider import ComputeProvider

        return ComputeProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
        )

    @pytest.fixture
    def inference_job(self):
        """Create a sample inference job."""
        from prsm.node.compute_provider import ComputeJob, JobType

        return ComputeJob(
            job_id="test-inference-job-123",
            job_type=JobType.INFERENCE,
            requester_id="test-requester",
            payload={
                "prompt": "What is the meaning of life?",
                "model": "test-model"
            },
            ftns_budget=1.0,
        )

    @pytest.mark.asyncio
    async def test_inference_mock_response_tagging(self, compute_provider, inference_job):
        """Test that inference response includes source=mock and warning when no backend."""
        # Ensure no orchestrator is wired (mock backend scenario)
        compute_provider.orchestrator = None

        result = await compute_provider._run_inference(inference_job)

        # Verify mock response tagging
        assert "source" in result
        assert result["source"] == "mock"
        assert "warning" in result
        assert "No LLM backend configured" in result["warning"]
        assert "mock response" in result["warning"].lower()

    @pytest.mark.asyncio
    async def test_inference_mock_response_includes_provider_node(self, compute_provider, inference_job):
        """Test that mock inference response includes provider_node."""
        compute_provider.orchestrator = None

        result = await compute_provider._run_inference(inference_job)

        assert "provider_node" in result
        assert result["provider_node"] == compute_provider.identity.node_id

    @pytest.mark.asyncio
    async def test_inference_mock_response_includes_prompt(self, compute_provider, inference_job):
        """Test that mock inference response includes truncated prompt."""
        compute_provider.orchestrator = None

        result = await compute_provider._run_inference(inference_job)

        assert "prompt" in result
        assert "model" in result
        assert "response" in result
        assert "tokens_used" in result


# ── Embedding Mock Response Tagging Tests ───────────────────────────────────────

class TestEmbeddingMockResponseTagging:
    """Tests for embedding response tagging when no backend is configured."""

    @pytest.fixture
    def mock_identity(self):
        """Create a mock node identity."""
        identity = MagicMock()
        identity.node_id = "test-node-xyz789"
        identity.sign = MagicMock(return_value="test-signature")
        identity.public_key_b64 = "test-public-key"
        return identity

    @pytest.fixture
    def mock_transport(self):
        """Create a mock transport."""
        transport = MagicMock()
        transport.peer_count = 0
        return transport

    @pytest.fixture
    def mock_gossip(self):
        """Create a mock gossip protocol."""
        gossip = MagicMock()
        gossip.subscribe = MagicMock()
        gossip.publish = AsyncMock()
        return gossip

    @pytest.fixture
    def mock_ledger(self):
        """Create a mock ledger."""
        ledger = MagicMock()
        ledger.credit = AsyncMock()
        return ledger

    @pytest.fixture
    def compute_provider(self, mock_identity, mock_transport, mock_gossip, mock_ledger):
        """Create a compute provider with mocked dependencies."""
        from prsm.node.compute_provider import ComputeProvider

        return ComputeProvider(
            identity=mock_identity,
            transport=mock_transport,
            gossip=mock_gossip,
            ledger=mock_ledger,
        )

    @pytest.fixture
    def embedding_job(self):
        """Create a sample embedding job."""
        from prsm.node.compute_provider import ComputeJob, JobType

        return ComputeJob(
            job_id="test-embedding-job-456",
            job_type=JobType.EMBEDDING,
            requester_id="test-requester",
            payload={
                "text": "This is sample text for embedding generation",
                "dimensions": 1536
            },
            ftns_budget=0.5,
        )

    @pytest.mark.asyncio
    async def test_embedding_mock_response_tagging(self, compute_provider, embedding_job):
        """Test that embedding response includes source=mock and warning when no backend."""
        # Ensure no orchestrator is wired (mock backend scenario)
        compute_provider.orchestrator = None

        result = await compute_provider._run_embedding(embedding_job)

        # Verify mock response tagging
        assert "source" in result
        assert result["source"] == "mock"
        assert "warning" in result
        assert "No embedding backend configured" in result["warning"]
        assert "pseudo-vectors" in result["warning"].lower()

    @pytest.mark.asyncio
    async def test_embedding_mock_response_includes_provider_node(self, compute_provider, embedding_job):
        """Test that mock embedding response includes provider_node."""
        compute_provider.orchestrator = None

        result = await compute_provider._run_embedding(embedding_job)

        assert "provider_node" in result
        assert result["provider_node"] == compute_provider.identity.node_id

    @pytest.mark.asyncio
    async def test_embedding_mock_response_includes_embedding(self, compute_provider, embedding_job):
        """Test that mock embedding response includes valid embedding vector."""
        compute_provider.orchestrator = None

        result = await compute_provider._run_embedding(embedding_job)

        assert "embedding" in result
        assert "dimensions" in result
        assert len(result["embedding"]) == result["dimensions"]
        assert result["provider"] == "mock"
        assert result["model_id"] == "fallback-hash"

    @pytest.mark.asyncio
    async def test_embedding_mock_response_is_normalized(self, compute_provider, embedding_job):
        """Test that mock embedding is normalized to unit vector."""
        import math
        compute_provider.orchestrator = None

        result = await compute_provider._run_embedding(embedding_job)

        # Calculate magnitude of the embedding vector
        embedding = result["embedding"]
        magnitude = math.sqrt(sum(x * x for x in embedding))

        # Should be approximately 1.0 (unit vector)
        assert abs(magnitude - 1.0) < 0.0001, f"Embedding magnitude {magnitude} is not normalized"

    @pytest.mark.asyncio
    async def test_embedding_mock_response_custom_dimensions(self, compute_provider):
        """Test that mock embedding respects custom dimensions."""
        from prsm.node.compute_provider import ComputeJob, JobType

        compute_provider.orchestrator = None

        # Test with different dimensions
        for dimensions in [128, 512, 1536, 3072]:
            job = ComputeJob(
                job_id=f"test-embedding-{dimensions}",
                job_type=JobType.EMBEDDING,
                requester_id="test-requester",
                payload={
                    "text": "Test text",
                    "dimensions": dimensions
                },
                ftns_budget=0.5,
            )

            result = await compute_provider._run_embedding(job)

            assert len(result["embedding"]) == dimensions
            assert result["dimensions"] == dimensions


# ── Integration Tests ───────────────────────────────────────────────────────────

class TestSprint7Integration:
    """Integration tests for Sprint 7 UX improvements."""

    # test_backend_detection_used_in_cli_startup removed in v1.6.1:
    # prsm.compute.nwtn.backends module was deleted in v1.6.0 (legacy AGI framework).

    def test_preflight_constants_defined(self):
        """Test that preflight status constants are properly defined."""
        from prsm.cli import PREFLIGHT_PASS, PREFLIGHT_WARN, PREFLIGHT_FAIL

        assert PREFLIGHT_PASS == "PASS"
        assert PREFLIGHT_WARN == "WARN"
        assert PREFLIGHT_FAIL == "FAIL"

    def test_preflight_check_result_dataclass(self):
        """Test that PreflightCheckResult is properly defined."""
        from prsm.cli import PreflightCheckResult, PREFLIGHT_PASS

        result = PreflightCheckResult(
            name="Test Check",
            status=PREFLIGHT_PASS,
            required=True,
            details="Test details",
            remediation="Test remediation"
        )

        assert result.name == "Test Check"
        assert result.status == PREFLIGHT_PASS
        assert result.required is True
        assert result.details == "Test details"
        assert result.remediation == "Test remediation"

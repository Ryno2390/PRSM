"""Tests for POST /compute/inference endpoint (Phase 3.x.1 Task 5)."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.compute.inference import (
    ContentTier,
    InferenceReceipt,
    InferenceRequest,
    InferenceResult,
    MockInferenceExecutor,
)
from prsm.compute.tee.models import PrivacyLevel, TEEType
from prsm.node.api import create_api_app
from prsm.node.identity import generate_node_identity


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def real_identity():
    """Real Ed25519 keypair so receipt signing actually verifies."""
    return generate_node_identity(display_name="test-inference-node")


@pytest.fixture
def mock_node(real_identity):
    """Mock node wired with a real MockInferenceExecutor + identity."""
    node = MagicMock()
    node.identity = real_identity
    node.inference_executor = MockInferenceExecutor(
        models=["mock-llama-3-8b", "mock-mistral-7b"],
        fixed_cost=Decimal("0.10"),
    )

    # Escrow — track create/release/refund calls
    node._payment_escrow = MagicMock()
    node._payment_escrow.create_escrow = AsyncMock(return_value={"job_id": "stub"})
    node._payment_escrow.release_escrow = AsyncMock(return_value=True)
    node._payment_escrow.refund_escrow = AsyncMock(return_value=True)

    # Privacy budget — track record_spend calls
    node.privacy_budget = MagicMock()
    node.privacy_budget.record_spend = MagicMock()

    # Disable other subsystems we don't exercise here
    node.transport = MagicMock()
    node.transport.peers = {}
    node.transport.address = "127.0.0.1:8765"
    node.compute_provider = True
    node.storage_provider = False

    return node


@pytest.fixture
def client(mock_node):
    """FastAPI TestClient wrapping the node API."""
    app = create_api_app(mock_node, enable_security=False)
    return TestClient(app)


# ── Validation / 400 paths ──────────────────────────────────────────────────


class TestInferenceValidation:
    def test_missing_prompt_returns_400(self, client):
        r = client.post("/compute/inference", json={"model_id": "mock-llama-3-8b"})
        assert r.status_code == 400
        assert "prompt" in r.json()["detail"].lower()

    def test_missing_model_id_returns_400(self, client):
        r = client.post("/compute/inference", json={"prompt": "hello"})
        assert r.status_code == 400
        assert "model_id" in r.json()["detail"].lower()

    def test_zero_budget_returns_400(self, client):
        r = client.post("/compute/inference", json={
            "prompt": "hello", "model_id": "mock-llama-3-8b", "budget_ftns": 0,
        })
        assert r.status_code == 400
        assert "budget" in r.json()["detail"].lower()

    def test_negative_budget_returns_400(self, client):
        r = client.post("/compute/inference", json={
            "prompt": "hello", "model_id": "mock-llama-3-8b", "budget_ftns": -1,
        })
        assert r.status_code == 400

    def test_invalid_privacy_tier_returns_400(self, client):
        r = client.post("/compute/inference", json={
            "prompt": "hello",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "privacy_tier": "ultraviolet",
        })
        assert r.status_code == 400

    def test_invalid_content_tier_returns_400(self, client):
        r = client.post("/compute/inference", json={
            "prompt": "hello",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "content_tier": "Z",
        })
        assert r.status_code == 400


# ── Service-not-available path ──────────────────────────────────────────────


class TestInferenceServiceUnavailable:
    def test_no_executor_returns_503(self, mock_node):
        # Wipe out the executor on the same mock node, then build a client
        mock_node.inference_executor = None
        app = create_api_app(mock_node, enable_security=False)
        c = TestClient(app)
        r = c.post("/compute/inference", json={
            "prompt": "hello", "model_id": "mock-llama-3-8b", "budget_ftns": 1.0,
        })
        assert r.status_code == 503
        assert "inference executor" in r.json()["detail"].lower()


# ── Successful execution path ───────────────────────────────────────────────


class TestInferenceSuccess:
    def _post(self, client, **overrides):
        body = {
            "prompt": "What is 2+2?",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
        }
        body.update(overrides)
        return client.post("/compute/inference", json=body)

    def test_returns_200_and_success(self, client):
        r = self._post(client)
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True

    def test_response_contains_output(self, client):
        body = self._post(client).json()
        assert isinstance(body["output"], str)
        assert len(body["output"]) > 0

    def test_response_contains_signed_receipt(self, client):
        body = self._post(client).json()
        receipt = body["receipt"]
        assert receipt is not None
        # API-side job_id matches the response job_id
        assert receipt["job_id"] == body["job_id"]
        assert receipt["job_id"].startswith("infer-")
        # Signature was applied
        assert receipt["settler_signature"] != ""
        assert receipt["settler_signature"] != "00" * 64
        assert receipt["settler_node_id"] != "mock-settler"

    def test_signed_receipt_is_verifiable(self, client, real_identity):
        from prsm.compute.inference import InferenceReceipt, verify_receipt
        body = self._post(client).json()
        receipt = InferenceReceipt.from_dict(body["receipt"])
        # Anyone with the settling node's public key can verify
        assert verify_receipt(receipt, public_key_b64=real_identity.public_key_b64)

    def test_request_id_is_returned(self, client):
        body = self._post(client).json()
        assert "request_id" in body
        assert body["request_id"].startswith("infer-")


# ── Escrow flow ─────────────────────────────────────────────────────────────


class TestInferenceEscrowFlow:
    def test_escrow_created_with_budget(self, client, mock_node):
        client.post("/compute/inference", json={
            "prompt": "x", "model_id": "mock-llama-3-8b", "budget_ftns": 2.5,
        })
        mock_node._payment_escrow.create_escrow.assert_called_once()
        kwargs = mock_node._payment_escrow.create_escrow.call_args.kwargs
        assert kwargs["amount"] == 2.5

    def test_escrow_released_on_success(self, client, mock_node):
        client.post("/compute/inference", json={
            "prompt": "x", "model_id": "mock-llama-3-8b", "budget_ftns": 1.0,
        })
        mock_node._payment_escrow.release_escrow.assert_called_once()
        mock_node._payment_escrow.refund_escrow.assert_not_called()

    def test_escrow_refunded_on_unknown_model(self, client, mock_node):
        r = client.post("/compute/inference", json={
            "prompt": "x", "model_id": "not-registered", "budget_ftns": 1.0,
        })
        assert r.status_code == 200
        assert r.json()["success"] is False
        mock_node._payment_escrow.refund_escrow.assert_called_once()
        mock_node._payment_escrow.release_escrow.assert_not_called()

    def test_escrow_creation_failure_returns_402(self, client, mock_node):
        mock_node._payment_escrow.create_escrow.side_effect = Exception("insufficient FTNS")
        r = client.post("/compute/inference", json={
            "prompt": "x", "model_id": "mock-llama-3-8b", "budget_ftns": 1.0,
        })
        assert r.status_code == 402
        assert "escrow" in r.json()["detail"].lower()

    def test_escrow_uses_api_job_id(self, client, mock_node):
        body = client.post("/compute/inference", json={
            "prompt": "x", "model_id": "mock-llama-3-8b", "budget_ftns": 1.0,
        }).json()
        # Same job_id flows through escrow create + release + receipt
        create_kwargs = mock_node._payment_escrow.create_escrow.call_args.kwargs
        release_kwargs = mock_node._payment_escrow.release_escrow.call_args.kwargs
        assert create_kwargs["job_id"] == body["job_id"]
        assert release_kwargs["job_id"] == body["job_id"]
        assert body["receipt"]["job_id"] == body["job_id"]


# ── Privacy budget tracking ─────────────────────────────────────────────────


class TestInferencePrivacyBudget:
    def test_standard_privacy_records_spend(self, client, mock_node):
        client.post("/compute/inference", json={
            "prompt": "x",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "privacy_tier": "standard",
        })
        mock_node.privacy_budget.record_spend.assert_called_once()
        epsilon, op, _job_id = mock_node.privacy_budget.record_spend.call_args.args
        assert epsilon == 8.0
        assert op == "inference"

    def test_high_privacy_records_correct_epsilon(self, client, mock_node):
        client.post("/compute/inference", json={
            "prompt": "x",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "privacy_tier": "high",
        })
        epsilon = mock_node.privacy_budget.record_spend.call_args.args[0]
        assert epsilon == 4.0

    def test_maximum_privacy_records_correct_epsilon(self, client, mock_node):
        client.post("/compute/inference", json={
            "prompt": "x",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "privacy_tier": "maximum",
        })
        epsilon = mock_node.privacy_budget.record_spend.call_args.args[0]
        assert epsilon == 1.0

    def test_none_privacy_does_not_record_spend(self, client, mock_node):
        client.post("/compute/inference", json={
            "prompt": "x",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "privacy_tier": "none",
        })
        mock_node.privacy_budget.record_spend.assert_not_called()


# ── Content tier handling ───────────────────────────────────────────────────


class TestInferenceContentTier:
    def test_tier_a_succeeds(self, client):
        r = client.post("/compute/inference", json={
            "prompt": "x",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "content_tier": "A",
        })
        assert r.status_code == 200
        assert r.json()["receipt"]["content_tier"] == "A"

    def test_tier_b_succeeds(self, client):
        r = client.post("/compute/inference", json={
            "prompt": "x",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "content_tier": "B",
        })
        assert r.status_code == 200
        assert r.json()["receipt"]["content_tier"] == "B"

    def test_tier_c_succeeds(self, client):
        r = client.post("/compute/inference", json={
            "prompt": "x",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
            "content_tier": "C",
        })
        assert r.status_code == 200
        assert r.json()["receipt"]["content_tier"] == "C"


# ── Acceptance criteria for Phase 3.x.1 Task 5 ──────────────────────────────


class TestTask5Acceptance:
    """Validates the explicit acceptance criteria from Phase 3.x.1 Task 5.

    Acceptance: API endpoint returns structured response; escrow + privacy
    budget integration verified.
    """

    def test_structured_response_shape(self, client):
        r = client.post("/compute/inference", json={
            "prompt": "Hello",
            "model_id": "mock-llama-3-8b",
            "budget_ftns": 1.0,
        })
        body = r.json()
        # Required response fields
        for key in ("success", "job_id", "request_id", "output", "receipt"):
            assert key in body, f"missing key: {key}"
        # Receipt structure
        receipt = body["receipt"]
        for key in (
            "job_id", "request_id", "model_id", "content_tier", "privacy_tier",
            "epsilon_spent", "tee_type", "tee_attestation", "output_hash",
            "duration_seconds", "cost_ftns", "settler_signature", "settler_node_id",
        ):
            assert key in receipt, f"receipt missing key: {key}"

    def test_escrow_integration(self, client, mock_node):
        """Escrow create + release on success path; create + refund on failure path."""
        # Success path
        client.post("/compute/inference", json={
            "prompt": "x", "model_id": "mock-llama-3-8b", "budget_ftns": 1.0,
        })
        assert mock_node._payment_escrow.create_escrow.call_count == 1
        assert mock_node._payment_escrow.release_escrow.call_count == 1
        # Failure path
        mock_node._payment_escrow.reset_mock()
        client.post("/compute/inference", json={
            "prompt": "x", "model_id": "unknown-model", "budget_ftns": 1.0,
        })
        assert mock_node._payment_escrow.create_escrow.call_count == 1
        assert mock_node._payment_escrow.refund_escrow.call_count == 1

    def test_privacy_budget_integration(self, client, mock_node):
        """Privacy budget records correct epsilon for non-none tiers."""
        for tier, expected_eps in [
            ("standard", 8.0),
            ("high", 4.0),
            ("maximum", 1.0),
        ]:
            mock_node.privacy_budget.record_spend.reset_mock()
            client.post("/compute/inference", json={
                "prompt": "x",
                "model_id": "mock-llama-3-8b",
                "budget_ftns": 1.0,
                "privacy_tier": tier,
            })
            assert mock_node.privacy_budget.record_spend.call_count == 1
            epsilon = mock_node.privacy_budget.record_spend.call_args.args[0]
            assert epsilon == expected_eps

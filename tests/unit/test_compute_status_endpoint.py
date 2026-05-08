"""B8 pass 3 — GET /compute/status/{job_id}.

Reads from ``node._payment_escrow.get_escrow(job_id)`` and returns
the escrow lifecycle for billing reconciliation + the
``prsm_agent_status`` MCP tool.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.payment_escrow import EscrowEntry, EscrowStatus


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


def _node_with_escrow(escrows_by_job_id: dict | None = None):
    """Build a minimal node stub with a payment_escrow that returns
    pre-seeded EscrowEntry objects from get_escrow(job_id)."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    if escrows_by_job_id is None:
        escrows_by_job_id = {}

    escrow_mock = MagicMock()
    escrow_mock.get_escrow = lambda jid: escrows_by_job_id.get(jid)
    node._payment_escrow = escrow_mock
    return node


def _client(node):
    app = create_api_app(node, enable_security=False)
    return TestClient(app)


def _entry(
    *,
    job_id: str = "forge-abc123",
    amount: float = 5.0,
    status: EscrowStatus = EscrowStatus.PENDING,
    provider_winner: str | None = None,
    completed_at: float | None = None,
) -> EscrowEntry:
    return EscrowEntry(
        escrow_id=f"esc-{job_id}",
        job_id=job_id,
        requester_id="test-requester",
        amount=amount,
        status=status,
        provider_winner=provider_winner,
        tx_lock="tx-lock-abc",
        tx_release="tx-release-xyz" if completed_at else None,
        created_at=1_700_000_000.0,
        completed_at=completed_at,
        metadata={"source": "test"},
    )


# ──────────────────────────────────────────────────────────────────────
# Happy path — known job_id
# ──────────────────────────────────────────────────────────────────────


class TestKnownJob:
    def test_pending_escrow_returns_pending_status(self):
        node = _node_with_escrow({
            "forge-job-1": _entry(job_id="forge-job-1"),
        })
        client = _client(node)
        resp = client.get("/compute/status/forge-job-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "forge-job-1"
        assert body["status"] == "pending"
        assert body["amount_ftns"] == 5.0
        assert body["completed_at"] is None

    def test_released_escrow_surfaces_provider_winner_and_release_tx(self):
        node = _node_with_escrow({
            "forge-job-2": _entry(
                job_id="forge-job-2",
                status=EscrowStatus.RELEASED,
                provider_winner="provider-node-7",
                completed_at=1_700_000_500.0,
            ),
        })
        client = _client(node)
        resp = client.get("/compute/status/forge-job-2")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "released"
        assert body["provider_winner"] == "provider-node-7"
        assert body["tx_release"] == "tx-release-xyz"
        assert body["completed_at"] == 1_700_000_500.0

    def test_refunded_escrow_returns_refunded_status(self):
        node = _node_with_escrow({
            "forge-job-3": _entry(
                job_id="forge-job-3",
                status=EscrowStatus.REFUNDED,
                completed_at=1_700_000_900.0,
            ),
        })
        resp = _client(node).get("/compute/status/forge-job-3")
        assert resp.status_code == 200
        assert resp.json()["status"] == "refunded"

    def test_metadata_dict_surfaced_unchanged(self):
        node = _node_with_escrow({
            "forge-job-4": _entry(job_id="forge-job-4"),
        })
        body = _client(node).get("/compute/status/forge-job-4").json()
        assert body["metadata"] == {"source": "test"}

    def test_response_shape_contract(self):
        # Pin the full JSON keyset so MCP clients (prsm_agent_status)
        # can rely on a stable contract.
        node = _node_with_escrow({
            "forge-job-5": _entry(job_id="forge-job-5"),
        })
        body = _client(node).get("/compute/status/forge-job-5").json()
        expected_keys = {
            "job_id",
            "escrow_id",
            "requester_id",
            "amount_ftns",
            "status",
            "provider_winner",
            "tx_lock",
            "tx_release",
            "created_at",
            "completed_at",
            "metadata",
        }
        assert set(body.keys()) == expected_keys


# ──────────────────────────────────────────────────────────────────────
# Error paths — 404 + 503
# ──────────────────────────────────────────────────────────────────────


class TestErrorPaths:
    def test_unknown_job_returns_404(self):
        node = _node_with_escrow({})  # empty
        resp = _client(node).get("/compute/status/forge-missing")
        assert resp.status_code == 404
        assert "No escrow record" in resp.json()["detail"]

    def test_404_message_explains_budget_zero_caveat(self):
        # The error message MUST cue MCP clients that budget=0 jobs
        # don't appear in the escrow ledger — otherwise users will
        # think their job vanished. Pin the explanatory phrase.
        node = _node_with_escrow({})
        body = _client(node).get("/compute/status/forge-missing").json()
        assert "budget=0" in body["detail"] or "budget=0)" in body["detail"]

    def test_no_payment_escrow_returns_503(self):
        node = MagicMock()
        node.identity.node_id = "test-node"
        node._payment_escrow = None  # not initialized
        resp = _client(node).get("/compute/status/forge-x")
        assert resp.status_code == 503
        assert "Payment escrow" in resp.json()["detail"]

"""GET /wallet/escrows/{escrow_id} — direct-lookup detail view.

Companion to /wallet/escrows (list view). Operators investigating
a specific escrow_id from logs / on-chain tx receipts use this
to fetch full detail without scanning the list.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.payment_escrow import EscrowEntry, EscrowStatus


def _entry(*, escrow_id="esc-x", job_id="forge-x",
           amount=5.0, status=EscrowStatus.PENDING):
    return EscrowEntry(
        escrow_id=escrow_id, job_id=job_id,
        requester_id="0x" + "11" * 20,
        amount=amount, status=status,
    )


def _node(*, by_id=None):
    """by_id: dict[escrow_id → EscrowEntry] for the stub
    get_by_escrow_id."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    if by_id is None:
        node._payment_escrow = None
    else:
        escrow_svc = MagicMock()
        escrow_svc.get_by_escrow_id = lambda eid: by_id.get(eid)
        node._payment_escrow = escrow_svc
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Service availability
# ──────────────────────────────────────────────────────────────────────


class TestEscrowDetailAvailability:
    def test_503_when_escrow_not_wired(self):
        node = _node()
        resp = _client(node).get("/wallet/escrows/esc-x")
        assert resp.status_code == 503


# ──────────────────────────────────────────────────────────────────────
# Detail view
# ──────────────────────────────────────────────────────────────────────


class TestEscrowDetail:
    def test_404_when_escrow_id_unknown(self):
        node = _node(by_id={})
        resp = _client(node).get("/wallet/escrows/missing-id")
        assert resp.status_code == 404

    def test_returns_full_record(self):
        entry = _entry(
            escrow_id="esc-abc",
            job_id="forge-aaa",
            amount=12.5,
        )
        node = _node(by_id={"esc-abc": entry})
        resp = _client(node).get("/wallet/escrows/esc-abc")
        assert resp.status_code == 200
        body = resp.json()
        assert body["escrow_id"] == "esc-abc"
        assert body["job_id"] == "forge-aaa"
        assert body["amount_ftns"] == 12.5
        assert body["status"] == "pending"

    def test_resolved_escrow_returns_full_lifecycle(self):
        entry = _entry(
            escrow_id="esc-released",
            status=EscrowStatus.RELEASED,
        )
        node = _node(by_id={"esc-released": entry})
        resp = _client(node).get("/wallet/escrows/esc-released")
        body = resp.json()
        assert body["status"] == "released"

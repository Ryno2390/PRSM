"""GET /wallet/escrows — operator-side escrow summary.

Operators tracking outstanding compute-budget commitments need
a way to enumerate their escrows without scanning history.

Backs the ``prsm_escrow_summary`` MCP tool. Uses
``PaymentEscrow.list_escrows_by_requester`` (shipped same
session as part of the balance_check aggregate-source sprint).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.payment_escrow import EscrowEntry, EscrowStatus


def _entry(*, job_id="forge-x", amount=5.0, status=EscrowStatus.PENDING,
           requester_id="0x" + "11" * 20):
    return EscrowEntry(
        escrow_id=f"esc-{job_id}", job_id=job_id,
        requester_id=requester_id, amount=amount, status=status,
    )


def _node(*, escrows_for_requester=None, requester_address=None):
    """Build a node with payment_escrow.list_escrows_by_requester
    returning a pre-seeded list, and ftns_ledger reporting the
    given connected_address (defaults to a stable test address)."""
    address = requester_address or "0x" + "11" * 20
    node = MagicMock()
    node.identity.node_id = "test-node"

    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = address
    ftns_ledger._decimals = 18
    node.ftns_ledger = ftns_ledger

    if escrows_for_requester is None:
        node._payment_escrow = None
    else:
        escrow_svc = MagicMock()
        escrow_svc.list_escrows_by_requester = MagicMock(
            return_value=escrows_for_requester,
        )
        node._payment_escrow = escrow_svc
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestWalletEscrowsAvailability:
    def test_503_when_escrow_not_wired(self):
        node = _node()
        resp = _client(node).get("/wallet/escrows")
        assert resp.status_code == 503

    def test_503_when_ftns_ledger_not_initialized(self):
        """Without a configured connected address, we can't
        determine which requester to filter on."""
        node = MagicMock()
        node.ftns_ledger = None
        node._payment_escrow = MagicMock()
        resp = _client(node).get("/wallet/escrows")
        assert resp.status_code == 503


class TestWalletEscrowsHappyPath:
    def test_empty_returns_empty_list(self):
        node = _node(escrows_for_requester=[])
        resp = _client(node).get("/wallet/escrows")
        assert resp.status_code == 200
        body = resp.json()
        assert body["escrows"] == []
        assert body["total"] == 0
        assert body["total_locked_ftns"] == 0.0
        assert body["address"]

    def test_returns_pending_escrows(self):
        escrows = [
            _entry(job_id="forge-a", amount=5.0),
            _entry(job_id="forge-b", amount=3.5),
        ]
        node = _node(escrows_for_requester=escrows)
        resp = _client(node).get("/wallet/escrows")
        body = resp.json()
        assert body["total"] == 2
        # Sum of amounts.
        assert body["total_locked_ftns"] == 8.5
        # Each entry surfaces job_id + amount + status.
        ids = {e["job_id"] for e in body["escrows"]}
        assert ids == {"forge-a", "forge-b"}

    def test_address_override_filters_by_address(self):
        """Default uses the node's connected address; explicit
        ?address=0x... overrides for any wallet."""
        captured_arg = {}

        def list_for(requester, *, pending_only=True):
            captured_arg["requester"] = requester
            captured_arg["pending_only"] = pending_only
            return []

        node = MagicMock()
        node.identity.node_id = "test-node"
        ftns_ledger = MagicMock()
        ftns_ledger._is_initialized = True
        ftns_ledger._connected_address = "0x" + "11" * 20
        node.ftns_ledger = ftns_ledger
        escrow_svc = MagicMock()
        escrow_svc.list_escrows_by_requester = list_for
        node._payment_escrow = escrow_svc

        target = "0x" + "ab" * 20
        resp = _client(node).get(f"/wallet/escrows?address={target}")
        assert resp.status_code == 200
        assert captured_arg["requester"] == target

    def test_include_terminal_returns_non_pending(self):
        """By default returns only pending; ?include_terminal=true
        returns all statuses (RELEASED + REFUNDED) for audit."""
        captured_arg = {}

        def list_for(requester, *, pending_only=True):
            captured_arg["pending_only"] = pending_only
            return []

        node = MagicMock()
        node.identity.node_id = "test-node"
        ftns_ledger = MagicMock()
        ftns_ledger._is_initialized = True
        ftns_ledger._connected_address = "0x" + "11" * 20
        node.ftns_ledger = ftns_ledger
        escrow_svc = MagicMock()
        escrow_svc.list_escrows_by_requester = list_for
        node._payment_escrow = escrow_svc

        # Default: pending_only=true.
        _client(node).get("/wallet/escrows")
        assert captured_arg["pending_only"] is True
        # Override: ?include_terminal=true → pending_only=false.
        _client(node).get("/wallet/escrows?include_terminal=true")
        assert captured_arg["pending_only"] is False

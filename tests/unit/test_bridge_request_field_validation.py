"""Sprint 161 — bridge request models field validation.

Pre-fix BridgeDepositRequest + BridgeWithdrawRequest had:
  destination_chain: int = Field(default=137, ...)
  source_chain:      int = Field(default=137, ...)

— no positivity or upper-bound constraint. Negative or absurdly
large chain IDs passed Pydantic validation, then either silently
hit "bridge not initialized" 503s or, on a wired bridge, would
have produced opaque downstream errors.

Sprint 160 introduced Field constraints on /content/upload;
sprint 161 extends the same hygiene to bridge requests.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    """Node with no bridge wired so post-validation 503 wouldn't
    mask Pydantic 422s. The handler reads `node.ftns_bridge`
    (no underscore prefix); explicitly None on a MagicMock node
    forces the 503 branch."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_bridge = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestBridgeDeposit:
    def test_negative_chain_id_returns_422(self):
        resp = _client(_node()).post("/bridge/deposit", json={
            "amount": 1.0, "chain_address": "0xabc",
            "destination_chain": -1,
        })
        assert resp.status_code == 422

    def test_zero_chain_id_returns_422(self):
        resp = _client(_node()).post("/bridge/deposit", json={
            "amount": 1.0, "chain_address": "0xabc",
            "destination_chain": 0,
        })
        assert resp.status_code == 422

    def test_excessive_chain_id_returns_422(self):
        """Beyond uint32 range — clearly bogus."""
        resp = _client(_node()).post("/bridge/deposit", json={
            "amount": 1.0, "chain_address": "0xabc",
            "destination_chain": 99999999999,
        })
        assert resp.status_code == 422

    def test_valid_chain_id_passes(self):
        """Polygon (137) is a known valid chain — passes
        Pydantic validation; downstream 503 from unwired bridge."""
        resp = _client(_node()).post("/bridge/deposit", json={
            "amount": 1.0, "chain_address": "0xabc",
            "destination_chain": 137,
        })
        assert resp.status_code == 503


class TestBridgeWithdraw:
    def test_negative_chain_id_returns_422(self):
        resp = _client(_node()).post("/bridge/withdraw", json={
            "amount": 1.0, "chain_address": "0xabc",
            "source_chain": -1,
        })
        assert resp.status_code == 422

    def test_zero_chain_id_returns_422(self):
        resp = _client(_node()).post("/bridge/withdraw", json={
            "amount": 1.0, "chain_address": "0xabc",
            "source_chain": 0,
        })
        assert resp.status_code == 422

    def test_valid_chain_id_passes(self):
        resp = _client(_node()).post("/bridge/withdraw", json={
            "amount": 1.0, "chain_address": "0xabc",
            "source_chain": 8453,  # Base mainnet
        })
        assert resp.status_code == 503

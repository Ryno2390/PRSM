"""Sprint 169 — /info surfaces the derived operator_address.

Pre-fix /info exposed node_id + api_version + network + chain_id
+ canonical_addresses, but NOT the on-chain operator_address.
Operators had to hit /admin/earnings-summary (which requires
auth in production) to confirm their running node knew its
on-chain identity.

Post-fix /info carries `operator_address` when derivable:
  - From ftns_ledger._connected_address (canonical, via
    FTNS_WALLET_PRIVATE_KEY)
  - Or PRSM_CREATOR_ADDRESS env fallback

Omitted entirely when unset, so test-only / unwired nodes don't
emit a misleading null/empty address.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node(*, operator_address=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._operator_address = operator_address
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def test_info_includes_operator_address_when_set():
    addr = "0xBbEB1cb42F1D5ad05B46eE023D6e4871D813C9a0"
    resp = _client(_node(operator_address=addr)).get("/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["operator_address"] == addr


def test_info_omits_operator_address_when_unset():
    """Defensive — unwired/test nodes don't surface empty/null
    address that would be misleading."""
    resp = _client(_node(operator_address=None)).get("/info")
    body = resp.json()
    assert "operator_address" not in body


def test_info_omits_operator_address_when_empty_string():
    """Empty string also treated as unset."""
    resp = _client(_node(operator_address="")).get("/info")
    body = resp.json()
    assert "operator_address" not in body

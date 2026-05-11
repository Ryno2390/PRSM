"""Sprint 241 — GET /node/identity/pubkey endpoint.

End-users running prsm_inference receive a signed
InferenceReceipt but had no way to fetch the settler's Ed25519
public key over HTTP for verification. The receipt carries
settler_node_id but the network exposed no pubkey lookup.

This endpoint surfaces the node's own pubkey for caller
verification flows. Future cross-node lookup (asking peer A for
peer B's pubkey) is out of scope for this sprint — operators
needing remote pubkeys query each settler's /node/identity/pubkey
directly.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client(identity=None):
    node = MagicMock()
    node.identity = identity
    node.ftns_ledger = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_returns_pubkey_when_identity_wired():
    identity = MagicMock()
    identity.node_id = "test-node"
    identity.public_key_b64 = (
        "BTLPdMcuFdK2GuoMxxv+Z+G8bRPCYqyR6Bt/X8FldhE="
    )
    resp = _client(identity).get("/node/identity/pubkey")
    assert resp.status_code == 200
    body = resp.json()
    assert body["node_id"] == "test-node"
    assert body["public_key_b64"] == (
        "BTLPdMcuFdK2GuoMxxv+Z+G8bRPCYqyR6Bt/X8FldhE="
    )


def test_503_when_identity_unwired():
    resp = _client(None).get("/node/identity/pubkey")
    assert resp.status_code == 503
    assert "identity" in resp.json()["detail"].lower()

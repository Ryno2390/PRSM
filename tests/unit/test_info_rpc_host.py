"""Sprint 170 — /info surfaces RPC host (not full URL) for safe
operator-side endpoint verification.

Pre-fix /info exposed network + chain_id but NOT the actual RPC
URL being used. Operators couldn't tell from /info whether they
were pointed at `https://mainnet.base.org` (default) or a paid
provider like `https://base-mainnet.g.alchemy.com/v2/<key>`.

Surfacing the full URL is unsafe — Alchemy / Infura URLs carry
API keys in the path. Sprint 170 exposes only the HOSTNAME so
operators can verify "am I pointed at the right provider" without
key leakage. URL path + query string stay private.

Example:
  PRSM_BASE_RPC_URL=https://base-sepolia.g.alchemy.com/v2/eqQvrt...
  /info → {"rpc_host": "base-sepolia.g.alchemy.com", ...}
                       ↑ key NOT exposed
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._operator_address = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def test_rpc_host_extracted_from_default_mainnet_url():
    with patch.dict(os.environ, {"PRSM_NETWORK": "mainnet"}, clear=False):
        os.environ.pop("PRSM_BASE_RPC_URL", None)
        os.environ.pop("BASE_RPC_URL", None)
        resp = _client(_node()).get("/info")
    body = resp.json()
    assert "rpc_host" in body
    # Default mainnet RPC is mainnet.base.org
    assert body["rpc_host"] == "mainnet.base.org"


def test_rpc_host_extracted_from_alchemy_url_key_masked():
    """Sprint 170 invariant — Alchemy keys live in URL path; host
    extraction must NOT include the path/key."""
    alchemy = "https://base-mainnet.g.alchemy.com/v2/SECRET_KEY_VALUE"
    with patch.dict(os.environ, {
        "PRSM_NETWORK": "mainnet",
        "PRSM_BASE_RPC_URL": alchemy,
    }, clear=False):
        resp = _client(_node()).get("/info")
    body = resp.json()
    assert body["rpc_host"] == "base-mainnet.g.alchemy.com"
    # Belt-and-suspenders: the secret key must NOT appear anywhere
    # in the response body.
    raw = resp.text
    assert "SECRET_KEY_VALUE" not in raw


def test_rpc_host_for_testnet():
    with patch.dict(os.environ, {
        "PRSM_NETWORK": "testnet",
        "PRSM_BASE_RPC_URL": "https://base-sepolia-rpc.publicnode.com",
    }, clear=False):
        resp = _client(_node()).get("/info")
    body = resp.json()
    assert body["rpc_host"] == "base-sepolia-rpc.publicnode.com"

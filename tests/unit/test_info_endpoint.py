"""GET /info — static node metadata.

Single read-only endpoint surfacing version, active network,
canonical contract addresses, and node identity. Designed for
operator triage + integration code that needs to know "what
network is this node on" without parsing /health/detailed.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node-123"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._royalty_distributor_client = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestInfoFields:
    def test_returns_node_id(self):
        node = _node()
        resp = _client(node).get("/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["node_id"] == "test-node-123"

    def test_returns_api_version(self):
        node = _node()
        resp = _client(node).get("/info")
        body = resp.json()
        assert "api_version" in body
        # Should be the version string from the FastAPI app config.
        assert body["api_version"]

    def test_returns_active_network(self):
        node = _node()
        with patch.dict(os.environ, {"PRSM_NETWORK": "mainnet"}):
            resp = _client(node).get("/info")
        body = resp.json()
        assert body["network"] == "mainnet"

    def test_returns_chain_id_for_mainnet(self):
        node = _node()
        with patch.dict(os.environ, {"PRSM_NETWORK": "mainnet"}):
            resp = _client(node).get("/info")
        body = resp.json()
        assert body["chain_id"] == 8453  # Base mainnet

    def test_returns_canonical_addresses(self):
        node = _node()
        with patch.dict(os.environ, {"PRSM_NETWORK": "mainnet"}):
            resp = _client(node).get("/info")
        body = resp.json()
        canonical = body["canonical_addresses"]
        # Mainnet pins as of 2026-05-09 post-A-08 ceremony.
        assert canonical["ftns_token"].lower() == \
            "0x5276a3756c85f2e9e46f6d34386167a209aa16e5"
        assert canonical["royalty_distributor"].lower() == \
            "0xfea9aeb99e02fdb799e2df3c9195dc4e5323df7e"
        assert canonical["foundation_safe"].lower() == \
            "0x91b0e6f85a371d82de94ed13a3812d9f5a4e5791"
        assert canonical["provenance_registry_v2"].lower() == \
            "0xe0cedda354f99526c7fbb9b9651e12adb2180dbf"


class TestInfoUnknownNetwork:
    def test_unknown_network_returns_partial_info(self):
        """Local / unknown PRSM_NETWORK still returns 200 with
        node_id + api_version; canonical_addresses may be empty
        or partial."""
        node = _node()
        with patch.dict(os.environ, {"PRSM_NETWORK": "local"}):
            resp = _client(node).get("/info")
        assert resp.status_code == 200
        body = resp.json()
        assert "node_id" in body
        assert "api_version" in body

"""GET /admin/earnings-summary — aggregate operator earnings view.

Composes per-stream signals operators need to answer "is my node
earning?" without manually querying each contract:

  * Royalty stream    -> royalty_distributor_client.claimable()
  * Heartbeat status  -> storage_slashing_client.last_heartbeat()
                        + grace window remaining
  * Distribution      -> compensation_distributor.last_distribution_timestamp()

Each stream is independent: any subset can be wired (env-gated) and
the endpoint reports per-stream availability instead of failing
hard. Operator can see at a glance which streams need configuration.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node(
    *,
    royalty_claimable=None,
    last_heartbeat=None,
    grace_seconds=None,
    last_distribution=None,
    operator_address="0xOPERATOR",
):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None

    if royalty_claimable is None:
        node._royalty_distributor_client = None
    else:
        client = MagicMock()
        client.claimable = MagicMock(return_value=royalty_claimable)
        client.address = "0xROYALTY"
        node._royalty_distributor_client = client

    if last_heartbeat is None:
        node._storage_slashing_client = None
    else:
        client = MagicMock()
        client.last_heartbeat = MagicMock(return_value=last_heartbeat)
        client.heartbeat_grace_seconds = MagicMock(return_value=grace_seconds or 3600)
        client.address = "0xSLASH"
        node._storage_slashing_client = client

    if last_distribution is None:
        node._compensation_distributor_client = None
    else:
        client = MagicMock()
        client.last_distribution_timestamp = MagicMock(return_value=last_distribution)
        client.address = "0xCOMP"
        node._compensation_distributor_client = client

    node._operator_address = operator_address
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestEarningsSummary:
    def test_all_streams_unwired(self):
        node = _node()
        resp = _client(node).get("/admin/earnings-summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["royalty"]["available"] is False
        assert body["heartbeat"]["available"] is False
        assert body["distribution"]["available"] is False

    def test_unwired_streams_carry_reason(self):
        """Sprint 152 — operators reading available=false need to
        know WHY (client not wired vs operator addr missing vs RPC
        error). Pre-fix the not-wired branch returned a flat
        {available: False} with no debug info."""
        node = _node()
        resp = _client(node).get("/admin/earnings-summary")
        body = resp.json()
        assert body["royalty"]["reason"] == "client_not_wired"
        assert body["heartbeat"]["reason"] == "client_not_wired"
        assert body["distribution"]["reason"] == "client_not_wired"

    def test_heartbeat_reason_when_operator_addr_missing(self):
        """Sprint 152 — slash client wired but no operator address
        configured → reason='operator_address_missing'.
        Distinguishable from 'client_not_wired' so the operator
        knows to set the right env var."""
        from unittest.mock import MagicMock as _MM
        node = _node()
        node._storage_slashing_client = _MM()
        node._operator_address = None
        resp = _client(node).get("/admin/earnings-summary")
        body = resp.json()
        assert body["heartbeat"]["available"] is False
        assert body["heartbeat"]["reason"] == "operator_address_missing"

    def test_royalty_only_wired(self):
        node = _node(royalty_claimable=12345)
        resp = _client(node).get("/admin/earnings-summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["royalty"]["available"] is True
        assert body["royalty"]["claimable_wei"] == 12345
        assert body["heartbeat"]["available"] is False

    def test_heartbeat_grace_remaining(self):
        now = int(time.time())
        node = _node(
            last_heartbeat=now - 100,
            grace_seconds=3600,
        )
        resp = _client(node).get("/admin/earnings-summary")
        body = resp.json()
        assert body["heartbeat"]["available"] is True
        assert body["heartbeat"]["last_heartbeat"] == now - 100
        assert body["heartbeat"]["grace_seconds"] == 3600
        # Should be ~3500 remaining (allow ±10s for test scheduler)
        assert 3490 <= body["heartbeat"]["grace_remaining"] <= 3510
        assert body["heartbeat"]["at_risk"] is False

    def test_heartbeat_at_risk_when_grace_low(self):
        now = int(time.time())
        node = _node(
            last_heartbeat=now - 3500,
            grace_seconds=3600,
        )
        resp = _client(node).get("/admin/earnings-summary")
        body = resp.json()
        # 100s remaining < 10% threshold of 3600 (360s)
        assert body["heartbeat"]["at_risk"] is True

    def test_heartbeat_expired(self):
        now = int(time.time())
        node = _node(
            last_heartbeat=now - 4000,
            grace_seconds=3600,
        )
        resp = _client(node).get("/admin/earnings-summary")
        body = resp.json()
        assert body["heartbeat"]["grace_remaining"] == 0
        assert body["heartbeat"]["expired"] is True
        assert body["heartbeat"]["at_risk"] is True

    def test_heartbeat_never_recorded(self):
        node = _node(last_heartbeat=0, grace_seconds=3600)
        resp = _client(node).get("/admin/earnings-summary")
        body = resp.json()
        assert body["heartbeat"]["available"] is True
        assert body["heartbeat"]["last_heartbeat"] == 0
        assert body["heartbeat"]["never_recorded"] is True

    def test_distribution_timestamp(self):
        now = int(time.time())
        node = _node(last_distribution=now - 7200)
        resp = _client(node).get("/admin/earnings-summary")
        body = resp.json()
        assert body["distribution"]["available"] is True
        assert body["distribution"]["last_distribution"] == now - 7200
        assert body["distribution"]["seconds_since"] >= 7200

    def test_distribution_never_run(self):
        node = _node(last_distribution=0)
        resp = _client(node).get("/admin/earnings-summary")
        body = resp.json()
        assert body["distribution"]["available"] is True
        assert body["distribution"]["never_distributed"] is True

    def test_per_stream_isolation_on_rpc_failure(self):
        node = _node(royalty_claimable=42, last_heartbeat=int(time.time()))

        # Royalty client raises — heartbeat should still report
        node._royalty_distributor_client.claimable.side_effect = (
            RuntimeError("rpc failed")
        )
        resp = _client(node).get("/admin/earnings-summary")
        body = resp.json()
        assert body["royalty"]["available"] is False
        assert "rpc failed" in body["royalty"]["error"]
        assert body["heartbeat"]["available"] is True

    def test_includes_operator_address(self):
        node = _node(royalty_claimable=10)
        resp = _client(node).get("/admin/earnings-summary")
        body = resp.json()
        assert body["operator_address"] == "0xOPERATOR"

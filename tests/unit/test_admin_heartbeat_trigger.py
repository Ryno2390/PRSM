"""POST /admin/heartbeat/trigger — manual heartbeat record.

Real operator action endpoint. When the heartbeat scheduler
crashes / is paused / is misconfigured, operators need a
manual trigger to avoid being slashed for missed heartbeats.
Equivalent to POST /staking/claim-rewards in spirit (operator
action, on-chain tx, returns tx_hash).

Status semantics:
  503 — slashing client not wired (no PK or wrong env)
  502 — on-chain call raised
  200 — {tx_hash, status: "CONFIRMED" | other TransferStatus}
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node(*, slashing_client=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node._storage_slashing_client = slashing_client
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestEndpoint:
    def test_503_when_client_not_wired(self):
        resp = _client(_node()).post("/admin/heartbeat/trigger")
        assert resp.status_code == 503

    def test_records_heartbeat_returns_tx_hash(self):
        client = MagicMock()
        # TransferStatus.CONFIRMED
        from prsm.economy.web3.provenance_registry import TransferStatus
        client.record_heartbeat = MagicMock(
            return_value=("0xTXHASH", TransferStatus.CONFIRMED),
        )
        node = _node(slashing_client=client)
        resp = _client(node).post("/admin/heartbeat/trigger")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tx_hash"] == "0xTXHASH"
        assert body["status"] == "CONFIRMED"
        client.record_heartbeat.assert_called_once()

    def test_502_when_chain_call_raises(self):
        client = MagicMock()
        client.record_heartbeat = MagicMock(
            side_effect=RuntimeError("rpc connection refused"),
        )
        node = _node(slashing_client=client)
        resp = _client(node).post("/admin/heartbeat/trigger")
        assert resp.status_code == 502
        assert "rpc connection refused" in resp.json()["detail"]

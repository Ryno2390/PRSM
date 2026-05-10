"""POST /admin/distribution/trigger — manual pull_and_distribute.

Operator action endpoint symmetric to heartbeat-trigger
(sprint 81). When the PullAndDistributeScheduler has crashed
or operators want to force an emission round (e.g., after
weight ratification) without waiting for the cadence, this
endpoint posts the on-chain tx.

Permissionless on contract side — anyone can call
pull_and_distribute; caller pays gas. Operator role is the
gas payer.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node(*, comp_client=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node._compensation_distributor_client = comp_client
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestEndpoint:
    def test_503_when_client_not_wired(self):
        resp = _client(_node()).post("/admin/distribution/trigger")
        assert resp.status_code == 503

    def test_returns_tx_hash_on_success(self):
        from prsm.economy.web3.provenance_registry import TransferStatus
        comp = MagicMock()
        comp.pull_and_distribute = MagicMock(
            return_value=("0xDISTRIBUTE_TX", TransferStatus.CONFIRMED),
        )
        resp = _client(_node(comp_client=comp)).post(
            "/admin/distribution/trigger",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["tx_hash"] == "0xDISTRIBUTE_TX"
        assert body["status"] == "CONFIRMED"
        comp.pull_and_distribute.assert_called_once()

    def test_502_on_chain_error(self):
        comp = MagicMock()
        comp.pull_and_distribute = MagicMock(
            side_effect=RuntimeError("zero balance to distribute"),
        )
        resp = _client(_node(comp_client=comp)).post(
            "/admin/distribution/trigger",
        )
        assert resp.status_code == 502
        assert "zero balance" in resp.json()["detail"]

"""Sprint 206 — api.py Pydantic float fields gain upper bound.

Sprint 201 added `allow_inf_nan=False` to api.py StakeRequest,
UnstakeRequest, BridgeDepositRequest, BridgeWithdrawRequest, but
left them open to any finite value. Sending amount=1e15 silently
passed Pydantic and crashed downstream 500. Mirror the dashboard
sprint-205 fix: le=1e12 ceiling.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._staking_service = MagicMock()
    node.ftns_bridge = MagicMock()
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


@pytest.mark.parametrize("path,body", [
    ("/staking/stake", {"amount": 1e15}),
    ("/staking/unstake", {"stake_id": "s1", "amount": 1e15}),
    ("/bridge/deposit", {"amount": 1e15, "chain_address": "0xabc"}),
    ("/bridge/withdraw", {"amount": 1e15, "chain_address": "0xabc"}),
])
def test_excessive_amount_rejected(path, body):
    resp = _client().post(path, json=body)
    assert resp.status_code == 422, resp.text

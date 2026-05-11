"""Sprint 201 — Pydantic float fields reject Infinity.

Sprint 200's commit message flagged the residual gap: Pydantic v2
`Field(gt=0)` rejects NaN but ACCEPTS Infinity (because `inf > 0`
is True). This sprint sweeps the remaining float-bearing request
models with `allow_inf_nan=False`:

  - StakeRequest.amount       (/staking/stake)
  - UnstakeRequest.amount     (/staking/unstake)
  - BridgeDepositRequest.amount  (/bridge/deposit)
  - BridgeWithdrawRequest.amount (/bridge/withdraw)
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
    node.ftns_bridge = MagicMock()
    node._staking_service = MagicMock()
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# Python's stdlib json refuses to serialize inf, so the wire
# payload is built as a raw string with the literal `Infinity`
# (parser-compatible per non-strict mode that Starlette uses).

@pytest.mark.parametrize("path,extra", [
    ("/staking/stake", ""),
    ("/bridge/deposit", ', "chain_address": "0xabc"'),
    ("/bridge/withdraw", ', "chain_address": "0xabc"'),
])
def test_inf_amount_rejected(path, extra):
    raw = '{"amount": Infinity' + extra + "}"
    resp = _client().post(
        path, content=raw,
        headers={"Content-Type": "application/json"},
    )
    # Pre-fix Pydantic accepted Infinity (`inf > 0` is True). Post
    # `allow_inf_nan=False` returns 422.
    assert resp.status_code == 422, resp.text


def test_unstake_inf_amount_rejected():
    resp = _client().post(
        "/staking/unstake",
        content='{"stake_id": "s1", "amount": Infinity}',
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422

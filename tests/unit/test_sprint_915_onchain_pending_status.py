"""Sprint 915 — operator endpoints must distinguish PENDING from FAILED.

The on-chain money-path sweep (sp914) flagged the operator/wallet endpoints
that broadcast a real Base tx and FLATTEN the sp911 typed errors. Each wraps
the chain call in a bare `except Exception` → HTTP 502, so an `OnChainPendingError`
(broadcast SUCCEEDED but `wait_for_transaction_receipt` timed out — the tx is in
the mempool and will likely confirm) is presented identically to a real failure.
An operator seeing 502 re-triggers → the re-broadcast races/reverts and burns gas
(the on-chain contracts guard double-execution: ZeroClaim revert, epoch guard,
harmless heartbeat — so this is OPERABILITY, not fund loss).

Fix (mirrors sp911 dispatch_content_access_royalties): catch OnChainPendingError
explicitly and return HTTP 202 with the tx_hash + a "do NOT re-trigger" note so
the operator reconciles via the receipt. BroadcastFailedError / OnChainRevertedError
remain 502 (both are safe to retry — chain saw nothing / rolled back atomically).

Scope note: auto_claim was investigated and is OUT of scope — StakingManager.
claim_rewards is an inert no-op since sp904 (returns Decimal('0'), no on-chain
broadcast, cannot raise OnChainPendingError).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.economy.web3.provenance_registry import (
    BroadcastFailedError,
    OnChainPendingError,
    OnChainRevertedError,
    TransferStatus,
)

_PENDING_TX = "0x" + "ab" * 32


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def _base_node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node._storage_slashing_client = None
    node._compensation_distributor_client = None
    node._royalty_distributor_client = None
    return node


# ── /admin/heartbeat/trigger ─────────────────────────────────────────────


def _heartbeat_node(record_heartbeat):
    node = _base_node()
    client = MagicMock()
    client.record_heartbeat = record_heartbeat
    node._storage_slashing_client = client
    return node


def test_heartbeat_pending_returns_202_with_tx_hash():
    node = _heartbeat_node(MagicMock(
        side_effect=OnChainPendingError("receipt timed out", _PENDING_TX)))
    resp = _client(node).post("/admin/heartbeat/trigger")
    assert resp.status_code == 202   # NOT 502
    body = resp.json()
    assert body["status"] == "PENDING"
    assert body["tx_hash"] == _PENDING_TX
    # Broadcast happened exactly once — endpoint must not auto-retry.
    node._storage_slashing_client.record_heartbeat.assert_called_once()


def test_heartbeat_reverted_stays_502():
    node = _heartbeat_node(MagicMock(
        side_effect=OnChainRevertedError("reverted")))
    resp = _client(node).post("/admin/heartbeat/trigger")
    assert resp.status_code == 502   # safe to retry — no state change


def test_heartbeat_broadcast_failed_stays_502():
    node = _heartbeat_node(MagicMock(
        side_effect=BroadcastFailedError("never reached network")))
    assert _client(node).post("/admin/heartbeat/trigger").status_code == 502


def test_heartbeat_success_still_200():
    node = _heartbeat_node(MagicMock(
        return_value=("0xOK", TransferStatus.CONFIRMED)))
    resp = _client(node).post("/admin/heartbeat/trigger")
    assert resp.status_code == 200
    assert resp.json()["tx_hash"] == "0xOK"


# ── /admin/distribution/trigger ──────────────────────────────────────────


def _distribution_node(pull_and_distribute):
    node = _base_node()
    client = MagicMock()
    client.pull_and_distribute = pull_and_distribute
    node._compensation_distributor_client = client
    return node


def test_distribution_pending_returns_202_with_tx_hash():
    node = _distribution_node(MagicMock(
        side_effect=OnChainPendingError("receipt timed out", _PENDING_TX)))
    resp = _client(node).post("/admin/distribution/trigger")
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "PENDING"
    assert body["tx_hash"] == _PENDING_TX
    node._compensation_distributor_client.pull_and_distribute.assert_called_once()


def test_distribution_reverted_stays_502():
    node = _distribution_node(MagicMock(
        side_effect=OnChainRevertedError("reverted")))
    assert _client(node).post("/admin/distribution/trigger").status_code == 502


def test_distribution_insufficient_funds_still_402():
    # The sprint-536 gas-top-up 402 special case must survive the refactor.
    node = _distribution_node(MagicMock(
        side_effect=RuntimeError("insufficient funds for gas")))
    assert _client(node).post("/admin/distribution/trigger").status_code == 402


def test_distribution_success_still_200():
    node = _distribution_node(MagicMock(
        return_value=("0xDIST", TransferStatus.CONFIRMED)))
    resp = _client(node).post("/admin/distribution/trigger")
    assert resp.status_code == 200
    assert resp.json()["tx_hash"] == "0xDIST"


# ── /wallet/royalty/claim ────────────────────────────────────────────────


def _royalty_node(*, claimable_wei, claim):
    node = _base_node()
    client = MagicMock()
    client.claimable = MagicMock(return_value=claimable_wei)
    client.claim = claim
    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = "0x" + "11" * 20
    ftns_ledger._decimals = 18
    node.ftns_ledger = ftns_ledger
    node._royalty_distributor_client = client
    return node


def test_royalty_claim_pending_returns_202_with_tx_hash():
    node = _royalty_node(
        claimable_wei=5 * 10**18,
        claim=MagicMock(side_effect=OnChainPendingError("receipt timed out", _PENDING_TX)),
    )
    resp = _client(node).post("/wallet/royalty/claim", json={"dry_run": False})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "PENDING"
    assert body["tx_hash"] == _PENDING_TX
    node._royalty_distributor_client.claim.assert_called_once()


def test_royalty_claim_reverted_stays_502():
    node = _royalty_node(
        claimable_wei=5 * 10**18,
        claim=MagicMock(side_effect=OnChainRevertedError("ZeroClaim")),
    )
    resp = _client(node).post("/wallet/royalty/claim", json={"dry_run": False})
    assert resp.status_code == 502


def test_royalty_claim_success_still_200_executed():
    node = _royalty_node(
        claimable_wei=3 * 10**18,
        claim=MagicMock(return_value=("0xCLAIMED", TransferStatus.CONFIRMED)),
    )
    resp = _client(node).post("/wallet/royalty/claim", json={"dry_run": False})
    assert resp.status_code == 200
    assert resp.json()["status"] == "EXECUTED"
    assert resp.json()["tx_hash"] == "0xCLAIMED"


# ── consumers: CLI must render PENDING (202), not exit-1 on it ────────────


def _resp(status_code, body):
    from unittest.mock import MagicMock as _MM
    r = _MM()
    r.status_code = status_code
    r.json = _MM(return_value=body)
    r.text = str(body)
    return r


_PENDING_TRIGGER = {
    "status": "PENDING", "tx_hash": _PENDING_TX,
    "detail": "broadcast OK but UNCONFIRMED; do NOT re-trigger.",
}
_PENDING_CLAIM = {
    "status": "PENDING", "tx_hash": _PENDING_TX, "claimable_ftns": 5.0,
    "detail": "broadcast OK but UNCONFIRMED; do NOT re-claim.",
}


def test_cli_trigger_heartbeat_pending_does_not_exit_1():
    from click.testing import CliRunner
    from unittest.mock import patch
    from prsm.cli import node
    with patch("httpx.Client") as MockClient:
        ci = MockClient.return_value.__enter__.return_value
        ci.post = MagicMock(return_value=_resp(202, _PENDING_TRIGGER))
        result = CliRunner().invoke(node, ["trigger-heartbeat", "-y"])
    assert result.exit_code == 0      # NOT 1
    assert _PENDING_TX in result.output
    assert "re-trigger" in result.output.lower()


def test_cli_trigger_distribution_pending_does_not_exit_1():
    from click.testing import CliRunner
    from unittest.mock import patch
    from prsm.cli import node
    with patch("httpx.Client") as MockClient:
        ci = MockClient.return_value.__enter__.return_value
        ci.post = MagicMock(return_value=_resp(202, _PENDING_TRIGGER))
        result = CliRunner().invoke(node, ["trigger-distribution", "-y"])
    assert result.exit_code == 0
    assert _PENDING_TX in result.output
    assert "re-trigger" in result.output.lower()


def test_cli_claim_royalty_pending_does_not_exit_1():
    from click.testing import CliRunner
    from unittest.mock import patch
    from prsm.cli import node
    with patch("httpx.Client") as MockClient:
        ci = MockClient.return_value.__enter__.return_value
        ci.post = MagicMock(return_value=_resp(202, _PENDING_CLAIM))
        result = CliRunner().invoke(node, ["claim-royalty", "--execute"])
    assert result.exit_code == 0
    assert _PENDING_TX in result.output
    assert "re-claim" in result.output.lower()


# ── consumers: MCP handlers must render PENDING distinctly ───────────────


@pytest.mark.asyncio
async def test_mcp_distribution_trigger_renders_pending():
    from unittest.mock import patch, AsyncMock
    from prsm import mcp_server
    with patch.object(mcp_server, "_call_node_api",
                      new=AsyncMock(return_value=_PENDING_TRIGGER)):
        out = await mcp_server.handle_prsm_distribution_trigger({})
    assert "PENDING" in out.upper() or "unconfirmed" in out.lower()
    assert _PENDING_TX in out
    assert "re-trigger" in out.lower()


@pytest.mark.asyncio
async def test_mcp_heartbeat_trigger_renders_pending():
    from unittest.mock import patch, AsyncMock
    from prsm import mcp_server
    with patch.object(mcp_server, "_call_node_api",
                      new=AsyncMock(return_value=_PENDING_TRIGGER)):
        out = await mcp_server.handle_prsm_heartbeat_trigger({})
    assert "PENDING" in out.upper() or "unconfirmed" in out.lower()
    assert _PENDING_TX in out
    assert "re-trigger" in out.lower()


@pytest.mark.asyncio
async def test_mcp_royalty_claim_renders_pending():
    from unittest.mock import patch, AsyncMock
    from prsm import mcp_server
    with patch.object(mcp_server, "_call_node_api",
                      new=AsyncMock(return_value=_PENDING_CLAIM)):
        out = await mcp_server.handle_prsm_royalty_claim({"dry_run": False})
    assert "PENDING" in out.upper()
    assert _PENDING_TX in out
    assert "re-claim" in out.lower()

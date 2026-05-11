"""Sprint 182 — endpoints with path params return 404 (not 500) on
malformed / non-existent IDs.

Three endpoints surfaced 500s on `nonexistent` path-param values
during sprint 182's endpoint-probing dogfood:

  GET /staking/stakes/nonexistent          → 500 (was)
  GET /staking/unstake-requests/nonexistent → 500 (was)
  DELETE /agents/nonexistent/allowance     → 500 (was)

Root cause: downstream DB layers reject the non-UUID input with
exceptions (ValueError / TypeError / DBAPI errors) BEFORE the
not-found path's None check fires. The handler's `if not record:
raise 404` never gets a chance.

Post-fix: wrap the lookup in try/except, map any raised exception
to 404 with detail noting the ID format may be wrong.

(The 500 was operator-confusing — a malformed ID is conceptually
a "not found" case, not a server fault.)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node_staking_raises():
    """Node whose staking_manager raises on lookup — simulates the
    DB rejecting malformed UUID."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    sm = MagicMock()
    sm.get_stake = AsyncMock(
        side_effect=ValueError("badly formed hexadecimal UUID string"),
    )
    sm.get_unstake_request = AsyncMock(
        side_effect=ValueError("badly formed hexadecimal UUID string"),
    )
    node.staking_manager = sm
    return node


def _node_ledger_raises():
    """Node whose ledger.revoke_agent_allowance raises."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    ledger = MagicMock()
    ledger.revoke_agent_allowance = AsyncMock(
        side_effect=RuntimeError("DB rejected agent_id"),
    )
    node.ledger = ledger
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestStakingMalformedID:
    def test_get_stake_malformed_id_returns_404(self):
        resp = _client(_node_staking_raises()).get(
            "/staking/stakes/nonexistent",
        )
        assert resp.status_code == 404
        assert "stake_id" in resp.json()["detail"].lower()

    def test_get_unstake_request_malformed_id_returns_404(self):
        resp = _client(_node_staking_raises()).get(
            "/staking/unstake-requests/nonexistent",
        )
        assert resp.status_code == 404
        assert "request_id" in resp.json()["detail"].lower()

    def test_get_stake_returns_none_falls_through_to_404(self):
        """Sprint 182 invariant — None-return path still produces
        404 (we didn't accidentally swallow the legit not-found
        case in the try/except)."""
        node = MagicMock()
        node.identity.node_id = "test-node"
        node.staking_manager = MagicMock()
        node.staking_manager.get_stake = AsyncMock(return_value=None)
        resp = _client(node).get("/staking/stakes/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404


class TestAgentAllowanceMalformedID:
    def test_revoke_malformed_agent_id_returns_404(self):
        resp = _client(_node_ledger_raises()).delete(
            "/agents/nonexistent/allowance",
        )
        assert resp.status_code == 404
        assert "agent_id" in resp.json()["detail"].lower()

    def test_revoke_falsy_return_returns_404(self):
        """Pre-existing None-return path still produces 404."""
        node = MagicMock()
        node.identity.node_id = "test-node"
        node.ledger = MagicMock()
        node.ledger.revoke_agent_allowance = AsyncMock(return_value=False)
        resp = _client(node).delete(
            "/agents/some-id/allowance",
        )
        assert resp.status_code == 404

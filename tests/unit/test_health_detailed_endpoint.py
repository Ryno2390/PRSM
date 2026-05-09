"""GET /health/detailed — structured subsystem readiness probe.

Closes the ops-monitoring gap that the legacy GET /health left
open: load balancers want a fast 200 from /health, but operators
running production nodes want a deeper check that surfaces which
subsystems are healthy / degraded / unhealthy.

Top-level status:
  - healthy: all wired subsystems operational
  - degraded: optional subsystems unavailable but core (FTNS
    ledger + payment escrow) works
  - unhealthy: core subsystem missing or erroring

Per-subsystem fields: {available, status, error?}.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node_minimal():
    """Bare node — no FTNS, no escrow, no anything."""
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._royalty_distributor_client = None
    return node


def _node_full(*, ftns_balance=42.0, escrow_count=0):
    """Fully-wired node with FTNS ledger + escrow + history +
    royalty client."""
    node = MagicMock()
    node.identity.node_id = "test-node"

    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = "0x" + "11" * 20
    ftns_ledger._decimals = 18
    ftns_ledger.get_balance = AsyncMock(return_value=ftns_balance)
    node.ftns_ledger = ftns_ledger

    from prsm.node.payment_escrow import PaymentEscrow, EscrowEntry, EscrowStatus
    led = MagicMock()
    led.get_balance = AsyncMock(return_value=100.0)
    led.transfer = AsyncMock()
    led.create_wallet = AsyncMock()
    escrow = PaymentEscrow(ledger=led, node_id="test-node")
    for i in range(escrow_count):
        entry = EscrowEntry(
            escrow_id=f"e{i}", job_id=f"j{i}",
            requester_id="0x" + "11" * 20, amount=1.0,
            status=EscrowStatus.PENDING,
        )
        escrow._escrows[entry.escrow_id] = entry
    node._payment_escrow = escrow

    from prsm.node.job_history import JobHistoryStore
    node._job_history = JobHistoryStore()

    royalty = MagicMock()
    royalty.claimable = MagicMock(return_value=0)
    node._royalty_distributor_client = royalty
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Top-level status
# ──────────────────────────────────────────────────────────────────────


class TestHealthDetailedStatus:
    def test_unhealthy_when_no_subsystems(self):
        """No FTNS ledger + no escrow → unhealthy. The node is
        nominally up but can't actually serve any value-bearing
        requests."""
        node = _node_minimal()
        resp = _client(node).get("/health/detailed")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert body["node_id"] == "test-node"

    def test_healthy_when_all_subsystems_wired(self):
        node = _node_full()
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        assert body["status"] == "healthy"

    def test_degraded_when_optional_missing(self):
        """FTNS + escrow wired (core); royalty client missing
        (optional) → degraded, not unhealthy."""
        node = _node_full()
        node._royalty_distributor_client = None
        node._job_history = None
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        assert body["status"] == "degraded"


# ──────────────────────────────────────────────────────────────────────
# Per-subsystem reporting
# ──────────────────────────────────────────────────────────────────────


class TestHealthDetailedSubsystems:
    def test_subsystems_object_present(self):
        node = _node_full()
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        assert "subsystems" in body
        ss = body["subsystems"]
        assert "ftns_ledger" in ss
        assert "payment_escrow" in ss
        assert "job_history" in ss
        assert "royalty_distributor" in ss

    def test_unavailable_subsystem_marked(self):
        node = _node_full()
        node._royalty_distributor_client = None
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        assert body["subsystems"]["royalty_distributor"]["available"] is False

    def test_available_subsystem_marked(self):
        node = _node_full()
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        assert body["subsystems"]["ftns_ledger"]["available"] is True
        assert body["subsystems"]["payment_escrow"]["available"] is True

    def test_payment_escrow_reports_pending_count(self):
        node = _node_full(escrow_count=3)
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        ss = body["subsystems"]["payment_escrow"]
        assert ss["pending_count"] == 3


# ──────────────────────────────────────────────────────────────────────
# Subsystem fail-soft
# ──────────────────────────────────────────────────────────────────────


class TestHealthDetailedCanonicalMatch:
    """Post-A-08-ceremony addition (2026-05-09): /health/detailed
    surfaces whether the operator's wired addresses match the
    canonical pins in networks.py. Operators get an instant
    verification step after a contract migration without curling
    individual contract addresses by hand.
    """

    def test_royalty_distributor_canonical_match_true(self):
        """When the wired royalty distributor matches the canonical
        mainnet pin, canonical_match is True."""
        node = _node_full()
        # _node_full's royalty mock has no distributor_address attr,
        # so add it pointing at the canonical v2.
        node._royalty_distributor_client.distributor_address = (
            "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e"
        )
        with patch.dict(__import__("os").environ, {"PRSM_NETWORK": "mainnet"}):
            resp = _client(node).get("/health/detailed")
        body = resp.json()
        royalty = body["subsystems"]["royalty_distributor"]
        assert royalty["available"] is True
        assert "wired_address" in royalty
        assert "canonical_address" in royalty
        assert royalty["canonical_match"] is True

    def test_royalty_distributor_canonical_match_false_on_v1_pin(self):
        """If operator is still pinned to v1 RoyaltyDistributor
        post-ceremony, canonical_match is False — gives operators
        an explicit signal to update their env override."""
        node = _node_full()
        node._royalty_distributor_client.distributor_address = (
            "0x3E8201B2cdC09bB1095Fc63c6DF1673fA9A4D6c2"  # v1
        )
        with patch.dict(__import__("os").environ, {"PRSM_NETWORK": "mainnet"}):
            resp = _client(node).get("/health/detailed")
        body = resp.json()
        royalty = body["subsystems"]["royalty_distributor"]
        assert royalty["canonical_match"] is False
        # canonical_address still surfaced so operator can see
        # what it SHOULD be.
        assert royalty["canonical_address"].lower() == \
            "0xfea9aeb99e02fdb799e2df3c9195dc4e5323df7e"

    def test_ftns_ledger_canonical_match_true(self):
        """Same canonical-match pattern applied to ftns_ledger
        (the FTNS token contract address pin)."""
        node = _node_full()
        node.ftns_ledger.contract_address = (
            "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
        )
        with patch.dict(__import__("os").environ, {"PRSM_NETWORK": "mainnet"}):
            resp = _client(node).get("/health/detailed")
        body = resp.json()
        ftns = body["subsystems"]["ftns_ledger"]
        assert ftns["wired_address"].lower() == \
            "0x5276a3756c85f2e9e46f6d34386167a209aa16e5"
        assert ftns["canonical_address"].lower() == \
            "0x5276a3756c85f2e9e46f6d34386167a209aa16e5"
        assert ftns["canonical_match"] is True

    def test_ftns_ledger_canonical_match_false_on_wrong_token(self):
        """Pinning to a non-canonical token address (e.g., a stale
        testnet token in a mainnet config) surfaces canonical_match
        False."""
        node = _node_full()
        node.ftns_ledger.contract_address = (
            "0xDEADbeefDEADbeefDEADbeefDEADbeefDEADbeef"
        )
        with patch.dict(__import__("os").environ, {"PRSM_NETWORK": "mainnet"}):
            resp = _client(node).get("/health/detailed")
        body = resp.json()
        ftns = body["subsystems"]["ftns_ledger"]
        assert ftns["canonical_match"] is False

    def test_provenance_registry_canonical_match_v2(self):
        """provenance_registry subsystem: canonical-match against
        V2 (the post-2026-05-06 deploy used by the v2 RoyaltyDistributor)."""
        node = _node_full()
        provenance_client = MagicMock()
        provenance_client.contract_address = (
            "0xe0cedDA354f99526c7fbb9b9651e12aDB2180dbf"  # V2
        )
        node._provenance_client = provenance_client
        with patch.dict(__import__("os").environ, {"PRSM_NETWORK": "mainnet"}):
            resp = _client(node).get("/health/detailed")
        body = resp.json()
        prov = body["subsystems"]["provenance_registry"]
        assert prov["canonical_match"] is True
        assert prov["canonical_address"].lower() == \
            "0xe0cedda354f99526c7fbb9b9651e12adb2180dbf"

    def test_provenance_registry_canonical_match_v1_pin_flagged(self):
        """Operator pinned to V1 ProvenanceRegistry post-A-08 should
        see canonical_match=False against the V2 canonical."""
        node = _node_full()
        provenance_client = MagicMock()
        provenance_client.contract_address = (
            "0xdF470BFa9eF310B196801D5105468515d0069915"  # V1
        )
        node._provenance_client = provenance_client
        with patch.dict(__import__("os").environ, {"PRSM_NETWORK": "mainnet"}):
            resp = _client(node).get("/health/detailed")
        body = resp.json()
        prov = body["subsystems"]["provenance_registry"]
        assert prov["canonical_match"] is False
        # canonical surfaced is V2 so operator sees what to update to.
        assert prov["canonical_address"].lower() == \
            "0xe0cedda354f99526c7fbb9b9651e12adb2180dbf"

    def test_canonical_check_handles_unknown_network_gracefully(self):
        """If PRSM_NETWORK is set to a value with no canonical
        addresses (e.g., 'local'), canonical_match should be
        omitted or null rather than crashing."""
        node = _node_full()
        node._royalty_distributor_client.distributor_address = (
            "0xfEa9aeB99e02FDb799E2Df3C9195Dc4e5323df7e"
        )
        with patch.dict(__import__("os").environ, {"PRSM_NETWORK": "local"}):
            resp = _client(node).get("/health/detailed")
        # Endpoint must NOT 500 when canonical lookup fails.
        assert resp.status_code == 200


class TestPaymentEscrowCleanupHealthProbe:
    """payment_escrow subsystem entry surfaces whether the
    periodic_cleanup task is running. Catches the silent-crash
    case where the cleanup loop dies + escrows stop auto-refunding.
    """

    def test_running_field_true_when_task_active(self):
        """When _escrow_cleanup_task is set + not done, the subsystem
        entry includes cleanup_task_running: True."""
        node = _node_full()
        # Simulate a running task.
        fake_task = MagicMock()
        fake_task.done.return_value = False
        node._escrow_cleanup_task = fake_task
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        escrow = body["subsystems"]["payment_escrow"]
        assert escrow.get("cleanup_task_running") is True

    def test_running_field_false_when_task_done(self):
        """If the cleanup task has completed (likely crashed since
        it's an infinite loop), surface cleanup_task_running: False
        so operators see the silent failure."""
        node = _node_full()
        fake_task = MagicMock()
        fake_task.done.return_value = True
        node._escrow_cleanup_task = fake_task
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        escrow = body["subsystems"]["payment_escrow"]
        assert escrow.get("cleanup_task_running") is False

    def test_running_field_absent_when_task_not_wired(self):
        """If the node hasn't started its cleanup task yet
        (e.g., test fixtures, single-shot scripts), the field is
        absent or null rather than False — signals "we don't know"
        rather than "definitely crashed"."""
        node = _node_full()
        # No _escrow_cleanup_task attr set.
        if hasattr(node, "_escrow_cleanup_task"):
            del node._escrow_cleanup_task
        resp = _client(node).get("/health/detailed")
        body = resp.json()
        escrow = body["subsystems"]["payment_escrow"]
        # Either absent or null/None. Both acceptable.
        assert "cleanup_task_running" not in escrow or \
            escrow["cleanup_task_running"] is None


class TestHealthDetailedFailSoft:
    def test_subsystem_check_raising_does_not_500(self):
        """If a subsystem health probe raises (e.g., RPC down),
        the endpoint must NOT 500 — surface the error in the
        subsystem entry + flag overall as degraded/unhealthy."""
        node = _node_full()
        # Force claimable to raise.
        node._royalty_distributor_client.claimable = MagicMock(
            side_effect=RuntimeError("rpc down"),
        )
        resp = _client(node).get("/health/detailed")
        assert resp.status_code == 200
        body = resp.json()
        royalty = body["subsystems"]["royalty_distributor"]
        # Surface the error for ops debugging.
        assert "error" in royalty


# ──────────────────────────────────────────────────────────────────────
# Backwards-compat: /health unchanged
# ──────────────────────────────────────────────────────────────────────


class TestHealthLegacyPreserved:
    def test_simple_health_still_returns_minimal_response(self):
        """/health is the load-balancer probe. Must stay minimal +
        fast — no subsystem checks added."""
        node = _node_full()
        resp = _client(node).get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "subsystems" not in body

"""Sprint 554 — user-signature groundwork for /wallet/withdraw.

First sprint in the user-sig arc (554/555/556/557). Adds the
on-disk schema + bookkeeping needed before sprint 555's verification
primitive and sprint 556's enforcement integration:

  requires_user_signature INTEGER NOT NULL DEFAULT 0   # per-wallet
                                                       # opt-in flag
  next_withdraw_nonce     INTEGER NOT NULL DEFAULT 0   # monotonic
                                                       # nonce counter

Plus a toggle endpoint so operators can flip the flag on/off
per wallet.

No verification logic — sprint 555 owns the EIP-712 verification
primitive; sprint 556 wires it into /wallet/withdraw.
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ── LocalLedger schema + API ──────────────────────────────


@pytest.mark.asyncio
async def test_local_ledger_default_requires_user_signature_false(
    tmp_path,
):
    from prsm.node.local_ledger import LocalLedger
    db = str(tmp_path / "ledger.db")
    ledger = LocalLedger(db_path=db)
    await ledger.initialize()
    wallet_id = "alice"
    await ledger.create_wallet(wallet_id, display_name="alice")
    assert await ledger.get_requires_user_signature(wallet_id) is False


@pytest.mark.asyncio
async def test_local_ledger_set_and_read_requires_user_signature(
    tmp_path,
):
    from prsm.node.local_ledger import LocalLedger
    ledger = LocalLedger(db_path=str(tmp_path / "ledger.db"))
    await ledger.initialize()
    wallet_id = "bob"
    await ledger.create_wallet(wallet_id, display_name="bob")
    await ledger.set_requires_user_signature(wallet_id, True)
    assert await ledger.get_requires_user_signature(wallet_id) is True
    # Round-trip toggle.
    await ledger.set_requires_user_signature(wallet_id, False)
    assert await ledger.get_requires_user_signature(wallet_id) is False


@pytest.mark.asyncio
async def test_local_ledger_default_nonce_zero(tmp_path):
    from prsm.node.local_ledger import LocalLedger
    ledger = LocalLedger(db_path=str(tmp_path / "ledger.db"))
    await ledger.initialize()
    wallet_id = "carol"
    await ledger.create_wallet(wallet_id, display_name="carol")
    assert await ledger.get_next_withdraw_nonce(wallet_id) == 0


@pytest.mark.asyncio
async def test_local_ledger_bump_returns_old_nonce_and_advances(
    tmp_path,
):
    """bump_withdraw_nonce returns the OLD value (the nonce that was
    just consumed) AND advances the counter atomically. Mirrors the
    common "fetch-and-increment" pattern."""
    from prsm.node.local_ledger import LocalLedger
    ledger = LocalLedger(db_path=str(tmp_path / "ledger.db"))
    await ledger.initialize()
    wallet_id = "dan"
    await ledger.create_wallet(wallet_id, display_name="dan")
    assert await ledger.bump_withdraw_nonce(wallet_id) == 0
    assert await ledger.get_next_withdraw_nonce(wallet_id) == 1
    assert await ledger.bump_withdraw_nonce(wallet_id) == 1
    assert await ledger.get_next_withdraw_nonce(wallet_id) == 2


@pytest.mark.asyncio
async def test_local_ledger_schema_migration_idempotent_on_legacy_db(
    tmp_path,
):
    """A legacy DB (pre-sprint-554) gets the new columns added on
    re-initialize; existing rows retain default values (0 = off, 0
    = nonce). Second initialize is a no-op."""
    from prsm.node.local_ledger import LocalLedger
    db = str(tmp_path / "ledger.db")

    # v1 ledger session.
    ledger_v1 = LocalLedger(db_path=db)
    await ledger_v1.initialize()
    wallet_id = "eve"
    await ledger_v1.create_wallet(wallet_id, display_name="eve")
    if hasattr(ledger_v1, "close"):
        await ledger_v1.close()

    # v2 (sprint 554) session — must auto-migrate.
    ledger_v2 = LocalLedger(db_path=db)
    await ledger_v2.initialize()
    assert (
        await ledger_v2.get_requires_user_signature(wallet_id)
    ) is False
    assert (
        await ledger_v2.get_next_withdraw_nonce(wallet_id)
    ) == 0
    if hasattr(ledger_v2, "close"):
        await ledger_v2.close()


# ── DAGLedger parity ──────────────────────────────────────


@pytest.mark.asyncio
async def test_dag_ledger_default_requires_user_signature_false(
    tmp_path,
):
    from prsm.node.dag_ledger import DAGLedger
    ledger = DAGLedger(db_path=str(tmp_path / "dag.db"))
    await ledger.initialize()
    wallet_id = "frank"
    await ledger.create_wallet(wallet_id, display_name="frank")
    assert await ledger.get_requires_user_signature(wallet_id) is False


@pytest.mark.asyncio
async def test_dag_ledger_set_and_bump_round_trip(tmp_path):
    from prsm.node.dag_ledger import DAGLedger
    ledger = DAGLedger(db_path=str(tmp_path / "dag.db"))
    await ledger.initialize()
    wallet_id = "grace"
    await ledger.create_wallet(wallet_id, display_name="grace")

    await ledger.set_requires_user_signature(wallet_id, True)
    assert (
        await ledger.get_requires_user_signature(wallet_id)
    ) is True
    assert await ledger.bump_withdraw_nonce(wallet_id) == 0
    assert await ledger.get_next_withdraw_nonce(wallet_id) == 1


# ── /wallet/require-signature toggle endpoint ─────────────


class _StubNodeLedger:
    """In-memory stand-in for the node's `ledger` attribute. Records
    set calls so the endpoint test can assert wiring."""

    def __init__(self):
        self._flags: dict = {}
        self._wallets = {"w1"}

    async def get_requires_user_signature(self, wallet_id):
        return self._flags.get(wallet_id, False)

    async def set_requires_user_signature(self, wallet_id, enabled):
        if wallet_id not in self._wallets:
            raise KeyError(f"unknown wallet {wallet_id!r}")
        self._flags[wallet_id] = bool(enabled)


def _stub_node(ledger):
    n = MagicMock()
    n.identity = MagicMock(node_id="stub")
    n.ledger = ledger
    return n


def _make_app(ledger):
    from prsm.node.api import create_api_app
    return create_api_app(_stub_node(ledger), enable_security=False)


def test_toggle_endpoint_enables_signature_requirement():
    ledger = _StubNodeLedger()
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/require-signature",
        json={"wallet_id": "w1", "enabled": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["wallet_id"] == "w1"
    assert body["requires_user_signature"] is True


def test_toggle_endpoint_disables_signature_requirement():
    ledger = _StubNodeLedger()
    ledger._flags["w1"] = True
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/require-signature",
        json={"wallet_id": "w1", "enabled": False},
    )
    assert response.status_code == 200, response.text
    assert response.json()["requires_user_signature"] is False


def test_toggle_endpoint_unknown_wallet_returns_404():
    ledger = _StubNodeLedger()
    app = _make_app(ledger)
    client = TestClient(app)
    response = client.post(
        "/wallet/require-signature",
        json={"wallet_id": "unknown", "enabled": True},
    )
    assert response.status_code == 404

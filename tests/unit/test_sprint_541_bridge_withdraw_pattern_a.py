"""Sprint 541 — Pattern A withdraw half (daemon-mediated bridge).

Symmetric inverse of sprint 540 deposit:
  - User has off-chain FTNS balance
  - Daemon debits off-chain (BRIDGE_WITHDRAW) FIRST (concurrency
    safety — concurrent withdraws can't both pass balance check)
  - Daemon broadcasts on-chain transfer from escrow → recipient
  - If broadcast fails, daemon credits refund (audit trail
    preserves both entries)

Pins cover:
  1. Happy path — sufficient balance + broadcast succeeds
  2. Insufficient balance → 402
  3. Invalid amount (<=0) → 422
  4. No recipient (unlinked + no to_eth_address) → 400
  5. Broadcast failure → off-chain credit refund + 502
  6. to_eth_address falls back to linked address
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.dag_ledger import DAGLedger, TransactionType


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(
        suffix=".db", delete=False,
    ) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
async def ledger_with_balance(tmp_db):
    """DAGLedger with alice → 10 FTNS off-chain balance."""
    led = DAGLedger(db_path=tmp_db)
    await led.initialize()
    # Genesis-credit alice
    await led.credit(
        wallet_id="alice", amount=10.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="test setup",
    )
    yield led
    if led._db is not None:
        await led._db.close()


def _build_app_with_node(local_ledger, ftns_ledger):
    """Construct a test FastAPI app with mocked node attributes."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    node.ledger = local_ledger
    node.ftns_ledger = ftns_ledger
    node.identity = MagicMock()
    node.identity.node_id = "alice"
    return create_api_app(node, enable_security=False)


def _build_ftns_ledger(broadcast_outcome="confirmed"):
    """Mock OnChainFTNSLedger.transfer. Returns a record or None."""
    led = MagicMock()
    led._connected_address = (
        "0x4acdE458766C704B2511583572303e77109cFFE8"
    )
    led.contract_address = (
        "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
    )
    led.chain_id = 8453
    if broadcast_outcome == "confirmed":
        tx_record = MagicMock()
        tx_record.tx_hash = "0x" + "ab" * 32
        tx_record.status = "confirmed"
        tx_record.block_number = 46175000
        led.transfer = AsyncMock(return_value=tx_record)
    elif broadcast_outcome == "rejected":
        tx_record = MagicMock()
        tx_record.tx_hash = "0x" + "ab" * 32
        tx_record.status = "rejected"
        tx_record.block_number = None
        led.transfer = AsyncMock(return_value=tx_record)
    else:  # None — broadcast returned no record
        led.transfer = AsyncMock(return_value=None)
    return led


@pytest.mark.asyncio
async def test_withdraw_happy_path(tmp_db):
    """Sufficient balance + confirmed broadcast → 200 +
    debit_tx_id + tx_hash."""
    led = DAGLedger(db_path=tmp_db)
    await led.initialize()
    await led.credit(
        wallet_id="alice", amount=10.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="setup",
    )
    ftns = _build_ftns_ledger("confirmed")
    app = _build_app_with_node(led, ftns)
    client = TestClient(app)
    r = client.post(
        "/wallet/withdraw",
        json={
            "amount_ftns": 1.5,
            "to_eth_address": (
                "0xCba5b409a504480d4C969a47FC74cd8c109F8B15"
            ),
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "confirmed"
    assert body["amount_ftns"] == 1.5
    assert body["tx_hash"] == "0x" + "ab" * 32
    # Off-chain balance should have decreased
    bal = await led.get_balance("alice")
    assert bal == 8.5
    await led._db.close()


@pytest.mark.asyncio
async def test_withdraw_insufficient_balance_402(tmp_db):
    """Balance check rejects withdraw exceeding balance."""
    led = DAGLedger(db_path=tmp_db)
    await led.initialize()
    await led.credit(
        wallet_id="alice", amount=0.5,
        tx_type=TransactionType.WELCOME_GRANT,
        description="setup",
    )
    ftns = _build_ftns_ledger("confirmed")
    app = _build_app_with_node(led, ftns)
    client = TestClient(app)
    r = client.post(
        "/wallet/withdraw",
        json={
            "amount_ftns": 1.0,
            "to_eth_address": "0x" + "f" * 40,
        },
    )
    assert r.status_code == 402
    assert "Insufficient" in r.json()["detail"]
    # Broadcast should NOT have been called
    ftns.transfer.assert_not_called()
    # Balance unchanged
    assert await led.get_balance("alice") == 0.5
    await led._db.close()


@pytest.mark.asyncio
async def test_withdraw_zero_amount_422(tmp_db):
    led = DAGLedger(db_path=tmp_db)
    await led.initialize()
    ftns = _build_ftns_ledger("confirmed")
    app = _build_app_with_node(led, ftns)
    client = TestClient(app)
    r = client.post(
        "/wallet/withdraw",
        json={
            "amount_ftns": 0,
            "to_eth_address": "0x" + "f" * 40,
        },
    )
    assert r.status_code == 422
    await led._db.close()


@pytest.mark.asyncio
async def test_withdraw_no_recipient_400(tmp_db):
    """No explicit to_eth_address + no linked address → 400."""
    led = DAGLedger(db_path=tmp_db)
    await led.initialize()
    await led.credit(
        wallet_id="alice", amount=5.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="setup",
    )
    ftns = _build_ftns_ledger("confirmed")
    app = _build_app_with_node(led, ftns)
    client = TestClient(app)
    r = client.post(
        "/wallet/withdraw",
        json={"amount_ftns": 1.0},  # no to_eth_address
    )
    assert r.status_code == 400
    assert "No recipient" in r.json()["detail"]
    await led._db.close()


@pytest.mark.asyncio
async def test_withdraw_linked_address_fallback(tmp_db):
    """When to_eth_address omitted, daemon uses the linked addr."""
    led = DAGLedger(db_path=tmp_db)
    await led.initialize()
    await led.credit(
        wallet_id="alice", amount=5.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="setup",
    )
    linked = "0xCba5b409a504480d4C969a47FC74cd8c109F8B15"
    await led.link_eth_address("alice", linked)
    ftns = _build_ftns_ledger("confirmed")
    app = _build_app_with_node(led, ftns)
    client = TestClient(app)
    r = client.post(
        "/wallet/withdraw",
        json={"amount_ftns": 1.0},  # no to_eth_address
    )
    assert r.status_code == 200, r.text
    assert r.json()["to_eth_address"] == linked.lower()
    # Verify ftns.transfer was called with that recipient
    ftns.transfer.assert_called_once()
    kwargs = ftns.transfer.call_args.kwargs
    assert kwargs["to_address"] == linked.lower()
    await led._db.close()


@pytest.mark.asyncio
async def test_withdraw_broadcast_failure_refund(tmp_db):
    """Broadcast returns None → daemon credits refund → 502."""
    led = DAGLedger(db_path=tmp_db)
    await led.initialize()
    await led.credit(
        wallet_id="alice", amount=10.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="setup",
    )
    ftns = _build_ftns_ledger(None)  # broadcast returns None
    app = _build_app_with_node(led, ftns)
    client = TestClient(app)
    bal_before = await led.get_balance("alice")
    r = client.post(
        "/wallet/withdraw",
        json={
            "amount_ftns": 2.0,
            "to_eth_address": "0x" + "f" * 40,
        },
    )
    assert r.status_code == 502
    assert "refunded" in r.json()["detail"].lower()
    # Balance should be RESTORED (debit + refund = net zero)
    bal_after = await led.get_balance("alice")
    assert bal_after == bal_before
    await led._db.close()


@pytest.mark.asyncio
async def test_withdraw_broadcast_rejected_refund(tmp_db):
    """Broadcast returns status='rejected' (revert) → refund."""
    led = DAGLedger(db_path=tmp_db)
    await led.initialize()
    await led.credit(
        wallet_id="alice", amount=10.0,
        tx_type=TransactionType.WELCOME_GRANT,
        description="setup",
    )
    ftns = _build_ftns_ledger("rejected")
    app = _build_app_with_node(led, ftns)
    client = TestClient(app)
    bal_before = await led.get_balance("alice")
    r = client.post(
        "/wallet/withdraw",
        json={
            "amount_ftns": 2.0,
            "to_eth_address": "0x" + "f" * 40,
        },
    )
    assert r.status_code == 502
    bal_after = await led.get_balance("alice")
    assert bal_after == bal_before  # net-zero after refund
    await led._db.close()

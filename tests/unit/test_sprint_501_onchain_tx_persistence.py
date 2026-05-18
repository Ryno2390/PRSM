"""Sprint 501 — persist OnChainFTNSLedger transactions to SQLite.

Sprint 500 shipped /wallet/transactions/onchain + the CLI flag,
but the underlying `OnChainFTNSLedger._transactions` is in-memory.
Every daemon restart wipes the audit trail, which makes the
history surface fragile for any real compliance use.

Sprint 501 adds SQLite persistence (aiosqlite, WAL journal, same
pattern as `LocalLedger`). The endpoint's `scope` field flips from
"in-memory (resets on daemon restart)" to "persistent" and reports
the db_path so operators know where the durable record lives.

Boundary: tests use a temp SQLite path + an OnChainFTNSLedger
without web3 wiring. We exercise the persistence helpers
(_record_tx, _update_tx_status, _load_persisted) directly — the
real broadcast path was already live-verified in sprints 498-500.
"""
from __future__ import annotations

import os
import tempfile
import time

import pytest

from prsm.economy.ftns_onchain import (
    OnChainFTNSLedger,
    FTNSTransaction,
)


@pytest.fixture
def tmp_db_path():
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
async def ledger(tmp_db_path):
    """OnChainFTNSLedger with persistence enabled, no web3.

    We bypass full initialize() (which tries to connect to
    Base RPC) and instead just open the SQLite half via
    `_init_persistence`. The persistence helpers must not
    depend on web3 being wired.
    """
    led = OnChainFTNSLedger(
        node_id="test-node",
        wallet_private_key=None,  # no broadcast
        db_path=tmp_db_path,
    )
    await led._init_persistence()
    yield led
    await led._close_persistence()


@pytest.mark.asyncio
async def test_ledger_accepts_db_path_constructor_arg():
    """Constructor must accept db_path (None disables
    persistence — preserves backwards compat for callers
    that don't want it)."""
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None, db_path=None,
    )
    assert led.db_path is None
    assert not led.is_persistent

    led2 = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path="/tmp/x.db",
    )
    assert led2.db_path == "/tmp/x.db"
    assert led2.is_persistent


@pytest.mark.asyncio
async def test_init_persistence_creates_table(
    tmp_db_path,
):
    """_init_persistence must create the
    onchain_transactions table + open the connection."""
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path,
    )
    await led._init_persistence()
    try:
        assert led._db is not None
        cur = await led._db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='onchain_transactions'"
        )
        row = await cur.fetchone()
        assert row is not None
    finally:
        await led._close_persistence()


@pytest.mark.asyncio
async def test_record_tx_persists_row(ledger):
    """_record_tx must INSERT a row keyed by job_id."""
    tx = FTNSTransaction(
        job_id="manual-aaa",
        from_addr="0xAAAA",
        to_addr="0xBBBB",
        amount_ftns=1.5,
        tx_hash="0x" + "11" * 32,
        status="pending",
        created_at=time.time(),
    )
    await ledger._record_tx(tx)
    cur = await ledger._db.execute(
        "SELECT job_id, tx_hash, status, amount_ftns "
        "FROM onchain_transactions WHERE job_id = ?",
        ("manual-aaa",),
    )
    row = await cur.fetchone()
    assert row is not None
    assert row[0] == "manual-aaa"
    assert row[1] == "0x" + "11" * 32
    assert row[2] == "pending"
    assert row[3] == 1.5


@pytest.mark.asyncio
async def test_update_tx_status_promotes_pending_to_confirmed(
    ledger,
):
    """After a TX is recorded as pending, the wait-for-receipt
    callback must update status + block_number in place."""
    tx = FTNSTransaction(
        job_id="manual-bbb",
        from_addr="0xAAAA",
        to_addr="0xBBBB",
        amount_ftns=0.5,
        tx_hash="0x" + "22" * 32,
        status="pending",
    )
    await ledger._record_tx(tx)

    tx.status = "confirmed"
    tx.block_number = 46160500
    await ledger._update_tx_status(tx)

    cur = await ledger._db.execute(
        "SELECT status, block_number FROM onchain_transactions "
        "WHERE job_id = ?",
        ("manual-bbb",),
    )
    row = await cur.fetchone()
    assert row[0] == "confirmed"
    assert row[1] == 46160500


@pytest.mark.asyncio
async def test_load_persisted_populates_in_memory_list(
    tmp_db_path,
):
    """A fresh ledger pointed at a populated DB must load
    rows into _transactions on init."""
    led1 = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path,
    )
    await led1._init_persistence()
    for i in range(3):
        await led1._record_tx(FTNSTransaction(
            job_id=f"manual-{i}",
            from_addr="0xAAAA",
            to_addr="0xBBBB",
            amount_ftns=float(i + 1),
            tx_hash=f"0x{i:064x}",
            status="confirmed",
            block_number=46160000 + i,
            created_at=1700000000.0 + i,
        ))
    await led1._close_persistence()

    # New ledger pointed at same DB
    led2 = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path,
    )
    await led2._init_persistence()
    try:
        assert len(led2._transactions) == 3
        # ordered by created_at ASC
        assert led2._transactions[0].job_id == "manual-0"
        assert led2._transactions[2].block_number == 46160002
        assert led2._transactions[1].amount_ftns == 2.0
    finally:
        await led2._close_persistence()


# (Endpoint `scope` field test moved to
#  test_sprint_501_endpoint_scope_field.py to avoid the
#  asyncio.AUTO mode interaction with the sync TestClient.)

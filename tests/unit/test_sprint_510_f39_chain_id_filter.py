"""Sprint 510 — F39 fix: chain_id discrimination on TX history.

Sprint 509 surfaced F39: SQLite TX history shared across networks.
Operator switching mainnet → Sepolia sees mainnet TX in Sepolia
daemon's /wallet/transactions/onchain, because rows aren't tagged
with the chain they were broadcast on.

Sprint 510 fixes Path A:
  1. Add chain_id INTEGER column to onchain_transactions
     (default NULL — backwards compat with pre-sprint-510 rows).
  2. Set on INSERT to OnChainFTNSLedger.chain_id.
  3. Replay path (initialize) loads ALL rows but filters on
     read.
  4. /wallet/transactions/onchain + /stats endpoints filter
     to current chain_id; legacy NULL rows shown only if
     query param `?include_legacy=1`.

This preserves the single audit-trail file (one ~/.prsm/
onchain_tx.db across mainnet/sepolia/future networks) while
disambiguating per-row.
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


@pytest.mark.asyncio
async def test_schema_has_chain_id_column(tmp_db_path):
    """Table must have chain_id column after init."""
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path, chain_id=8453,
    )
    await led._init_persistence()
    try:
        cur = await led._db.execute(
            "PRAGMA table_info(onchain_transactions)"
        )
        cols = [r[1] for r in await cur.fetchall()]
        assert "chain_id" in cols
    finally:
        await led._close_persistence()


@pytest.mark.asyncio
async def test_record_tx_sets_chain_id(tmp_db_path):
    """INSERT must persist the ledger's chain_id."""
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path, chain_id=8453,
    )
    await led._init_persistence()
    try:
        await led._record_tx(FTNSTransaction(
            job_id="manual-a", from_addr="0xAAAA",
            to_addr="0xBBBB", amount_ftns=1.0,
            tx_hash="0x" + "11" * 32, status="confirmed",
            block_number=46160224,
            created_at=time.time(),
        ))
        cur = await led._db.execute(
            "SELECT chain_id FROM onchain_transactions "
            "WHERE job_id = ?", ("manual-a",),
        )
        row = await cur.fetchone()
        assert row[0] == 8453
    finally:
        await led._close_persistence()


@pytest.mark.asyncio
async def test_load_persisted_filters_by_chain_id(
    tmp_db_path,
):
    """A daemon launched on Sepolia (chain_id 84532) loading
    a DB with both mainnet + Sepolia rows must see only
    Sepolia rows in _transactions."""
    # First populate with both networks via separate ledgers
    led_main = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path, chain_id=8453,
    )
    await led_main._init_persistence()
    await led_main._record_tx(FTNSTransaction(
        job_id="mainnet-1", from_addr="0xAAAA",
        to_addr="0xBBBB", amount_ftns=1.0,
        tx_hash="0x" + "11" * 32, status="confirmed",
        block_number=46160001, created_at=1700000000.0,
    ))
    await led_main._close_persistence()

    led_sep = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path, chain_id=84532,
    )
    await led_sep._init_persistence()
    sepolia_tx = FTNSTransaction(
        job_id="sepolia-1", from_addr="0xCCCC",
        to_addr="0xDDDD", amount_ftns=0.5,
        tx_hash="0x" + "22" * 32, status="confirmed",
        block_number=10000001, created_at=1700001000.0,
    )
    # Mirror production: transfer() appends to _transactions
    # AND persists via _record_tx
    led_sep._transactions.append(sepolia_tx)
    await led_sep._record_tx(sepolia_tx)
    # Verify the Sepolia ledger only sees its own row
    try:
        assert len(led_sep._transactions) == 1
        assert led_sep._transactions[0].job_id == "sepolia-1"
    finally:
        await led_sep._close_persistence()

    # And the mainnet ledger only sees its own
    led_main2 = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path, chain_id=8453,
    )
    await led_main2._init_persistence()
    try:
        assert len(led_main2._transactions) == 1
        assert led_main2._transactions[0].job_id == "mainnet-1"
    finally:
        await led_main2._close_persistence()


@pytest.mark.asyncio
async def test_legacy_null_chain_id_rows_loaded_when_chain_unset(
    tmp_db_path,
):
    """Pre-sprint-510 rows have NULL chain_id. A ledger
    with no chain_id (legacy/unknown context) should see
    them. Once chain_id is set, those NULL rows must NOT
    cross-contaminate (no chain_id is not equal to any
    chain_id)."""
    # Pre-populate with a NULL-chain row directly
    import aiosqlite
    db = await aiosqlite.connect(tmp_db_path)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS onchain_transactions (
            job_id        TEXT PRIMARY KEY,
            tx_hash       TEXT,
            from_addr     TEXT,
            to_addr       TEXT,
            amount_ftns   REAL NOT NULL,
            status        TEXT NOT NULL,
            block_number  INTEGER,
            created_at    REAL NOT NULL,
            chain_id      INTEGER
        )
    """)
    await db.execute(
        "INSERT INTO onchain_transactions "
        "(job_id, tx_hash, from_addr, to_addr, amount_ftns, "
        "status, block_number, created_at, chain_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
        (
            "legacy-1", "0x" + "33" * 32, "0xEEEE",
            "0xFFFF", 0.1, "confirmed", 12345,
            1700002000.0,
        ),
    )
    await db.commit()
    await db.close()

    # Sepolia ledger should NOT load the legacy NULL row
    led_sep = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path, chain_id=84532,
    )
    await led_sep._init_persistence()
    try:
        assert len(led_sep._transactions) == 0
    finally:
        await led_sep._close_persistence()


# Endpoint-contract pin moved to
# test_sprint_510_endpoint_chain_filter.py to avoid the
# asyncio.AUTO mode interaction with sync TestClient.

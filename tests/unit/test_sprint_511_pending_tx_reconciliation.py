"""Sprint 511 — reconcile pending TX from interrupted broadcast.

Sprint 501's persistence stores TX as "pending" the moment they're
broadcast, then upgrades to "confirmed"/"rejected" after
wait_for_transaction_receipt returns (~2-30s). If the daemon is
killed during that window — kernel OOM, SIGKILL, container
restart — the TX confirmed on-chain but the local audit trail
stays "pending" forever.

Sprint 511 fixes via reconciliation at _init_persistence time.
After loading rows for the current chain, iterate any rows where
status="pending" and call w3.eth.get_transaction_receipt(tx_hash):
  - Receipt found + status=1 → mark confirmed, set block_number
  - Receipt found + status=0 → mark rejected
  - Receipt not found (still in mempool / never broadcast) →
    leave as pending (transient — next restart can retry)

Each reconciled row gets _update_tx_status persistence so the
fix survives subsequent restarts.

Boundary: tests inject a mocked w3 + pending rows, verify the
reconcile pass mutates status correctly.
"""
from __future__ import annotations

import os
import tempfile
import time
from unittest.mock import MagicMock

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
async def test_reconcile_pending_to_confirmed_on_chain(
    tmp_db_path,
):
    """A pending row whose tx_hash has a receipt with
    status=1 must be promoted to confirmed."""
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path, chain_id=8453,
    )
    await led._init_persistence()
    pending = FTNSTransaction(
        job_id="manual-x", from_addr="0xAAAA",
        to_addr="0xBBBB", amount_ftns=1.0,
        tx_hash="0x" + "11" * 32,
        status="pending",
        created_at=time.time(),
    )
    led._transactions.append(pending)
    await led._record_tx(pending)

    # Inject a mocked w3 reporting the TX confirmed
    led.w3 = MagicMock()
    fake_receipt = {"status": 1, "blockNumber": 46160500}
    led.w3.eth.get_transaction_receipt.return_value = (
        fake_receipt
    )

    await led._reconcile_pending_transactions()

    assert pending.status == "confirmed"
    assert pending.block_number == 46160500
    # Persistence updated too — re-read should match
    cur = await led._db.execute(
        "SELECT status, block_number FROM "
        "onchain_transactions WHERE job_id = ?",
        ("manual-x",),
    )
    row = await cur.fetchone()
    assert row[0] == "confirmed"
    assert row[1] == 46160500
    await led._close_persistence()


@pytest.mark.asyncio
async def test_reconcile_pending_to_rejected_on_chain(
    tmp_db_path,
):
    """Receipt status=0 → rejected."""
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path, chain_id=8453,
    )
    await led._init_persistence()
    pending = FTNSTransaction(
        job_id="manual-y", from_addr="0xAAAA",
        to_addr="0xBBBB", amount_ftns=1.0,
        tx_hash="0x" + "22" * 32,
        status="pending",
        created_at=time.time(),
    )
    led._transactions.append(pending)
    await led._record_tx(pending)

    led.w3 = MagicMock()
    led.w3.eth.get_transaction_receipt.return_value = (
        {"status": 0, "blockNumber": 46160600}
    )

    await led._reconcile_pending_transactions()

    assert pending.status == "rejected"
    await led._close_persistence()


@pytest.mark.asyncio
async def test_reconcile_no_receipt_leaves_pending(
    tmp_db_path,
):
    """If RPC raises (TX still in mempool / never
    broadcast), leave the row as pending so a future
    restart can retry. Must NOT crash."""
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path, chain_id=8453,
    )
    await led._init_persistence()
    pending = FTNSTransaction(
        job_id="manual-z", from_addr="0xAAAA",
        to_addr="0xBBBB", amount_ftns=1.0,
        tx_hash="0x" + "33" * 32,
        status="pending",
        created_at=time.time(),
    )
    led._transactions.append(pending)
    await led._record_tx(pending)

    led.w3 = MagicMock()
    led.w3.eth.get_transaction_receipt.side_effect = (
        Exception("TransactionNotFound")
    )

    await led._reconcile_pending_transactions()

    assert pending.status == "pending"
    await led._close_persistence()


@pytest.mark.asyncio
async def test_reconcile_skips_already_confirmed(
    tmp_db_path,
):
    """Reconciliation must not re-touch confirmed rows
    (avoid wasting RPC calls on already-settled TX)."""
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path, chain_id=8453,
    )
    await led._init_persistence()
    confirmed = FTNSTransaction(
        job_id="manual-w", from_addr="0xAAAA",
        to_addr="0xBBBB", amount_ftns=1.0,
        tx_hash="0x" + "44" * 32,
        status="confirmed",
        block_number=12345,
        created_at=time.time(),
    )
    led._transactions.append(confirmed)
    await led._record_tx(confirmed)

    led.w3 = MagicMock()
    await led._reconcile_pending_transactions()
    led.w3.eth.get_transaction_receipt.assert_not_called()
    await led._close_persistence()


@pytest.mark.asyncio
async def test_reconcile_is_noop_when_no_w3(
    tmp_db_path,
):
    """If w3 is not initialized, reconciliation must
    silently no-op (the next daemon start with a real w3
    will retry)."""
    led = OnChainFTNSLedger(
        node_id="t", wallet_private_key=None,
        db_path=tmp_db_path, chain_id=8453,
    )
    await led._init_persistence()
    pending = FTNSTransaction(
        job_id="manual-v", from_addr="0xAAAA",
        to_addr="0xBBBB", amount_ftns=1.0,
        tx_hash="0x" + "55" * 32,
        status="pending",
        created_at=time.time(),
    )
    led._transactions.append(pending)
    await led._record_tx(pending)

    led.w3 = None
    # Must not raise
    await led._reconcile_pending_transactions()
    assert pending.status == "pending"
    await led._close_persistence()

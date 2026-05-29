"""Sprint 898 — cross-node FTNS transfer is double-credit-safe.

Highest-blast-radius money path: ledger_sync applies validly-signed
FTNS transfers gossiped from other nodes. `_on_ftns_transaction`
guarded replay with `has_seen_nonce` (check) → `record_nonce` (act)
and dedup with `has_transaction` (check) → `credit` (act). Both are
check-then-act sequences across `await` boundaries with no atomic gate.

Gossip routinely floods the SAME signed transaction via multiple peers,
so two `_on_ftns_transaction` coroutines for the same nonce/tx_id can
interleave: both pass `has_seen_nonce` (neither has recorded yet) and
both pass `has_transaction` (neither has credited yet) → BOTH credit →
the recipient is credited 2× for one transfer. A cross-node
double-credit / counterfeit-FTNS bug.

sp898 makes the nonce an ATOMIC claim. `record_nonce` now returns
whether it actually inserted the row; the `seen_nonces.nonce PRIMARY
KEY` makes `INSERT OR IGNORE` the serialization point, so exactly one
concurrent handler for a given nonce wins (claimed=True) and proceeds to
credit — the rest get False and stop. The cheap `has_seen_nonce`
fast-path stays; the atomic claim is the real gate.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from unittest.mock import MagicMock

import pytest

from prsm.node.local_ledger import (
    LocalLedger,
    Transaction,
    TransactionType,
)
from prsm.node.identity import generate_node_identity
from prsm.node.ledger_sync import LedgerSync


async def _make_ledger():
    ledger = LocalLedger(":memory:")
    await ledger.initialize()
    return ledger


def _ledger_sync(recipient, ledger):
    return LedgerSync(
        identity=recipient,
        gossip=MagicMock(),
        ledger=ledger,
        transport=MagicMock(),
    )


def _signed_transfer_data(sender, recipient, amount):
    tx = Transaction(
        tx_id=str(uuid.uuid4()),
        tx_type=TransactionType.TRANSFER,
        from_wallet=sender.node_id,
        to_wallet=recipient.node_id,
        amount=amount,
        description="cross-node xfer",
        timestamp=time.time(),
        signature="",
    )
    canonical = LedgerSync._canonical_tx_payload(tx, tx.tx_id)
    sig = sender.sign(
        json.dumps(canonical, sort_keys=True).encode(),
    )
    return {
        **canonical,
        "signature": sig,
        "origin_public_key": sender.public_key_b64,
    }


# ── Deterministic anchor: record_nonce is an atomic claim ────

@pytest.mark.asyncio
async def test_record_nonce_returns_true_then_false_on_dup():
    """record_nonce must report whether IT claimed the nonce: True on
    first insert, False on a duplicate. This bool is the atomic gate
    the handler relies on. Pre-sp898 it returned None."""
    ledger = await _make_ledger()
    first = await ledger.record_nonce("nonce-abc", "origin-1")
    second = await ledger.record_nonce("nonce-abc", "origin-2")
    assert first is True
    assert second is False


# ── The bug: concurrent gossip dupes double-credit ───────────

@pytest.mark.asyncio
async def test_concurrent_duplicate_transactions_credit_once():
    """8 concurrent deliveries of ONE signed tx (gossip flood via many
    peers) must credit the recipient exactly once. Pre-sp898 the
    check-then-act guards let several interleave and double-credit."""
    ledger = await _make_ledger()
    recipient = generate_node_identity("recipient")
    sender = generate_node_identity("sender")
    ls = _ledger_sync(recipient, ledger)

    data = _signed_transfer_data(sender, recipient, 10.0)
    await asyncio.gather(*[
        ls._on_ftns_transaction("transfer", data, sender.node_id)
        for _ in range(8)
    ])

    bal = await ledger.get_balance(recipient.node_id)
    assert bal == 10.0, f"double-credit: balance {bal} != 10.0"


# ── Regressions: legit single delivery + sequential replay ───

@pytest.mark.asyncio
async def test_single_delivery_credits_once():
    ledger = await _make_ledger()
    recipient = generate_node_identity("recipient")
    sender = generate_node_identity("sender")
    ls = _ledger_sync(recipient, ledger)

    data = _signed_transfer_data(sender, recipient, 7.5)
    await ls._on_ftns_transaction("transfer", data, sender.node_id)
    assert await ledger.get_balance(recipient.node_id) == 7.5


@pytest.mark.asyncio
async def test_sequential_replay_rejected():
    """Deliver the same tx twice in sequence — the second is a replay
    and must NOT credit again (the has_seen_nonce fast-path)."""
    ledger = await _make_ledger()
    recipient = generate_node_identity("recipient")
    sender = generate_node_identity("sender")
    ls = _ledger_sync(recipient, ledger)

    data = _signed_transfer_data(sender, recipient, 3.0)
    await ls._on_ftns_transaction("transfer", data, sender.node_id)
    await ls._on_ftns_transaction("transfer", data, sender.node_id)
    assert await ledger.get_balance(recipient.node_id) == 3.0


@pytest.mark.asyncio
async def test_bad_signature_never_claims_nonce():
    """A bogus-signature message must be rejected WITHOUT claiming the
    nonce — otherwise an attacker could burn a victim's future nonce
    (griefing). The real tx must still credit afterward."""
    ledger = await _make_ledger()
    recipient = generate_node_identity("recipient")
    sender = generate_node_identity("sender")
    ls = _ledger_sync(recipient, ledger)

    data = _signed_transfer_data(sender, recipient, 5.0)
    forged = {**data, "signature": "AAAA" + data["signature"][4:]}
    await ls._on_ftns_transaction("transfer", forged, sender.node_id)
    # Nonce not burned → the genuine tx still applies.
    assert await ledger.has_seen_nonce(data["nonce"]) is False
    await ls._on_ftns_transaction("transfer", data, sender.node_id)
    assert await ledger.get_balance(recipient.node_id) == 5.0


# ── Same guarantees on the DAGLedger backend ─────────────────
# node.py wires self.ledger as EITHER LocalLedger OR DAGLedger by
# config; both must enforce the atomic claim, and the None→bool change
# must not regress the DAG path (else `if not claimed` would reject
# every transaction).

async def _make_dag_ledger():
    from prsm.node.dag_ledger import DAGLedger
    ledger = DAGLedger(":memory:", verify_signatures=False)
    await ledger.initialize()
    return ledger


@pytest.mark.asyncio
async def test_dag_record_nonce_returns_true_then_false_on_dup():
    ledger = await _make_dag_ledger()
    assert await ledger.record_nonce("n1", "o1") is True
    assert await ledger.record_nonce("n1", "o2") is False


@pytest.mark.asyncio
async def test_dag_concurrent_duplicate_transactions_credit_once():
    ledger = await _make_dag_ledger()
    recipient = generate_node_identity("recipient")
    sender = generate_node_identity("sender")
    await ledger.create_wallet(recipient.node_id, "recipient")
    ls = _ledger_sync(recipient, ledger)

    data = _signed_transfer_data(sender, recipient, 10.0)
    await asyncio.gather(*[
        ls._on_ftns_transaction("transfer", data, sender.node_id)
        for _ in range(8)
    ])
    assert await ledger.get_balance(recipient.node_id) == 10.0


@pytest.mark.asyncio
async def test_dag_single_delivery_credits_once():
    ledger = await _make_dag_ledger()
    recipient = generate_node_identity("recipient")
    sender = generate_node_identity("sender")
    await ledger.create_wallet(recipient.node_id, "recipient")
    ls = _ledger_sync(recipient, ledger)

    data = _signed_transfer_data(sender, recipient, 7.5)
    await ls._on_ftns_transaction("transfer", data, sender.node_id)
    assert await ledger.get_balance(recipient.node_id) == 7.5

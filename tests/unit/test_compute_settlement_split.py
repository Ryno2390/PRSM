"""§4 step 6 — compute-participant settlement post-QO-aggregation.

Two surfaces tested:
  1. ``PaymentEscrow.release_escrow_split`` — multi-recipient atomic
     distribution. Closes the gap where the prompter's compute
     budget was previously paid entirely to the prompter's own node.
  2. ``/compute/forge`` integration — when the QueryOrchestrator
     returns ``participants`` on the AggregatedResult, the endpoint
     dispatches an escrow split (aggregator share + uniform per-
     participant compute share) instead of the legacy single-
     provider release.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.payment_escrow import (
    EscrowAlreadyFinalizedError,
    EscrowEntry,
    EscrowStatus,
    PaymentEscrow,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _Tx:
    tx_id: str


class _StubLedger:
    """Tracks transfers for assertion. Always succeeds unless seeded
    with a per-recipient failure."""

    def __init__(self, escrow_balance: float = 100.0, fail_for=None):
        self._balance = escrow_balance
        self.transfers = []
        self._counter = 0
        self._fail_for = fail_for or set()

    async def get_balance(self, wallet: str) -> float:
        return self._balance

    async def transfer(
        self, from_wallet: str, to_wallet: str, amount: float,
        description: str,
    ):
        if to_wallet in self._fail_for:
            raise ValueError(f"simulated failure for {to_wallet}")
        self._counter += 1
        tx = _Tx(tx_id=f"tx-{self._counter}")
        self.transfers.append({
            "from": from_wallet,
            "to": to_wallet,
            "amount": amount,
            "tx_id": tx.tx_id,
            "description": description,
        })
        return tx


def _seed_escrow(
    escrow: PaymentEscrow, *, job_id="job-x", amount=100.0,
    requester_id="prompter-1",
) -> EscrowEntry:
    entry = EscrowEntry(
        escrow_id=f"esc-{job_id}",
        job_id=job_id,
        requester_id=requester_id,
        amount=amount,
        status=EscrowStatus.PENDING,
    )
    escrow._escrows[entry.escrow_id] = entry
    return entry


def _make_escrow(escrow_balance: float = 100.0, fail_for=None):
    ledger = _StubLedger(escrow_balance=escrow_balance, fail_for=fail_for)
    return PaymentEscrow(ledger=ledger, node_id="test-node"), ledger


# ──────────────────────────────────────────────────────────────────────
# PaymentEscrow.release_escrow_split
# ──────────────────────────────────────────────────────────────────────


class TestSplitHappyPath:
    def test_distributes_to_each_recipient(self):
        escrow, ledger = _make_escrow()
        _seed_escrow(escrow)
        splits = [("agg-node", 5.0), ("worker-a", 47.5), ("worker-b", 47.5)]
        txs = asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=splits,
        ))
        assert txs is not None and len(txs) == 3
        # Each recipient got one transfer with the right amount.
        per_recipient = {t["to"]: t["amount"] for t in ledger.transfers if t["to"] != "prompter-1"}
        assert per_recipient == {"agg-node": 5.0, "worker-a": 47.5, "worker-b": 47.5}

    def test_remainder_refunded_to_requester(self):
        escrow, ledger = _make_escrow(escrow_balance=100.0)
        _seed_escrow(escrow, amount=100.0)
        splits = [("agg-node", 5.0), ("worker-a", 45.0)]  # only 50 of 100
        asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=splits,
        ))
        # Remainder transfer landed on the requester.
        refund = next(t for t in ledger.transfers if t["to"] == "prompter-1")
        assert refund["amount"] == pytest.approx(50.0)

    def test_no_remainder_when_total_equals_amount(self):
        escrow, ledger = _make_escrow()
        _seed_escrow(escrow, amount=100.0)
        splits = [("a", 60.0), ("b", 40.0)]
        asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=splits,
        ))
        # No transfer went to the requester.
        assert not any(t["to"] == "prompter-1" for t in ledger.transfers)

    def test_escrow_status_marked_released(self):
        escrow, _ = _make_escrow()
        entry = _seed_escrow(escrow)
        asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=[("a", 10.0)],
        ))
        assert entry.status == EscrowStatus.RELEASED
        assert entry.completed_at is not None
        assert entry.tx_release is not None

    def test_metadata_records_split_breakdown(self):
        # An audit trail must reconstruct who got paid what without
        # log-replay. Pin the metadata shape.
        escrow, _ = _make_escrow()
        entry = _seed_escrow(escrow)
        splits = [("a", 30.0), ("b", 20.0)]
        asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=splits,
        ))
        recorded = entry.metadata.get("splits", [])
        assert len(recorded) == 2
        recipients = {entry["recipient"] for entry in recorded}
        assert recipients == {"a", "b"}
        # Each entry has tx_id linkage.
        for r in recorded:
            assert r["tx_id"] is not None

    def test_provider_winner_marked_split(self):
        # No single winner — provider_winner reflects this so audit
        # tooling doesn't misinterpret a split as single-recipient.
        escrow, _ = _make_escrow()
        entry = _seed_escrow(escrow)
        asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=[("a", 10.0), ("b", 10.0)],
        ))
        assert entry.provider_winner == "split:2"


class TestSplitValidation:
    def test_sum_exceeds_escrow_amount_raises(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow, amount=10.0)
        with pytest.raises(ValueError, match="exceeds escrow amount"):
            asyncio.run(escrow.release_escrow_split(
                job_id="job-x",
                splits=[("a", 6.0), ("b", 5.0)],  # sum=11 > 10
            ))

    def test_empty_splits_returns_none(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow)
        result = asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=[],
        ))
        assert result is None

    def test_negative_amount_rejected(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow)
        with pytest.raises(ValueError, match="positive number"):
            asyncio.run(escrow.release_escrow_split(
                job_id="job-x", splits=[("a", -1.0)],
            ))

    def test_zero_amount_rejected(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow)
        with pytest.raises(ValueError, match="positive number"):
            asyncio.run(escrow.release_escrow_split(
                job_id="job-x", splits=[("a", 0.0)],
            ))

    def test_empty_recipient_id_rejected(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow)
        with pytest.raises(ValueError, match="non-empty"):
            asyncio.run(escrow.release_escrow_split(
                job_id="job-x", splits=[("", 5.0)],
            ))


class TestSplitStateMachine:
    def test_double_release_is_no_op(self):
        escrow, _ = _make_escrow()
        _seed_escrow(escrow)
        first = asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=[("a", 10.0)],
        ))
        assert first is not None
        # Second call returns None (idempotent).
        second = asyncio.run(escrow.release_escrow_split(
            job_id="job-x", splits=[("a", 10.0)],
        ))
        assert second is None

    def test_release_after_refund_raises(self):
        escrow, _ = _make_escrow()
        entry = _seed_escrow(escrow)
        # Manually set REFUNDED to simulate prior refund without
        # going through the full refund path.
        entry.status = EscrowStatus.REFUNDED
        with pytest.raises(EscrowAlreadyFinalizedError):
            asyncio.run(escrow.release_escrow_split(
                job_id="job-x", splits=[("a", 10.0)],
            ))

    def test_unknown_job_returns_none(self):
        escrow, _ = _make_escrow()
        # No seeded escrow.
        result = asyncio.run(escrow.release_escrow_split(
            job_id="missing", splits=[("a", 10.0)],
        ))
        assert result is None


class TestSplitFailureMode:
    def test_per_recipient_failure_keeps_escrow_pending(self):
        # If one recipient transfer fails partway, we want the
        # caller to see None + the escrow stay PENDING (so cleanup
        # can refund or operator can retry). v1 doesn't roll back
        # already-completed transfers — operator must reconcile.
        escrow, ledger = _make_escrow(fail_for={"bad-worker"})
        entry = _seed_escrow(escrow)
        result = asyncio.run(escrow.release_escrow_split(
            job_id="job-x",
            splits=[("good-a", 10.0), ("bad-worker", 10.0)],
        ))
        assert result is None
        assert entry.status == EscrowStatus.PENDING


# ──────────────────────────────────────────────────────────────────────
# /compute/forge integration — §4 step 6 settlement routing
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _FakeAggregatedResult:
    query_id: bytes
    payload: bytes
    aggregator_node_id: str
    contributing_shards: tuple
    participants: tuple


@dataclass
class _FakeParticipant:
    shard_cid: str
    source_agent_pubkey: bytes
    creator_id: str


class _FakeOrchestrator:
    def __init__(self, participants):
        self._participants = participants

    async def dispatch_query(
        self, *, query, prompter_node_id, query_id,
        requires_tee=False, governance_denylist=frozenset(),
    ):
        return _FakeAggregatedResult(
            query_id=query_id,
            payload=b'{"count": 3}',
            aggregator_node_id="agg-node-7",
            contributing_shards=tuple(p.shard_cid for p in self._participants),
            participants=tuple(self._participants),
        )


def _node_with_orchestrator(participants, escrow):
    node = MagicMock()
    node.identity.node_id = "prompter-1"
    node.privacy_budget = None
    node.agent_forge = _FakeOrchestrator(participants)
    node._payment_escrow = escrow
    return node


def _client(node):
    app = create_api_app(node, enable_security=False)
    return TestClient(app)


class TestForgeSplitRouting:
    """End-to-end through /compute/forge: confirm the QO path
    triggers a multi-recipient escrow split rather than the legacy
    single-provider release."""

    def _setup(
        self, *, monkeypatch, agg_share_bps="500", n_participants=3,
        budget=10.0,
    ):
        # Real PaymentEscrow with stubbed ledger.
        ledger = _StubLedger(escrow_balance=budget)
        escrow = PaymentEscrow(ledger=ledger, node_id="test-node")
        # Pre-create the escrow that /compute/forge expects.
        # /compute/forge calls create_escrow internally; the stub
        # ledger doesn't validate balances on create, so we let
        # the endpoint do its thing.
        async def _create_escrow_stub(**kwargs):
            entry = EscrowEntry(
                escrow_id=f"esc-{kwargs['job_id']}",
                job_id=kwargs["job_id"],
                requester_id=kwargs["requester_id"],
                amount=kwargs["amount"],
                status=EscrowStatus.PENDING,
            )
            escrow._escrows[entry.escrow_id] = entry
            return entry
        escrow.create_escrow = _create_escrow_stub  # monkey-patch

        if agg_share_bps is not None:
            monkeypatch.setenv("PRSM_AGGREGATOR_SHARE_BPS", agg_share_bps)
        else:
            monkeypatch.delenv("PRSM_AGGREGATOR_SHARE_BPS", raising=False)

        participants = [
            _FakeParticipant(
                shard_cid=f"cid-{i}",
                source_agent_pubkey=bytes([i] * 32),
                creator_id=f"creator-{i}",
            )
            for i in range(n_participants)
        ]
        node = _node_with_orchestrator(participants, escrow)
        return _client(node), ledger, escrow, participants

    def test_split_dispatched_when_participants_present(
        self, monkeypatch,
    ):
        client, ledger, escrow, participants = self._setup(
            monkeypatch=monkeypatch, n_participants=3, budget=10.0,
        )
        resp = client.post("/compute/forge", json={
            "query": "Count records",
            "budget_ftns": 10.0,
        })
        assert resp.status_code == 200
        # Aggregator + 3 participants = 4 recipient transfers.
        non_refund = [t for t in ledger.transfers if t["to"] != "prompter-1"]
        assert len(non_refund) == 4
        recipients = {t["to"] for t in non_refund}
        assert "agg-node-7" in recipients
        # source_agent_pubkey is the recipient — hex-encoded in the
        # /compute/forge marshal step.
        for p in participants:
            assert p.source_agent_pubkey.hex() in recipients

    def test_default_aggregator_share_is_5pct(self, monkeypatch):
        # Default PRSM_AGGREGATOR_SHARE_BPS = 500 → 5% of budget.
        client, ledger, escrow, _ = self._setup(
            monkeypatch=monkeypatch, agg_share_bps=None, budget=100.0,
            n_participants=2,
        )
        client.post("/compute/forge", json={
            "query": "q", "budget_ftns": 100.0,
        })
        agg_tx = next(t for t in ledger.transfers if t["to"] == "agg-node-7")
        assert agg_tx["amount"] == pytest.approx(5.0)

    def test_aggregator_share_env_override_honored(self, monkeypatch):
        # Set 10% (1000 bps) and verify aggregator gets 10 of 100.
        client, ledger, escrow, _ = self._setup(
            monkeypatch=monkeypatch, agg_share_bps="1000", budget=100.0,
            n_participants=4,
        )
        client.post("/compute/forge", json={
            "query": "q", "budget_ftns": 100.0,
        })
        agg_tx = next(t for t in ledger.transfers if t["to"] == "agg-node-7")
        assert agg_tx["amount"] == pytest.approx(10.0)
        # Compute total = 90; per-participant = 22.5 across 4.
        worker_txs = [
            t for t in ledger.transfers
            if t["to"] not in ("agg-node-7", "prompter-1")
        ]
        assert all(
            t["amount"] == pytest.approx(22.5) for t in worker_txs
        )

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        # Garbage env value → use 500 default.
        client, ledger, _, _ = self._setup(
            monkeypatch=monkeypatch, agg_share_bps="not-a-number",
            budget=100.0, n_participants=1,
        )
        client.post("/compute/forge", json={
            "query": "q", "budget_ftns": 100.0,
        })
        agg_tx = next(t for t in ledger.transfers if t["to"] == "agg-node-7")
        assert agg_tx["amount"] == pytest.approx(5.0)  # 5% default

    def test_response_surfaces_participants(self, monkeypatch):
        # MCP clients consuming /compute/forge can inspect the
        # participants list in the response (audit-side surface).
        client, _, _, participants = self._setup(
            monkeypatch=monkeypatch, n_participants=2, budget=10.0,
        )
        body = client.post("/compute/forge", json={
            "query": "q", "budget_ftns": 10.0,
        }).json()
        assert "participants" in body["result"]
        assert len(body["result"]["participants"]) == 2
        for entry in body["result"]["participants"]:
            assert "shard_cid" in entry
            assert "source_agent_pubkey_hex" in entry
            assert "creator_id" in entry

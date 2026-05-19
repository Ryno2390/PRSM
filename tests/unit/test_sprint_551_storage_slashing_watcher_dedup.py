"""Sprint 551 — StorageSlashingWatcher event-dedup persistence.

Final sibling in the sprint-549/550 audit trifecta. Same shape:
crash mid-loop → restart re-dispatches every event the previous
run handled. Three event types here too:

  - HeartbeatRecorded — provider liveness proof
  - ProofFailureSlashed — provider failed a custody proof
  - HeartbeatMissingSlashed — provider went silent past the limit

Slash-side duplicates are particularly bad for operator-side audit:
  - slash_event_log.append() writes DUPLICATE slash rows — the
    operator's "who's been slashed how much" view diverges from
    the on-chain truth, and any per-provider reputation/cooldown
    logic keyed off the local log double-penalizes.
  - heartbeat_log.append() duplicates corrupt liveness audit.
  - Webhook fires (slash.heartbeat_recorded / slash.proof_failure_
    slashed / slash.heartbeat_missing_slashed) re-trigger external
    monitoring / paging.

Primitive + node.py builder shipped in sprint 549; sprint 550
applied to KeyDistributionWatcher. This sprint mirrors that
work for the last sibling.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── event schema extension ─────────────────────────────────


def test_heartbeat_recorded_event_accepts_identifiers():
    from prsm.economy.web3.storage_slashing import (
        HeartbeatRecordedEvent,
    )
    event = HeartbeatRecordedEvent(
        provider="0x" + "11" * 20, timestamp=1700000000,
    )
    assert getattr(event, "tx_hash", None) is None
    assert getattr(event, "log_index", None) is None

    event_with_id = HeartbeatRecordedEvent(
        provider="0x" + "11" * 20, timestamp=1700000000,
        tx_hash="0x" + "aa" * 32, log_index=0,
    )
    assert event_with_id.tx_hash == "0x" + "aa" * 32
    assert event_with_id.log_index == 0


def test_proof_failure_slashed_event_accepts_identifiers():
    from prsm.economy.web3.storage_slashing import (
        ProofFailureSlashedEvent,
    )
    event = ProofFailureSlashedEvent(
        provider="0x" + "11" * 20,
        challenger="0x" + "22" * 20,
        shard_id=b"\x33" * 32,
        evidence_hash=b"\x44" * 32,
        slash_id=b"\x55" * 32,
    )
    assert getattr(event, "tx_hash", None) is None
    assert getattr(event, "log_index", None) is None

    event_with_id = ProofFailureSlashedEvent(
        provider="0x" + "11" * 20,
        challenger="0x" + "22" * 20,
        shard_id=b"\x33" * 32,
        evidence_hash=b"\x44" * 32,
        slash_id=b"\x55" * 32,
        tx_hash="0x" + "bb" * 32, log_index=3,
    )
    assert event_with_id.tx_hash == "0x" + "bb" * 32
    assert event_with_id.log_index == 3


def test_heartbeat_missing_slashed_event_accepts_identifiers():
    from prsm.economy.web3.storage_slashing import (
        HeartbeatMissingSlashedEvent,
    )
    event = HeartbeatMissingSlashedEvent(
        provider="0x" + "11" * 20,
        challenger="0x" + "22" * 20,
        last_heartbeat_at=1700000000,
        slash_id=b"\x66" * 32,
    )
    assert getattr(event, "tx_hash", None) is None

    event_with_id = HeartbeatMissingSlashedEvent(
        provider="0x" + "11" * 20,
        challenger="0x" + "22" * 20,
        last_heartbeat_at=1700000000,
        slash_id=b"\x66" * 32,
        tx_hash="0x" + "cc" * 32, log_index=1,
    )
    assert event_with_id.tx_hash == "0x" + "cc" * 32
    assert event_with_id.log_index == 1


# ── decoder threads identifiers ────────────────────────────


def test_get_heartbeat_recorded_events_populates_identifiers():
    from prsm.economy.web3.storage_slashing import (
        StorageSlashingClient,
    )

    raw_log = {
        "args": {
            "provider": "0x" + "11" * 20,
            "timestamp": 1700000000,
        },
        "transactionHash": bytes.fromhex("aa" * 32),
        "logIndex": 0,
    }
    hb = MagicMock()
    hb.return_value.get_logs.return_value = [raw_log]

    client = StorageSlashingClient.__new__(StorageSlashingClient)
    client.contract = MagicMock()
    client.contract.events.HeartbeatRecorded = hb

    events = client.get_heartbeat_recorded_events(
        from_block=100, to_block=200,
    )
    assert len(events) == 1
    assert events[0].tx_hash == "0x" + "aa" * 32
    assert events[0].log_index == 0


def test_get_proof_failure_slashed_events_populates_identifiers():
    from prsm.economy.web3.storage_slashing import (
        StorageSlashingClient,
    )

    raw_log = {
        "args": {
            "provider": "0x" + "11" * 20,
            "challenger": "0x" + "22" * 20,
            "shardId": b"\x33" * 32,
            "evidenceHash": b"\x44" * 32,
            "slashId": b"\x55" * 32,
        },
        "transactionHash": bytes.fromhex("bb" * 32),
        "logIndex": 3,
    }
    slashed = MagicMock()
    slashed.return_value.get_logs.return_value = [raw_log]

    client = StorageSlashingClient.__new__(StorageSlashingClient)
    client.contract = MagicMock()
    client.contract.events.ProofFailureSlashed = slashed

    events = client.get_proof_failure_slashed_events(
        from_block=100, to_block=200,
    )
    assert len(events) == 1
    assert events[0].tx_hash == "0x" + "bb" * 32
    assert events[0].log_index == 3


def test_get_heartbeat_missing_slashed_events_populates_identifiers():
    from prsm.economy.web3.storage_slashing import (
        StorageSlashingClient,
    )

    raw_log = {
        "args": {
            "provider": "0x" + "11" * 20,
            "challenger": "0x" + "22" * 20,
            "lastHeartbeatAt": 1700000000,
            "slashId": b"\x66" * 32,
        },
        "transactionHash": bytes.fromhex("cc" * 32),
        "logIndex": 1,
    }
    missing = MagicMock()
    missing.return_value.get_logs.return_value = [raw_log]

    client = StorageSlashingClient.__new__(StorageSlashingClient)
    client.contract = MagicMock()
    client.contract.events.HeartbeatMissingSlashed = missing

    events = client.get_heartbeat_missing_slashed_events(
        from_block=100, to_block=200,
    )
    assert len(events) == 1
    assert events[0].tx_hash == "0x" + "cc" * 32
    assert events[0].log_index == 1


# ── watcher integration ───────────────────────────────────


class _StubClient:
    def __init__(
        self,
        latest_block,
        recorded=None, proof=None, missing=None,
    ):
        self._latest = latest_block
        self._recorded = list(recorded or [])
        self._proof = list(proof or [])
        self._missing = list(missing or [])

    def latest_block(self):
        return self._latest

    def get_heartbeat_recorded_events(
        self, from_block, to_block, argument_filters=None,
    ):
        return list(self._recorded)

    def get_proof_failure_slashed_events(
        self, from_block, to_block, argument_filters=None,
    ):
        return list(self._proof)

    def get_heartbeat_missing_slashed_events(
        self, from_block, to_block, argument_filters=None,
    ):
        return list(self._missing)


@pytest.mark.asyncio
async def test_watcher_does_not_double_dispatch_across_restart(
    tmp_path,
):
    from prsm.economy.web3.storage_slashing import (
        HeartbeatRecordedEvent,
        ProofFailureSlashedEvent,
        HeartbeatMissingSlashedEvent,
    )
    from prsm.economy.web3.storage_slashing_watcher import (
        StorageSlashingWatcher,
    )
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
        InMemoryLastProcessedBlockStore,
    )

    state = InMemoryLastProcessedBlockStore()
    state.save("storage_slashing", 100)
    dedup_db = str(tmp_path / "dedup.db")

    recorded = [HeartbeatRecordedEvent(
        provider="0x" + "11" * 20, timestamp=1700000000,
        tx_hash="0x" + "01" * 32, log_index=0,
    )]
    proof = [ProofFailureSlashedEvent(
        provider="0x" + "11" * 20,
        challenger="0x" + "22" * 20,
        shard_id=b"\x33" * 32,
        evidence_hash=b"\x44" * 32,
        slash_id=b"\x55" * 32,
        tx_hash="0x" + "02" * 32, log_index=0,
    )]
    missing = [HeartbeatMissingSlashedEvent(
        provider="0x" + "11" * 20,
        challenger="0x" + "22" * 20,
        last_heartbeat_at=1700000000,
        slash_id=b"\x66" * 32,
        tx_hash="0x" + "03" * 32, log_index=0,
    )]
    client = _StubClient(
        latest_block=200,
        recorded=recorded, proof=proof, missing=missing,
    )

    v1_recorded, v1_proof, v1_missing = [], [], []
    async def _r(e): v1_recorded.append(e)
    async def _p(e): v1_proof.append(e)
    async def _m(e): v1_missing.append(e)

    watcher_v1 = StorageSlashingWatcher(
        client=client,
        on_heartbeat_recorded=_r,
        on_proof_failure_slashed=_p,
        on_heartbeat_missing_slashed=_m,
        state_store=state,
        dedup_store=EventDedupStore(dedup_db),
    )
    await watcher_v1.tick()
    assert (len(v1_recorded), len(v1_proof), len(v1_missing)) == (
        1, 1, 1,
    )

    # Simulate crash: rewind state.
    state.save("storage_slashing", 100)

    v2_recorded, v2_proof, v2_missing = [], [], []
    async def _r2(e): v2_recorded.append(e)
    async def _p2(e): v2_proof.append(e)
    async def _m2(e): v2_missing.append(e)

    watcher_v2 = StorageSlashingWatcher(
        client=client,
        on_heartbeat_recorded=_r2,
        on_proof_failure_slashed=_p2,
        on_heartbeat_missing_slashed=_m2,
        state_store=state,
        dedup_store=EventDedupStore(dedup_db),
    )
    await watcher_v2.tick()
    assert v2_recorded == [], "HeartbeatRecorded re-dispatched"
    assert v2_proof == [], "ProofFailureSlashed re-dispatched"
    assert v2_missing == [], "HeartbeatMissingSlashed re-dispatched"


@pytest.mark.asyncio
async def test_watcher_dedup_kwarg_optional(tmp_path):
    """Back-compat: dedup_store=None preserves pre-sprint behavior."""
    from prsm.economy.web3.storage_slashing import (
        HeartbeatRecordedEvent,
    )
    from prsm.economy.web3.storage_slashing_watcher import (
        StorageSlashingWatcher,
    )
    from prsm.economy.web3.last_processed_block_store import (
        InMemoryLastProcessedBlockStore,
    )

    state = InMemoryLastProcessedBlockStore()
    state.save("storage_slashing", 100)
    recorded = [HeartbeatRecordedEvent(
        provider="0x" + "11" * 20, timestamp=1700000000,
        tx_hash="0x" + "0f" * 32, log_index=0,
    )]
    client = _StubClient(latest_block=200, recorded=recorded)

    calls = []
    async def _r(e): calls.append(e)

    watcher = StorageSlashingWatcher(
        client=client,
        on_heartbeat_recorded=_r,
        state_store=state,
    )
    await watcher.tick()
    assert len(calls) == 1

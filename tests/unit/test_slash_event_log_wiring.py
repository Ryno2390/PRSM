"""StorageSlashingWatcher → SlashEventRing wiring.

Verifies that the slash_event_log= kwarg on
_build_storage_slashing_watcher_or_none() actually threads into
the watcher callbacks so on-chain events get recorded.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.economy.web3.storage_slashing import (
    HeartbeatMissingSlashedEvent, ProofFailureSlashedEvent,
)
from prsm.node.node import _build_storage_slashing_watcher_or_none
from prsm.node.slash_event_log import SlashEventRing


@pytest.fixture
def mock_client():
    return MagicMock()


@pytest.fixture
def opted_in_env():
    with patch.dict(os.environ, {
        "PRSM_STORAGE_SLASHING_WATCHER_ENABLED": "1",
    }):
        yield


def test_proof_failure_slash_recorded_in_ring(mock_client, opted_in_env):
    ring = SlashEventRing()
    watcher = _build_storage_slashing_watcher_or_none(
        client=mock_client,
        slash_event_log=ring,
    )
    assert watcher is not None

    event = ProofFailureSlashedEvent(
        provider="0xPROV",
        challenger="0xCHAL",
        shard_id=b"\x11" * 32,
        evidence_hash=b"\x22" * 32,
        slash_id=b"\x33" * 32,
    )
    watcher._on_proof(event)

    entries = ring.recent()
    assert len(entries) == 1
    assert entries[0].kind == "proof_failure_slashed"
    assert entries[0].provider == "0xPROV"
    assert entries[0].challenger == "0xCHAL"
    assert entries[0].extras["shard_id"] == "0x" + "11" * 32


def test_heartbeat_missing_slash_recorded_in_ring(
    mock_client, opted_in_env,
):
    ring = SlashEventRing()
    watcher = _build_storage_slashing_watcher_or_none(
        client=mock_client,
        slash_event_log=ring,
    )
    event = HeartbeatMissingSlashedEvent(
        provider="0xPROV",
        challenger="0xCHAL",
        last_heartbeat_at=1700000000,
        slash_id=b"\x44" * 32,
    )
    watcher._on_missing(event)

    entries = ring.recent()
    assert len(entries) == 1
    assert entries[0].kind == "heartbeat_missing_slashed"
    assert entries[0].extras["last_heartbeat_at"] == 1700000000


def test_no_ring_argument_does_not_crash(mock_client, opted_in_env):
    """Operator who doesn't pass a ring still gets a working watcher."""
    watcher = _build_storage_slashing_watcher_or_none(
        client=mock_client,
        slash_event_log=None,
    )
    event = HeartbeatMissingSlashedEvent(
        provider="0xPROV", challenger="0xCHAL",
        last_heartbeat_at=0, slash_id=b"\x55" * 32,
    )
    # Should NOT raise
    watcher._on_missing(event)


def test_ring_failure_does_not_break_watcher_callback(
    mock_client, opted_in_env,
):
    """If ring.append() raises, callback still completes (logging
    side effect runs, no exception propagates)."""
    ring = MagicMock()
    ring.append.side_effect = RuntimeError("simulated ring failure")
    watcher = _build_storage_slashing_watcher_or_none(
        client=mock_client,
        slash_event_log=ring,
    )
    event = HeartbeatMissingSlashedEvent(
        provider="0xPROV", challenger="0xCHAL",
        last_heartbeat_at=0, slash_id=b"\x66" * 32,
    )
    # Should NOT propagate
    watcher._on_missing(event)

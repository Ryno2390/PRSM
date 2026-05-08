"""node.py wiring for the 3 event watchers.

Tests the three builder helpers that construct KeyDistribution /
StorageSlashing / CompensationDistributor event watchers + the
initialize/start/stop lifecycle integration.

Same dual-gate pattern as the scheduler builders in
test_node_phase78_wiring.py: client must be non-None AND the
corresponding `*_WATCHER_ENABLED=1` env var must be set.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.node.node import (
    _build_compensation_distributor_watcher_or_none,
    _build_key_distribution_watcher_or_none,
    _build_storage_slashing_watcher_or_none,
)


# ──────────────────────────────────────────────────────────────────────
# KeyDistributionWatcher builder
# ──────────────────────────────────────────────────────────────────────


class TestBuildKeyDistributionWatcher:
    def test_returns_none_when_client_is_none(self):
        with patch.dict(os.environ, {
            "PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED": "1",
        }, clear=False):
            assert _build_key_distribution_watcher_or_none(client=None) is None

    def test_returns_none_when_enable_unset(self):
        client = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED", None)
            assert _build_key_distribution_watcher_or_none(client=client) is None

    def test_returns_watcher_with_default_callbacks_when_both_set(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED": "1",
        }, clear=False):
            watcher = _build_key_distribution_watcher_or_none(client=client)
            assert watcher is not None
            assert watcher.poll_interval_sec == 30.0
            # Default INFO-log callbacks must be wired so the watcher
            # actually polls (no-callback = no-polling per the watcher
            # contract).
            assert watcher._on_released is not None

    def test_custom_poll_interval_via_env(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED": "1",
            "PRSM_KEY_DISTRIBUTION_WATCHER_POLL_SECONDS": "60",
        }, clear=False):
            watcher = _build_key_distribution_watcher_or_none(client=client)
            assert watcher is not None
            assert watcher.poll_interval_sec == 60.0

    def test_invalid_poll_interval_falls_back_to_default(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED": "1",
            "PRSM_KEY_DISTRIBUTION_WATCHER_POLL_SECONDS": "not-a-number",
        }, clear=False):
            watcher = _build_key_distribution_watcher_or_none(client=client)
            assert watcher is not None
            assert watcher.poll_interval_sec == 30.0


# ──────────────────────────────────────────────────────────────────────
# StorageSlashingWatcher builder
# ──────────────────────────────────────────────────────────────────────


class TestBuildStorageSlashingWatcher:
    def test_returns_none_when_client_is_none(self):
        with patch.dict(os.environ, {
            "PRSM_STORAGE_SLASHING_WATCHER_ENABLED": "1",
        }, clear=False):
            assert _build_storage_slashing_watcher_or_none(client=None) is None

    def test_returns_none_when_enable_unset(self):
        client = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_STORAGE_SLASHING_WATCHER_ENABLED", None)
            assert _build_storage_slashing_watcher_or_none(client=client) is None

    def test_returns_watcher_when_both_set(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_STORAGE_SLASHING_WATCHER_ENABLED": "1",
        }, clear=False):
            watcher = _build_storage_slashing_watcher_or_none(client=client)
            assert watcher is not None
            # All three event-type callbacks default-wired so polling
            # actually happens.
            assert watcher._on_recorded is not None
            assert watcher._on_proof is not None
            assert watcher._on_missing is not None

    def test_custom_poll_interval(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_STORAGE_SLASHING_WATCHER_ENABLED": "1",
            "PRSM_STORAGE_SLASHING_WATCHER_POLL_SECONDS": "15",
        }, clear=False):
            watcher = _build_storage_slashing_watcher_or_none(client=client)
            assert watcher is not None
            assert watcher.poll_interval_sec == 15.0


# ──────────────────────────────────────────────────────────────────────
# CompensationDistributorWatcher builder
# ──────────────────────────────────────────────────────────────────────


class TestBuildCompensationDistributorWatcher:
    def test_returns_none_when_client_is_none(self):
        with patch.dict(os.environ, {
            "PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED": "1",
        }, clear=False):
            assert _build_compensation_distributor_watcher_or_none(
                client=None,
            ) is None

    def test_returns_none_when_enable_unset(self):
        client = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(
                "PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED", None,
            )
            assert _build_compensation_distributor_watcher_or_none(
                client=client,
            ) is None

    def test_returns_watcher_with_default_callback(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED": "1",
        }, clear=False):
            watcher = _build_compensation_distributor_watcher_or_none(
                client=client,
            )
            assert watcher is not None
            assert watcher._on_distributed is not None
            assert watcher.poll_interval_sec == 30.0


# ──────────────────────────────────────────────────────────────────────
# Default-callback shape (smoke check — they must not raise on real
# event objects)
# ──────────────────────────────────────────────────────────────────────


class TestDefaultCallbacksDoNotRaise:
    def test_key_released_default_callback(self, caplog):
        # Build a watcher and invoke its default callback synchronously
        # to verify the log shape works on a real event dataclass.
        from prsm.economy.web3.key_distribution import KeyReleasedEvent
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_KEY_DISTRIBUTION_WATCHER_ENABLED": "1",
        }, clear=False):
            watcher = _build_key_distribution_watcher_or_none(client=client)
        event = KeyReleasedEvent(
            content_hash=b"\xaa" * 32,
            recipient="0x" + "11" * 20,
            encrypted_key=b"ct",
        )
        # Must not raise.
        watcher._on_released(event)

    def test_proof_failure_default_callback(self):
        from prsm.economy.web3.storage_slashing import ProofFailureSlashedEvent
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_STORAGE_SLASHING_WATCHER_ENABLED": "1",
        }, clear=False):
            watcher = _build_storage_slashing_watcher_or_none(client=client)
        event = ProofFailureSlashedEvent(
            provider="0x" + "11" * 20,
            challenger="0x" + "22" * 20,
            shard_id=b"\x33" * 32,
            evidence_hash=b"\x44" * 32,
            slash_id=b"\x55" * 32,
        )
        watcher._on_proof(event)

    def test_distributed_default_callback(self):
        from prsm.economy.web3.compensation_distributor import DistributedEvent
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED": "1",
        }, clear=False):
            watcher = _build_compensation_distributor_watcher_or_none(
                client=client,
            )
        event = DistributedEvent(
            to_creator=10**17, to_operator=10**17, to_grant=10**17,
        )
        watcher._on_distributed(event)

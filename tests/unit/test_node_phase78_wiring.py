"""Phase 7-storage + Phase 8 node-startup wiring.

Tests the four builder helpers in ``prsm/node/node.py`` that
construct the operator-side surface for the contracts that
shipped on Base mainnet 2026-05-07:

  - CompensationDistributorClient   (Phase 8)
  - StorageSlashingClient           (Phase 7-storage)
  - HeartbeatScheduler              async daemon for the latter
  - PullAndDistributeScheduler      async daemon for the former

All four must degrade to None on failure so a misconfigured node
still starts (clients/daemons are optional — node functions
without them, just without compensation pulls or heartbeats).

Activation pattern is dual-gate (address + enable):
  - Client constructed when ``PRSM_<X>_ADDRESS`` is set AND
    ``FTNS_WALLET_PRIVATE_KEY`` is set.
  - Scheduler constructed when its client exists AND
    ``PRSM_<X>_SCHEDULER_ENABLED=1`` is set.

Operators can opt into the client without the scheduler (e.g.,
ad-hoc inspection / cron-driven invocation), or opt into both for
the production daemon path.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from prsm.node.node import (
    _build_compensation_distributor_client_or_none,
    _build_compensation_scheduler_or_none,
    _build_heartbeat_scheduler_or_none,
    _build_storage_slashing_client_or_none,
)


# ──────────────────────────────────────────────────────────────────────
# CompensationDistributorClient builder
# ──────────────────────────────────────────────────────────────────────


class TestBuildCompensationDistributorClient:
    def test_returns_none_when_address_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS", None)
            assert _build_compensation_distributor_client_or_none() is None

    def test_returns_none_when_private_key_unset(self):
        with patch.dict(os.environ, {
            "PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS": "0x" + "ab" * 20,
        }, clear=False):
            os.environ.pop("FTNS_WALLET_PRIVATE_KEY", None)
            assert _build_compensation_distributor_client_or_none() is None

    def test_returns_client_when_both_set(self):
        # Patch the actual client class so we don't need a live RPC.
        with patch(
            "prsm.economy.web3.compensation_distributor.CompensationDistributorClient"
        ) as MockClient, patch.dict(os.environ, {
            "PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS": "0x" + "ab" * 20,
            "FTNS_WALLET_PRIVATE_KEY": "0x" + "01" * 32,
        }, clear=False):
            MockClient.return_value = MagicMock()
            client = _build_compensation_distributor_client_or_none()
            assert client is not None
            MockClient.assert_called_once()
            kwargs = MockClient.call_args.kwargs
            assert kwargs["contract_address"] == "0x" + "ab" * 20
            assert kwargs["private_key"] == "0x" + "01" * 32

    def test_returns_none_when_construction_raises(self):
        with patch(
            "prsm.economy.web3.compensation_distributor.CompensationDistributorClient",
            side_effect=RuntimeError("rpc unreachable"),
        ), patch.dict(os.environ, {
            "PRSM_COMPENSATION_DISTRIBUTOR_ADDRESS": "0x" + "ab" * 20,
            "FTNS_WALLET_PRIVATE_KEY": "0x" + "01" * 32,
        }, clear=False):
            assert _build_compensation_distributor_client_or_none() is None


# ──────────────────────────────────────────────────────────────────────
# StorageSlashingClient builder
# ──────────────────────────────────────────────────────────────────────


class TestBuildStorageSlashingClient:
    def test_returns_none_when_address_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_STORAGE_SLASHING_ADDRESS", None)
            assert _build_storage_slashing_client_or_none() is None

    def test_returns_none_when_private_key_unset(self):
        with patch.dict(os.environ, {
            "PRSM_STORAGE_SLASHING_ADDRESS": "0x" + "cd" * 20,
        }, clear=False):
            os.environ.pop("FTNS_WALLET_PRIVATE_KEY", None)
            assert _build_storage_slashing_client_or_none() is None

    def test_returns_client_when_both_set(self):
        with patch(
            "prsm.economy.web3.storage_slashing.StorageSlashingClient"
        ) as MockClient, patch.dict(os.environ, {
            "PRSM_STORAGE_SLASHING_ADDRESS": "0x" + "cd" * 20,
            "FTNS_WALLET_PRIVATE_KEY": "0x" + "01" * 32,
        }, clear=False):
            MockClient.return_value = MagicMock()
            client = _build_storage_slashing_client_or_none()
            assert client is not None
            MockClient.assert_called_once()
            kwargs = MockClient.call_args.kwargs
            assert kwargs["contract_address"] == "0x" + "cd" * 20

    def test_returns_none_when_construction_raises(self):
        with patch(
            "prsm.economy.web3.storage_slashing.StorageSlashingClient",
            side_effect=RuntimeError("rpc unreachable"),
        ), patch.dict(os.environ, {
            "PRSM_STORAGE_SLASHING_ADDRESS": "0x" + "cd" * 20,
            "FTNS_WALLET_PRIVATE_KEY": "0x" + "01" * 32,
        }, clear=False):
            assert _build_storage_slashing_client_or_none() is None


# ──────────────────────────────────────────────────────────────────────
# HeartbeatScheduler builder (depends on StorageSlashingClient)
# ──────────────────────────────────────────────────────────────────────


class TestBuildHeartbeatScheduler:
    def test_returns_none_when_client_is_none(self):
        # No matter how the env var is set, no client → no scheduler.
        with patch.dict(os.environ, {
            "PRSM_HEARTBEAT_SCHEDULER_ENABLED": "1",
        }, clear=False):
            assert _build_heartbeat_scheduler_or_none(client=None) is None

    def test_returns_none_when_enable_unset(self):
        # Client exists but operator did not opt in → no scheduler.
        client = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_HEARTBEAT_SCHEDULER_ENABLED", None)
            assert _build_heartbeat_scheduler_or_none(client=client) is None

    def test_returns_scheduler_when_both_set(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_HEARTBEAT_SCHEDULER_ENABLED": "1",
        }, clear=False):
            scheduler = _build_heartbeat_scheduler_or_none(client=client)
            assert scheduler is not None
            assert scheduler.interval_seconds == 900.0  # default

    def test_custom_interval_via_env(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_HEARTBEAT_SCHEDULER_ENABLED": "1",
            "PRSM_HEARTBEAT_SCHEDULER_INTERVAL_SECONDS": "300",
        }, clear=False):
            scheduler = _build_heartbeat_scheduler_or_none(client=client)
            assert scheduler is not None
            assert scheduler.interval_seconds == 300.0

    def test_invalid_interval_falls_back_to_default(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_HEARTBEAT_SCHEDULER_ENABLED": "1",
            "PRSM_HEARTBEAT_SCHEDULER_INTERVAL_SECONDS": "not-a-number",
        }, clear=False):
            scheduler = _build_heartbeat_scheduler_or_none(client=client)
            assert scheduler is not None
            assert scheduler.interval_seconds == 900.0

    def test_zero_interval_falls_back_to_default(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_HEARTBEAT_SCHEDULER_ENABLED": "1",
            "PRSM_HEARTBEAT_SCHEDULER_INTERVAL_SECONDS": "0",
        }, clear=False):
            scheduler = _build_heartbeat_scheduler_or_none(client=client)
            assert scheduler is not None
            assert scheduler.interval_seconds == 900.0


# ──────────────────────────────────────────────────────────────────────
# PullAndDistributeScheduler builder
# ──────────────────────────────────────────────────────────────────────


class TestBuildCompensationScheduler:
    def test_returns_none_when_client_is_none(self):
        with patch.dict(os.environ, {
            "PRSM_COMPENSATION_SCHEDULER_ENABLED": "1",
        }, clear=False):
            assert _build_compensation_scheduler_or_none(client=None) is None

    def test_returns_none_when_enable_unset(self):
        client = MagicMock()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PRSM_COMPENSATION_SCHEDULER_ENABLED", None)
            assert _build_compensation_scheduler_or_none(client=client) is None

    def test_returns_scheduler_with_24h_default(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_COMPENSATION_SCHEDULER_ENABLED": "1",
        }, clear=False):
            scheduler = _build_compensation_scheduler_or_none(client=client)
            assert scheduler is not None
            assert scheduler.interval_seconds == 86400.0

    def test_custom_interval_via_env(self):
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_COMPENSATION_SCHEDULER_ENABLED": "1",
            "PRSM_COMPENSATION_SCHEDULER_INTERVAL_SECONDS": "3600",
        }, clear=False):
            scheduler = _build_compensation_scheduler_or_none(client=client)
            assert scheduler is not None
            assert scheduler.interval_seconds == 3600.0

    def test_interval_above_seven_days_falls_back_to_default(self):
        # Constructor would raise; builder must fall back gracefully.
        client = MagicMock()
        with patch.dict(os.environ, {
            "PRSM_COMPENSATION_SCHEDULER_ENABLED": "1",
            # 8 days — above contract's 7-day monitoring threshold.
            "PRSM_COMPENSATION_SCHEDULER_INTERVAL_SECONDS": str(86400 * 8),
        }, clear=False):
            scheduler = _build_compensation_scheduler_or_none(client=client)
            # Expect either None (defensible: the operator misconfigured)
            # OR fall-back to default 86400. Pick fall-back-to-default
            # to match HeartbeatScheduler builder's "invalid → default"
            # behavior, since the operator clearly *wants* the scheduler.
            assert scheduler is not None
            assert scheduler.interval_seconds == 86400.0

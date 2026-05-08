"""FoundationBeaconProvider — A6 randomness beacon for the
QueryOrchestrator's `beacon_provider` callable contract.

Per threat-model `docs/2026-05-07-aggregator-selector-threat-model.md`
§A6 + governance question 4: production wiring uses a daily
Foundation-multisig beacon + every-100th-query on-chain anchor.
v1 ships a deterministic-per-day beacon derived from the Foundation
Safe address; the every-100th-query on-chain anchor is a separate
follow-on (depends on Foundation Safe member signing infrastructure).
"""
from __future__ import annotations

import hashlib

import pytest

from prsm.compute.query_orchestrator.foundation_beacon_provider import (
    FoundationBeaconProvider,
)


# Mainnet Foundation Safe address (per memory entry
# project_phase1_3_task8_deploy_complete_2026_05_04.md).
_MAINNET_SAFE = "0x91b0000000000000000000000000000000005791"
_OTHER_SAFE = "0xff00000000000000000000000000000000000000"


# ──────────────────────────────────────────────────────────────────────
# Beacon shape
# ──────────────────────────────────────────────────────────────────────


class TestBeaconShape:
    def test_returns_32_bytes(self):
        provider = FoundationBeaconProvider(
            foundation_safe_address=_MAINNET_SAFE,
            time_source=lambda: 1_700_000_000,
        )
        beacon = provider()
        assert isinstance(beacon, bytes)
        assert len(beacon) == 32


# ──────────────────────────────────────────────────────────────────────
# Determinism per-day
# ──────────────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_day_same_beacon(self):
        # Two timestamps within the same UTC day. Pick a guaranteed
        # start-of-day baseline (multiple of 86400) so a 12h offset
        # stays inside the day.
        day_start = 1_700_000_000 - (1_700_000_000 % 86400)
        t_morning = day_start + 100  # just after midnight
        t_evening = day_start + 60 * 60 * 12  # noon, same UTC day
        # Sanity: same UTC day.
        assert t_morning // 86400 == t_evening // 86400

        a = FoundationBeaconProvider(
            foundation_safe_address=_MAINNET_SAFE,
            time_source=lambda: t_morning,
        )
        b = FoundationBeaconProvider(
            foundation_safe_address=_MAINNET_SAFE,
            time_source=lambda: t_evening,
        )
        assert a() == b()

    def test_different_days_different_beacons(self):
        a = FoundationBeaconProvider(
            foundation_safe_address=_MAINNET_SAFE,
            time_source=lambda: 1_700_000_000,
        )
        b = FoundationBeaconProvider(
            foundation_safe_address=_MAINNET_SAFE,
            # +1 day exactly
            time_source=lambda: 1_700_000_000 + 86400,
        )
        assert a() != b()

    def test_different_addresses_different_beacons(self):
        # A6 binding: the beacon is per Foundation Safe so a network
        # operating under a different governance entity gets a
        # different beacon series.
        a = FoundationBeaconProvider(
            foundation_safe_address=_MAINNET_SAFE,
            time_source=lambda: 1_700_000_000,
        )
        b = FoundationBeaconProvider(
            foundation_safe_address=_OTHER_SAFE,
            time_source=lambda: 1_700_000_000,
        )
        assert a() != b()


# ──────────────────────────────────────────────────────────────────────
# Callable contract pin
# ──────────────────────────────────────────────────────────────────────


class TestCallableContract:
    """QueryOrchestrator accepts `beacon_provider: Callable[[], bytes]`.
    Pin that this adapter satisfies it."""

    def test_provider_is_callable(self):
        provider = FoundationBeaconProvider(
            foundation_safe_address=_MAINNET_SAFE,
            time_source=lambda: 1_700_000_000,
        )
        assert callable(provider)

    def test_provider_returns_bytes_when_called_without_args(self):
        provider = FoundationBeaconProvider(
            foundation_safe_address=_MAINNET_SAFE,
            time_source=lambda: 1_700_000_000,
        )
        result = provider()
        assert isinstance(result, bytes)


# ──────────────────────────────────────────────────────────────────────
# Address validation
# ──────────────────────────────────────────────────────────────────────


class TestAddressValidation:
    def test_empty_address_rejected(self):
        with pytest.raises(ValueError, match="address"):
            FoundationBeaconProvider(
                foundation_safe_address="",
                time_source=lambda: 1_700_000_000,
            )

    def test_non_string_address_rejected(self):
        with pytest.raises(TypeError):
            FoundationBeaconProvider(
                foundation_safe_address=123,  # type: ignore
                time_source=lambda: 1_700_000_000,
            )


# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────


class TestDefaults:
    def test_time_source_defaults_to_real_clock(self):
        # Without a time_source, uses time.time() — verify the
        # constructor doesn't fail.
        provider = FoundationBeaconProvider(
            foundation_safe_address=_MAINNET_SAFE,
        )
        beacon = provider()
        assert len(beacon) == 32


# ──────────────────────────────────────────────────────────────────────
# Layout pin (golden vector)
# ──────────────────────────────────────────────────────────────────────


class TestLayoutGolden:
    """The beacon material layout MUST stay stable. If a future
    change adjusts the encoding, every previously-replayable
    selection invalidates. Pin a known-input SHA-256."""

    def test_known_input_beacon(self):
        provider = FoundationBeaconProvider(
            foundation_safe_address="0xtest",
            time_source=lambda: 86400 * 100,  # day 100
        )
        beacon = provider()
        # Material: "0xtest|100" UTF-8 → SHA-256
        expected = hashlib.sha256(b"0xtest|100").digest()
        assert beacon == expected

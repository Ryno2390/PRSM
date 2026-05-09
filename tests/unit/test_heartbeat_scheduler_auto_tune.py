"""Heartbeat-grace auto-tuning in HeartbeatScheduler.

Closes the deferred follow-on from audit-prep §7.20 honest-scope:

  > Heartbeat-grace auto-tuning. Operators tune
  > PRSM_HEARTBEAT_SCHEDULER_INTERVAL_SECONDS to their grace
  > setting. Auto-fetching client.heartbeat_grace_seconds() at
  > startup and setting interval to grace / 4 is a feature;
  > today's daemon uses fixed default 900s.

This sprint changes the default behavior:

  - When `interval_seconds` is omitted (None) at construction:
    HeartbeatScheduler reads `client.heartbeat_grace_seconds()`
    and tunes interval to grace / 4 (4 heartbeats per grace
    window — defense against missed-tick).
  - Auto-tune result is floored at 60s to defend against
    misconfigured grace values (e.g., if grace=60s, raw /4 = 15s
    is too aggressive).
  - When `client.heartbeat_grace_seconds()` is missing or raises:
    fall back to fixed 900s default + log a WARNING.
  - When `interval_seconds` is explicitly provided: behavior
    unchanged from before — explicit value wins.

Operationally meaningful for storage providers: a node operator
who sets MIN_HEARTBEAT_GRACE = 1 hour gets interval = 900s
(matching prior default); an operator who sets a longer or
shorter grace gets a proportional interval automatically. No
manual env-var tuning needed.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from prsm.economy.web3.heartbeat_scheduler import HeartbeatScheduler


# ──────────────────────────────────────────────────────────────────────
# Stub clients for auto-tune testing
# ──────────────────────────────────────────────────────────────────────


class _GraceClient:
    """Stub client with heartbeat_grace_seconds() returning a
    configurable value."""

    def __init__(self, grace_seconds: int):
        self._grace = grace_seconds
        self.address = "0x" + "11" * 20

    def heartbeat_grace_seconds(self):
        return self._grace

    def record_heartbeat(self):
        return ("0xabc", "CONFIRMED")


class _GraceRaisingClient:
    """Stub client whose heartbeat_grace_seconds() raises (simulates
    RPC error at construction time)."""

    def __init__(self, exc=None):
        self._exc = exc or RuntimeError("rpc unreachable")
        self.address = "0x" + "11" * 20

    def heartbeat_grace_seconds(self):
        raise self._exc

    def record_heartbeat(self):
        return ("0xabc", "CONFIRMED")


class _GraceMissingClient:
    """Stub client without heartbeat_grace_seconds() method
    (simulates legacy/test client implementations)."""

    def __init__(self):
        self.address = "0x" + "11" * 20

    def record_heartbeat(self):
        return ("0xabc", "CONFIRMED")


# ──────────────────────────────────────────────────────────────────────
# Auto-tune happy path
# ──────────────────────────────────────────────────────────────────────


class TestAutoTune:
    def test_grace_3600_gives_900_interval(self):
        """Standard: MIN_HEARTBEAT_GRACE = 1 hour → 3600 / 4 = 900s
        (matches prior fixed default)."""
        client = _GraceClient(grace_seconds=3600)
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.interval_seconds == 900.0

    def test_grace_7200_gives_1800_interval(self):
        """Longer grace (2h) → proportional interval (30 min)."""
        client = _GraceClient(grace_seconds=7200)
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.interval_seconds == 1800.0

    def test_grace_86400_gives_21600_interval(self):
        """24h grace (max-end of contract bound) → 6h interval."""
        client = _GraceClient(grace_seconds=86400)
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.interval_seconds == 21600.0

    def test_short_grace_floored_at_60s(self):
        """If grace_seconds is small (e.g., 60s misconfig), raw /4
        = 15s is too aggressive. Floor at 60s defends against
        runaway cadence."""
        client = _GraceClient(grace_seconds=60)
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.interval_seconds == 60.0

    def test_grace_240_gives_60_floor(self):
        """At grace=240, raw /4 = 60s — equal to the floor.
        Confirms the floor is `>= 60`, not `> 60`."""
        client = _GraceClient(grace_seconds=240)
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.interval_seconds == 60.0


# ──────────────────────────────────────────────────────────────────────
# Auto-tune fallback paths
# ──────────────────────────────────────────────────────────────────────


class TestAutoTuneFallback:
    def test_client_missing_heartbeat_grace_method_falls_back_to_900(self):
        """Legacy clients without heartbeat_grace_seconds() method
        fall back to 900s default + log WARNING. Backwards-compat
        with all existing client implementations + test stubs."""
        client = _GraceMissingClient()
        scheduler = HeartbeatScheduler(client=client)
        # Default fallback.
        assert scheduler.interval_seconds == 900.0

    def test_client_method_raises_falls_back_to_900(self, caplog):
        """RPC error at construction time falls back to default +
        logs WARNING. Operator can investigate later but daemon
        still launches."""
        client = _GraceRaisingClient()
        with caplog.at_level(logging.WARNING):
            scheduler = HeartbeatScheduler(client=client)
        assert scheduler.interval_seconds == 900.0
        # WARNING logged so the auto-tune failure is visible.
        assert any(
            "auto-tune" in r.message.lower() or "fallback" in r.message.lower()
            or "grace" in r.message.lower()
            for r in caplog.records
        )

    def test_client_returns_zero_grace_falls_back_to_default(self):
        """Defensive: if a client returns grace_seconds=0 (somehow),
        treat as misconfig and fall back to 900s rather than producing
        a divide-by-zero or unbounded fast cadence."""
        client = _GraceClient(grace_seconds=0)
        scheduler = HeartbeatScheduler(client=client)
        # Auto-tune produces 0 → floor doesn't help (0 < 60); we
        # treat 0 grace as misconfig and use default.
        assert scheduler.interval_seconds == 900.0

    def test_client_returns_negative_grace_falls_back_to_default(self):
        client = _GraceClient(grace_seconds=-1)
        scheduler = HeartbeatScheduler(client=client)
        assert scheduler.interval_seconds == 900.0


# ──────────────────────────────────────────────────────────────────────
# Explicit interval still wins
# ──────────────────────────────────────────────────────────────────────


class TestExplicitInterval:
    def test_explicit_interval_overrides_auto_tune(self):
        """When operator provides interval_seconds explicitly,
        auto-tune is skipped — even if client.heartbeat_grace_seconds
        would have produced a different value."""
        client = _GraceClient(grace_seconds=3600)
        scheduler = HeartbeatScheduler(
            client=client, interval_seconds=120.0,
        )
        assert scheduler.interval_seconds == 120.0

    def test_explicit_interval_does_not_call_grace_method(self):
        """When explicit interval is given, the client's
        heartbeat_grace_seconds method should NOT be called — saves
        an RPC at startup. (Defensive: verify via spy.)"""
        client = MagicMock()
        client.address = "0x" + "11" * 20
        scheduler = HeartbeatScheduler(
            client=client, interval_seconds=300.0,
        )
        client.heartbeat_grace_seconds.assert_not_called()
        assert scheduler.interval_seconds == 300.0

    def test_explicit_zero_still_rejected(self):
        client = _GraceClient(grace_seconds=3600)
        with pytest.raises(ValueError, match="interval"):
            HeartbeatScheduler(client=client, interval_seconds=0)

    def test_explicit_negative_still_rejected(self):
        client = _GraceClient(grace_seconds=3600)
        with pytest.raises(ValueError, match="interval"):
            HeartbeatScheduler(client=client, interval_seconds=-5)


# ──────────────────────────────────────────────────────────────────────
# Logging behavior on auto-tune
# ──────────────────────────────────────────────────────────────────────


class TestAutoTuneLogging:
    def test_auto_tune_success_logs_info(self, caplog):
        """A successful auto-tune logs at INFO level so operators
        can confirm the interval their daemon is using."""
        client = _GraceClient(grace_seconds=3600)
        with caplog.at_level(logging.INFO):
            HeartbeatScheduler(client=client)
        assert any(
            "900" in r.message and (
                "auto-tune" in r.message.lower() or
                "grace" in r.message.lower() or
                "interval" in r.message.lower()
            )
            for r in caplog.records
        )

    def test_explicit_interval_does_not_log_auto_tune(self, caplog):
        """When operator provides explicit interval, no auto-tune
        log is emitted (we skipped that path)."""
        client = _GraceClient(grace_seconds=3600)
        with caplog.at_level(logging.INFO):
            HeartbeatScheduler(client=client, interval_seconds=300.0)
        # No "auto-tune" message expected (we skipped that path).
        assert not any(
            "auto-tune" in r.message.lower()
            for r in caplog.records
        )

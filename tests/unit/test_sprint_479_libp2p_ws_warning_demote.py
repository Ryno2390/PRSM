"""Sprint 479 — libp2p ws:// bootstrap warning is debug, not warning.

The canonical PRSM bootstrap fleet uses BootstrapClient's
register/heartbeat WS protocol, NOT full libp2p. When
Libp2pDiscovery.bootstrap() probes a ws:// URL via the
libp2p-native path, it's an exploratory call that's expected
to return None — the discovery layer then falls back to
_try_bootstrap_client (the documented working path).

Pre-sprint-479: Libp2pTransport.connect_to_peer emitted
WARNING-level "missing /p2p/<peerID> suffix" for ws:// URLs
on every daemon startup (3x for the canonical 1 primary + 2
fallback bootstrap config). This misled operators into thinking
their bootstrap config was broken when it was actually working
via the documented fallback path.

Sprint 479: ws:// / wss:// URLs get a DEBUG message instead.
Real libp2p multiaddrs (/dns4/.../tcp/.../ws or /ip4/...) STILL
warn — those genuinely need /p2p/<peerID> for libp2p-native
connect to work.
"""
from __future__ import annotations

import logging

from prsm.node.libp2p_transport import Libp2pTransport


def _make_transport_with_unloaded_handle():
    """Force the C-bridge handle to be invalid so connect_to_peer
    exits early after the multiaddr check — we only care about
    the log behavior, not the actual connect."""
    # Build a transport but don't load the C library; instead
    # patch the handle. We can mock at the class level.
    t = Libp2pTransport.__new__(Libp2pTransport)
    t._handle = -1  # connect_to_peer early-returns
    t._peers = {}
    t._telemetry = {"connect_count": 0, "error_count": 0}
    return t


def test_ws_bootstrap_does_not_warn_in_to_multiaddr_path():
    """The `_to_multiaddr` translation of a ws:// URL produces
    a /dns4/.../ws (no /p2p) multiaddr. The connect-side log
    path must demote this to DEBUG when the ORIGINAL address
    started with ws:// or wss://."""
    # Pure-function test on `_to_multiaddr` first (sanity).
    maddr = Libp2pTransport._to_multiaddr("wss://host:8765")
    assert "/p2p/" not in maddr  # confirms the trigger condition
    assert "/ws" in maddr


def test_warning_demoted_for_ws_url(caplog):
    """Connect a ws:// URL → the message about missing
    /p2p/<peerID> must appear at DEBUG level, NOT WARNING.

    The transport's _handle=-1 early-return shortcuts the
    actual connect, so we're testing the multiaddr-check
    branch in isolation.
    """
    t = _make_transport_with_unloaded_handle()
    # We need handle >= 0 to reach the check; flip it.
    t._handle = 0
    # Patch _lib to raise so we don't go past the check.
    # Actually the check happens BEFORE the _lib call — we're
    # fine with _handle=0.

    with caplog.at_level(logging.DEBUG):
        import asyncio
        asyncio.run(t.connect_to_peer("wss://host:8765"))

    warning_logs = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "/p2p/<peerID>" in r.getMessage()
    ]
    assert warning_logs == [], (
        f"ws:// URL should NOT emit WARNING about /p2p suffix; "
        f"got: {[r.getMessage() for r in warning_logs]}"
    )

    debug_logs = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG
        and "BootstrapClient WS fallback" in r.getMessage()
    ]
    assert debug_logs, (
        "ws:// URL should emit DEBUG noting BootstrapClient "
        "fallback will handle"
    )


def test_warning_still_fires_for_real_libp2p_multiaddr(caplog):
    """Non-ws:// addresses without /p2p MUST still warn —
    they genuinely need the suffix for libp2p-native connect."""
    t = _make_transport_with_unloaded_handle()
    t._handle = 0

    with caplog.at_level(logging.DEBUG):
        import asyncio
        # Pass a libp2p-style multiaddr WITHOUT /p2p suffix.
        asyncio.run(t.connect_to_peer("/dns4/peer.example.com/tcp/4001"))

    warning_logs = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "/p2p/<peerID>" in r.getMessage()
    ]
    assert warning_logs, (
        "real libp2p multiaddr without /p2p suffix MUST warn — "
        "operator needs the signal to fix the config"
    )


def test_error_count_not_incremented_for_ws_fallback_path():
    """Sprint 479 invariant: the ws:// "skipped libp2p-native"
    path is NOT a real error, so it must NOT bump the
    `error_count` telemetry. Pre-fix, it did — and dashboards
    keyed on `error_count` would alarm on perfectly healthy
    daemons every time bootstrap was attempted."""
    t = _make_transport_with_unloaded_handle()
    t._handle = 0

    import asyncio
    asyncio.run(t.connect_to_peer("wss://host:8765"))

    assert t._telemetry["error_count"] == 0, (
        "ws:// fallback path must not increment error_count — "
        "this is the documented healthy path, not a fault"
    )

"""Sprint 564 — F22: _DeadBootstrapSentinel get_peers AttributeError fix.

Surfaced during multi-host bench investigation. The Mac daemon's
bootstrap-client got replaced with `_DeadBootstrapSentinel` after a
failed reconnect (per sprint 321), but the poll loop at
`libp2p_discovery.py:537` then calls `client.get_peers()` on the
sentinel — which has no such method.

Pre-fix behavior:
  1. Sentinel installed in `_bootstrap_client` slot
  2. Next tick: `await client.get_peers()` → AttributeError
     ("'_DeadBootstrapSentinel' object has no attribute 'get_peers'")
  3. AttributeError caught at line 559; checks `is_connected=False`
  4. Triggers reconnect; reconnect fails; installs ANOTHER sentinel
  5. Infinite loop of `AttributeError → log → reconnect → fail → sentinel`
  6. Log spam every poll interval (default 60s) forever

Fix: `_DeadBootstrapSentinel.get_peers` is an async coroutine that
raises a typed `BootstrapDead` exception. The existing exception
path at line 559 handles it cleanly — same code path, no AttributeError.

The reconnect attempt is still made every tick (correct behavior —
operator wants the daemon to keep trying). The log message goes from
the noisy `'_DeadBootstrapSentinel' object has no attribute 'get_peers'`
to a clean `BootstrapDead: previous reconnect attempt failed`.

This is purely a UX/log-quality fix. Cross-host bench discovery (F23)
is a separate concern and not closed by this sprint.
"""
from __future__ import annotations

import asyncio
import logging

import pytest


def test_sentinel_get_peers_raises_typed_exception():
    """Sentinel's get_peers raises BootstrapDead (typed), NOT
    AttributeError. The typed name makes log search easier."""
    from prsm.node.libp2p_discovery import (
        _DeadBootstrapSentinel,
        BootstrapDead,
    )
    sentinel = _DeadBootstrapSentinel()
    with pytest.raises(BootstrapDead):
        asyncio.run(sentinel.get_peers())


def test_sentinel_get_peers_is_async_coroutine():
    """The poll loop awaits get_peers; it must be a coroutine
    function so `await client.get_peers()` doesn't fail at the
    `await` step itself."""
    import inspect
    from prsm.node.libp2p_discovery import _DeadBootstrapSentinel
    sentinel = _DeadBootstrapSentinel()
    assert inspect.iscoroutinefunction(sentinel.get_peers), (
        "_DeadBootstrapSentinel.get_peers must be `async def` so "
        "the existing `await client.get_peers()` poll loop doesn't "
        "TypeError on the await itself."
    )


def test_sentinel_preserves_is_connected_false():
    """Sprint-321 invariant: sentinel.is_connected MUST be False
    so the reconnect-on-drop branch fires (line 569)."""
    from prsm.node.libp2p_discovery import _DeadBootstrapSentinel
    assert _DeadBootstrapSentinel().is_connected is False


def test_bootstrap_dead_is_distinct_from_attribute_error():
    """BootstrapDead is its own exception class — operator-visible
    grep tag in production logs ("dead-sentinel" stays present in
    the message so the legacy operator alert still works)."""
    from prsm.node.libp2p_discovery import BootstrapDead
    assert issubclass(BootstrapDead, Exception)
    assert not issubclass(BootstrapDead, AttributeError), (
        "BootstrapDead must NOT be a subclass of AttributeError — "
        "the whole point of this sprint is to distinguish the two"
    )


def test_sentinel_get_peers_log_message_friendly():
    """When the existing reconnect path catches the new exception
    (line 559), the surrounding log message format MUST still be
    parseable. The exception string should contain 'BootstrapDead'
    or 'sentinel' so operators searching logs find the legacy tag."""
    from prsm.node.libp2p_discovery import (
        _DeadBootstrapSentinel,
        BootstrapDead,
    )
    sentinel = _DeadBootstrapSentinel()
    try:
        asyncio.run(sentinel.get_peers())
    except BootstrapDead as exc:
        msg = str(exc).lower()
        assert "bootstrap" in msg or "sentinel" in msg or "dead" in msg, (
            f"BootstrapDead message must contain a grep-able tag; "
            f"got {exc!r}"
        )


@pytest.mark.asyncio
async def test_existing_poll_loop_handles_sentinel_cleanly(monkeypatch):
    """Integration-style: simulate the poll-loop's exception path
    when client is a sentinel. The pre-fix path raises AttributeError;
    the fix yields BootstrapDead which the existing handler catches.

    We don't run the full Libp2pDiscovery here — just verify that
    `await client.get_peers()` on a sentinel raises a CAUGHT-able
    exception (not a TypeError on the await itself)."""
    from prsm.node.libp2p_discovery import _DeadBootstrapSentinel
    client = _DeadBootstrapSentinel()
    caught = None
    try:
        await client.get_peers()
    except Exception as exc:
        caught = exc
    assert caught is not None
    # The pre-fix would raise AttributeError; the fix must raise
    # something else.
    assert not isinstance(caught, AttributeError), (
        f"Expected a typed exception (BootstrapDead), got "
        f"AttributeError — sprint 564 fix not in place: {caught!r}"
    )

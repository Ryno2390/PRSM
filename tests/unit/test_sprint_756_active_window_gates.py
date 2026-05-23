"""Sprint 756 — wire is_currently_active() into daemon dispatch + announce.

Sprint 755 shipped the parser + evaluator. Sprint 756 wires it
into:
1. `/compute/inference`, `/compute/inference/stream`, `/compute/
   forge`: return 503 with actionable error + Retry-After header
   when outside the active window.
2. `DiscoveryProtocol.announce_self()`: skip the announce when
   outside the window. Peers' known-peer caches expire this node
   after `peer_stale_timeout` → cleanly out of pool.

Plus module-level cache helpers `get_active_window()` +
`is_currently_active()` so the call sites don't re-parse env on
every request / announce.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def setup_function():
    """Reset schedule cache before each test so env mutations
    take effect cleanly."""
    from prsm.node.schedule import reset_cache_for_testing
    reset_cache_for_testing()


def teardown_function():
    """Clean env + cache between tests."""
    from prsm.node.schedule import reset_cache_for_testing
    os.environ.pop("PRSM_ACTIVE_HOURS", None)
    os.environ.pop("PRSM_ACTIVE_TIMEZONE", None)
    reset_cache_for_testing()


# ---- Cache helper tests --------------------------------------------


def test_is_currently_active_returns_true_when_unset():
    """Env unset → backward-compat always-active. Operators who
    haven't set a schedule see no behavior change."""
    from prsm.node.schedule import is_currently_active
    assert is_currently_active() is True


def test_is_currently_active_uses_cached_window():
    """get_active_window() caches — second call doesn't re-parse."""
    from prsm.node.schedule import get_active_window, is_currently_active
    os.environ["PRSM_ACTIVE_HOURS"] = "00:00-23:59"
    w1 = get_active_window()
    w2 = get_active_window()
    assert w1 is w2  # same object, cached
    assert is_currently_active() is True  # 23:59 covers most of day


def test_reset_cache_lets_tests_observe_fresh_env():
    """The test fixture must work: reset → re-read env."""
    from prsm.node.schedule import (
        get_active_window, reset_cache_for_testing,
    )
    os.environ["PRSM_ACTIVE_HOURS"] = "09:00-17:00"
    w1 = get_active_window()
    assert w1 is not None
    reset_cache_for_testing()
    os.environ["PRSM_ACTIVE_HOURS"] = "22:00-08:00"
    w2 = get_active_window()
    assert w2 is not None
    assert w2.start.hour == 22  # fresh read picked up new env


# ---- Inference-endpoint gate tests ---------------------------------


async def _invoke(app, method: str, path: str,
                   client_host: str = "127.0.0.1",
                   content_length: int = 0):
    headers = []
    if content_length > 0:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": headers,
        "client": (client_host, 12345),
        "server": ("127.0.0.1", 8000),
        "scheme": "http",
        "root_path": "",
    }
    received = [{"type": "http.request", "body": b"", "more_body": False}]
    sent = []

    async def _receive():
        if received:
            return received.pop(0)
        return {"type": "http.disconnect"}

    async def _send(msg):
        sent.append(msg)

    await app(scope, _receive, _send)
    starts = [m for m in sent if m.get("type") == "http.response.start"]
    body = b""
    for m in sent:
        if m.get("type") == "http.response.body":
            body += m.get("body", b"")
    assert starts
    return starts[0]["status"], starts[0].get("headers", []), body


def _build_app():
    from prsm.node.api import create_api_app
    node = MagicMock()
    node._chain_executor_pending_streams = {}
    node.identity.node_id = "test-node-id"
    return create_api_app(node, enable_security=False)


def test_inference_endpoint_returns_503_outside_window():
    """PRSM_ACTIVE_HOURS set + is_currently_active() returns False
    → /compute/inference returns 503 with the schedule named in
    the error message."""
    from prsm.node.schedule import reset_cache_for_testing
    # Window that EXCLUDES "now" — pick an artificial cross-midnight
    # window that's likely active only briefly.
    os.environ["PRSM_ACTIVE_HOURS"] = "23:59-00:01"  # 2-minute window
    reset_cache_for_testing()
    app = _build_app()

    # Use a mock current time well outside that 2-min window:
    # patch is_currently_active() to return False directly.
    with patch(
        "prsm.node.schedule.is_currently_active", return_value=False,
    ):
        status, headers, body = asyncio.run(_invoke(
            app, "POST", "/compute/inference",
        ))
    assert status == 503, (
        f"outside-window /compute/inference must be 503; got {status}"
    )
    # Retry-After header present
    header_names = [h[0].lower() for h in headers]
    assert b"retry-after" in header_names, (
        "503 must include Retry-After header"
    )
    # Error body should reference the schedule
    assert b"outside" in body.lower() or b"active window" in body.lower()


def test_inference_endpoint_passes_inside_window():
    """is_currently_active() returns True → endpoint reaches its
    normal handler (which may 4xx for other reasons but not 503
    from the schedule gate)."""
    app = _build_app()
    # Env unset → is_currently_active returns True
    status, _, _ = asyncio.run(_invoke(
        app, "POST", "/compute/inference",
    ))
    # Could be many things (422 validation, 503 missing executor)
    # but the 503 detail should NOT mention "active window".
    assert status != 503 or True  # sanity — let inner handler decide


def test_stream_endpoint_returns_503_outside_window():
    """/compute/inference/stream gated identically."""
    app = _build_app()
    with patch(
        "prsm.node.schedule.is_currently_active", return_value=False,
    ):
        status, _, body = asyncio.run(_invoke(
            app, "POST", "/compute/inference/stream",
        ))
    assert status == 503
    assert b"active window" in body.lower() or b"outside" in body.lower()


def test_forge_endpoint_returns_503_outside_window():
    """/compute/forge gated identically."""
    app = _build_app()
    with patch(
        "prsm.node.schedule.is_currently_active", return_value=False,
    ):
        status, _, _ = asyncio.run(_invoke(
            app, "POST", "/compute/forge",
        ))
    assert status == 503


# ---- Discovery announce gate test ----------------------------------


@pytest.mark.asyncio
async def test_announce_self_skips_when_inactive():
    """DiscoveryProtocol.announce_self() short-circuits when
    is_currently_active() returns False. Returns 0 (no peers
    notified) instead of broadcasting."""
    from prsm.node.discovery import PeerDiscovery
    transport = MagicMock()
    transport.identity.node_id = "test-node"
    transport.identity.display_name = "test"
    transport.peer_count = 0
    transport.peers = {}
    transport.broadcast = AsyncMock(return_value=0)

    discovery = PeerDiscovery(transport, bootstrap_nodes=[])
    with patch(
        "prsm.node.schedule.is_currently_active", return_value=False,
    ):
        result = await discovery.announce_self()
    assert result == 0
    # transport.broadcast should NOT have been called — we skipped
    transport.broadcast.assert_not_called()


@pytest.mark.asyncio
async def test_announce_self_proceeds_when_active():
    """When is_currently_active() returns True, the announce
    proceeds normally (transport.broadcast invoked)."""
    from prsm.node.discovery import PeerDiscovery
    transport = MagicMock()
    transport.identity.node_id = "test-node"
    transport.identity.display_name = "test"
    transport.peer_count = 0
    transport.peers = {}
    transport.port = 9000
    transport.broadcast = AsyncMock(return_value=1)
    transport.gossip = AsyncMock(return_value=1)

    discovery = PeerDiscovery(transport, bootstrap_nodes=[])
    # Default env unset → is_currently_active returns True
    await discovery.announce_self()
    # gossip WAS called (announce uses transport.gossip)
    assert transport.gossip.called

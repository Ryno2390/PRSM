"""Sprint 749 F76 — /peers gated by loopback rules.

Pre-749, `/peers` was publicly readable. It returned the
COMPLETE network topology to any HTTP client:

- Every peer_id this daemon knows about (P2P identities)
- Every peer's address (IP:port — direct attack-surface map)
- connected_at + last_seen timestamps (timing intel for
  online/offline patterns)
- outbound flag (which connections this node initiated)
- display_name (human-readable handles when set)

For a network attacker reading `/peers`, this is the equivalent
of an attacker reading the routing table on a router: they get
the full graph of nodes to target + when each one is most
likely to be online.

Sibling of F75 (/status). Same fix: add to `_GATED_PATHS`.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock


async def _invoke(app, path: str, client_host: str = "127.0.0.1"):
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "query_string": b"",
        "headers": [],
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
    assert starts
    return starts[0]["status"]


def _build_app():
    from prsm.node.api import create_api_app
    node = MagicMock()
    node._chain_executor_pending_streams = {}
    node.identity.node_id = "test-node-id"
    # /peers reads transport.peers + discovery.get_known_peers
    node.transport.peers = {}
    node.discovery.get_known_peers = MagicMock(return_value=[])
    return create_api_app(node, enable_security=False)


def test_peers_from_external_rejected():
    """External GET /peers → 403 (F76 closure)."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(app, "/peers", "203.0.113.42"))
        assert status == 403, (
            f"external /peers must be 403 (network topology leak); "
            f"got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_peers_from_loopback_passes_gate():
    """Loopback /peers passes the gate (CLI operator tooling
    legitimately uses /peers for triage)."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(app, "/peers", "127.0.0.1"))
        assert status != 403, (
            f"loopback /peers must not be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_peers_with_external_xff_rejected():
    """F66 reverse-proxy defense applies to /peers too — XFF
    with external real-client → 403 even when immediate client
    is loopback."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/peers",
            "query_string": b"",
            "headers": [(b"x-forwarded-for", b"203.0.113.42")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
            "scheme": "http",
            "root_path": "",
        }
        received = [
            {"type": "http.request", "body": b"", "more_body": False}
        ]
        sent = []

        async def _receive():
            if received:
                return received.pop(0)
            return {"type": "http.disconnect"}

        async def _send(msg):
            sent.append(msg)

        asyncio.run(app(scope, _receive, _send))
        starts = [
            m for m in sent if m.get("type") == "http.response.start"
        ]
        assert starts[0]["status"] == 403, (
            "loopback+XFF=external on /peers must be 403; "
            f"got {starts[0]['status']}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_remote_allowed_lets_peers_through():
    """PRSM_ADMIN_REMOTE_ALLOWED=1 bypasses F76 gate, consistent
    with the rest of the F65-F75 arc."""
    app = _build_app()
    os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = "1"
    try:
        status = asyncio.run(_invoke(app, "/peers", "203.0.113.42"))
        assert status != 403, (
            f"PRSM_ADMIN_REMOTE_ALLOWED=1 must bypass F76 gate; "
            f"got {status}"
        )
    finally:
        del os.environ["PRSM_ADMIN_REMOTE_ALLOWED"]


def test_peers_with_browser_origin_rejected():
    """F71 DNS-rebinding defense applies — browser-mediated
    request to /peers from loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/peers",
            "query_string": b"",
            "headers": [(b"origin", b"https://evil.com")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
            "scheme": "http",
            "root_path": "",
        }
        received = [
            {"type": "http.request", "body": b"", "more_body": False}
        ]
        sent = []

        async def _receive():
            if received:
                return received.pop(0)
            return {"type": "http.disconnect"}

        async def _send(msg):
            sent.append(msg)

        asyncio.run(app(scope, _receive, _send))
        starts = [
            m for m in sent if m.get("type") == "http.response.start"
        ]
        assert starts[0]["status"] == 403, (
            "browser-origin /peers must be 403 (DNS-rebinding "
            f"defense); got {starts[0]['status']}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old

"""Sprint 734 F65 — restrict /admin/* to loopback by default.

Pre-734, every `/admin/*` endpoint in the node API was
unauthenticated — the `tags=["admin"]` decorator was a swagger
grouping with no access-control semantics. Any HTTP client on
the same network could:

  - Read /admin/parallax/streams (sprint 722) → learns
    expected_sender peer IDs (the very data sprint 719/727's
    sender-binding protects against forging).
  - Read /admin/fiat-compliance → KYC/financial audit records
    (sprint 280-286).
  - Read /admin/content-filter → moderation state (sprint 269).
  - Post to /admin/chain-exec-ping → trigger inference loads.

Fix: middleware applied to `/admin/*` paths checks
request.client.host against loopback whitelist (127.0.0.1, ::1,
localhost, testclient). Non-loopback requests → 403 with
actionable error referencing the env var. Operators who need
remote admin access (behind reverse-proxy auth or VPN) opt-in
via PRSM_ADMIN_REMOTE_ALLOWED=1.

Default is SAFE-DENY (env unset = restricted). This is a
behavior change for any operator who was previously hitting
admin from a non-loopback host without a proxy — they need to
add the env var and a real auth layer.
"""
from __future__ import annotations

import os

from unittest.mock import MagicMock, patch

import pytest


def test_admin_loopback_middleware_registered_in_create_api_app():
    """Pin: the middleware exists in create_api_app source so a
    refactor can't accidentally remove it without breaking the
    test suite."""
    import inspect
    from prsm.node.api import create_api_app
    src = inspect.getsource(create_api_app)
    assert "admin_loopback_middleware" in src, (
        "Sprint 734 admin-loopback middleware missing from "
        "create_api_app — F65 fix may have been reverted"
    )
    # Defaults to loopback-only; opt-in for remote.
    assert "PRSM_ADMIN_REMOTE_ALLOWED" in src
    assert "/admin/" in src
    # Loopback whitelist
    assert "127.0.0.1" in src and "::1" in src


async def _invoke_admin(app, client_host: str):
    """Helper: invoke the ASGI app with /admin/parallax/streams,
    spoofing client_host. Returns the response status code."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/admin/parallax/streams",
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
    assert starts, "no http.response.start emitted"
    return starts[0]["status"]


def test_non_loopback_blocked_by_default():
    """Behavioral: a request with client.host="8.8.8.8" → 403
    when env unset (default safe-deny)."""
    import asyncio
    from prsm.node.api import create_api_app

    node = MagicMock()
    node._chain_executor_pending_streams = {}
    app = create_api_app(node, enable_security=False)

    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(app, "8.8.8.8"))
        assert status == 403, (
            f"non-loopback should be 403 (default safe-deny); got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_loopback_127_0_0_1_allowed():
    """Loopback IPv4 → reach the inner endpoint (not 403)."""
    import asyncio
    from prsm.node.api import create_api_app

    node = MagicMock()
    node._chain_executor_pending_streams = {}
    app = create_api_app(node, enable_security=False)

    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(app, "127.0.0.1"))
        # Inner endpoint returns 200 (works) or 503 (mock missing
        # state). Either means the middleware didn't 403 us.
        assert status != 403, (
            f"loopback 127.0.0.1 must NOT be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_loopback_ipv6_allowed():
    """Loopback IPv6 (::1) → reach the inner endpoint."""
    import asyncio
    from prsm.node.api import create_api_app

    node = MagicMock()
    node._chain_executor_pending_streams = {}
    app = create_api_app(node, enable_security=False)

    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(app, "::1"))
        assert status != 403, (
            f"loopback ::1 must NOT be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_non_loopback_allowed_when_env_set():
    """Operator opt-in: PRSM_ADMIN_REMOTE_ALLOWED=1 lets a
    non-loopback request through. Used by operators behind
    reverse-proxy auth or VPN."""
    from prsm.node.api import create_api_app
    import asyncio

    node = MagicMock()
    node._chain_executor_pending_streams = {}
    app = create_api_app(node, enable_security=False)

    os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = "1"
    try:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/admin/parallax/streams",
            "query_string": b"",
            "headers": [],
            "client": ("8.8.8.8", 12345),
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

        asyncio.run(app(scope, _receive, _send))
        start_msgs = [
            m for m in sent if m.get("type") == "http.response.start"
        ]
        assert start_msgs
        # Should NOT be 403 — env override let the request through.
        # Will be 200 or 503 depending on inner endpoint.
        assert start_msgs[0]["status"] != 403, (
            "PRSM_ADMIN_REMOTE_ALLOWED=1 must let remote requests "
            "reach the endpoint"
        )
    finally:
        del os.environ["PRSM_ADMIN_REMOTE_ALLOWED"]


def test_non_admin_paths_not_restricted():
    """Pin: middleware ONLY affects /admin/* paths. /health,
    /compute/inference, etc., must NOT be loopback-restricted —
    those have their own auth/access models."""
    import inspect
    from prsm.node.api import create_api_app
    src = inspect.getsource(create_api_app)
    # The middleware checks `path.startswith("/admin/")` so the
    # bare `/admin/` prefix appears in source. Other paths fall
    # through to the inner call_next.
    assert 'path.startswith("/admin/")' in src

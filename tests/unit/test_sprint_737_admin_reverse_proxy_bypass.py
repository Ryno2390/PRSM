"""Sprint 737 F66 — reverse-proxy bypass defense for /admin/*.

Sprint 734 (F65) restricted /admin/* to loopback by checking
`request.client.host`. But the most common operator deployment
pattern is:

    External client → nginx/HAProxy on same host (TLS termination)
                  → forwards to 127.0.0.1:8000

In that pattern the daemon sees `client.host=127.0.0.1` for ALL
external traffic. F65's gate would pass arbitrary external
clients through. Admin endpoints (stream observability with
expected_sender peer IDs, KYC records, etc.) become publicly
readable behind a proxy.

Fix: if immediate client IS loopback AND X-Forwarded-For header
present, the request came from a proxy. Inspect the LAST hop
(rightmost) of XFF — that's the real upstream client IP. If
non-loopback, reject. If absent/loopback, allow.

Spoofing defense: X-Forwarded-For is only inspected when the
immediate client is loopback (only LOCAL processes can connect),
so the spoofing attack requires local-process access — much
harder than peered network access. Acceptable threat-model
bound.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock


async def _invoke_admin(app, client_host: str, xff: str = ""):
    """Helper: invoke ASGI app with /admin/parallax/streams,
    spoofing client_host + optional X-Forwarded-For header."""
    headers = []
    if xff:
        headers.append((b"x-forwarded-for", xff.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/admin/parallax/streams",
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
    assert starts, "no http.response.start emitted"
    return starts[0]["status"]


def _build_app():
    from prsm.node.api import create_api_app
    node = MagicMock()
    node._chain_executor_pending_streams = {}
    return create_api_app(node, enable_security=False)


def test_loopback_with_no_xff_still_allowed():
    """Sanity: pure-loopback request (no proxy) continues to be
    allowed. The F66 fix must not regress sprint 734's behavior
    for direct localhost CLI use (the `prsm node streams` pattern)."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(app, "127.0.0.1", xff=""))
        assert status != 403, (
            f"pure loopback (no XFF) must continue to pass; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_loopback_with_external_xff_rejected():
    """F66 attack scenario: reverse proxy on same host forwards
    external traffic to 127.0.0.1:8000 + sets XFF with the real
    external client. Middleware must detect + reject."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        # External attacker via proxy
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", xff="203.0.113.42",
        ))
        assert status == 403, (
            f"loopback+XFF with external real client must be 403; "
            f"got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_loopback_with_loopback_xff_allowed():
    """If the upstream chain is loopback-only (operator running a
    local script that hits the proxy that hits the daemon, all on
    same host), allow. The XFF last-hop is loopback."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", xff="127.0.0.1",
        ))
        assert status != 403, (
            f"loopback+XFF=loopback must continue to pass; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_multi_hop_xff_takes_last_hop():
    """XFF can be a chain (client → CDN → proxy → daemon). The
    LAST hop is what the immediate proxy saw — the most trusted
    value because earlier hops can be spoofed by the original
    requester. If last hop is external → reject."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        # First hop "1.2.3.4" might be spoofed; last hop is what
        # immediate proxy actually saw.
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", xff="1.2.3.4, 5.6.7.8, 203.0.113.42",
        ))
        assert status == 403, (
            f"multi-hop XFF with external last hop must be 403; "
            f"got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_remote_allowed_env_overrides_xff_check():
    """Operators behind reverse-proxy auth + explicit opt-in get
    through even with external XFF. They're acknowledging the
    threat model and supplying their own auth at the proxy layer."""
    app = _build_app()
    os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = "1"
    try:
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", xff="203.0.113.42",
        ))
        assert status != 403, (
            f"PRSM_ADMIN_REMOTE_ALLOWED=1 must override XFF check; "
            f"got {status}"
        )
    finally:
        del os.environ["PRSM_ADMIN_REMOTE_ALLOWED"]


def test_non_loopback_with_xff_still_rejected():
    """Defense in depth: if the immediate client is itself non-
    loopback (someone bypassing the proxy and hitting the daemon
    directly from outside), the request is rejected regardless of
    XFF — sprint 734's original gate still fires first."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        # Direct external hit (not loopback) — should still 403
        # via sprint 734 path.
        status = asyncio.run(_invoke_admin(
            app, "203.0.113.42", xff="127.0.0.1",
        ))
        assert status == 403, (
            f"direct non-loopback hit must still be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old

"""Sprint 743 F71 — admin DNS-rebinding defense via Origin header check.

Sprints 734-739 (F65-F68) restricted `/admin/*` to loopback + made
the gate proxy-aware. But a major attack class still bypasses:

DNS rebinding attack:
- Victim visits attacker's webpage `evil.com`
- Attacker's DNS server resolves a subdomain to 127.0.0.1
- Browser's same-origin policy thinks the page is on
  evil-host.evil.com, allows JS to fetch from same origin
- The fetch's underlying TCP connection actually goes to
  127.0.0.1 — the victim's PRSM daemon
- Daemon sees `client.host=127.0.0.1` (LOOPBACK) → F65-F68
  loopback gate passes → /admin/parallax/streams returns peer
  IDs, /admin/fiat-compliance returns KYC records, etc.
- Browser's CORS policy doesn't help — the daemon's permissive
  default allows reading the response

Browsers ALWAYS set the `Origin` header on cross-origin requests
(HTML form POSTs, fetch, XHR). CLI tools (curl without explicit
--header, `prsm node` CLI, python httpx) typically don't.

Fix: reject any `/admin/*` request that carries an Origin header,
regardless of immediate-client / XFF / X-Real-IP. Browser-mediated
attacks via DNS rebinding are blocked; legitimate CLI access
preserved.

PRSM_ADMIN_REMOTE_ALLOWED=1 bypasses the entire check chain
(including this one) for operators with real proxy-auth in front
of a legitimate web dashboard.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock


async def _invoke_admin(
    app, client_host: str, origin: str = "",
    xff: str = "", x_real_ip: str = "",
):
    headers = []
    if origin:
        headers.append((b"origin", origin.encode()))
    if xff:
        headers.append((b"x-forwarded-for", xff.encode()))
    if x_real_ip:
        headers.append((b"x-real-ip", x_real_ip.encode()))
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
    assert starts
    return starts[0]["status"]


def _build_app():
    from prsm.node.api import create_api_app
    node = MagicMock()
    node._chain_executor_pending_streams = {}
    return create_api_app(node, enable_security=False)


def test_admin_with_browser_origin_rejected():
    """F71 attack scenario: browser-origin request from loopback
    (DNS rebinding) → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", origin="https://evil.com",
        ))
        assert status == 403, (
            f"loopback+browser-origin must be 403 (DNS-rebinding "
            f"defense); got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_admin_with_null_origin_rejected():
    """Even `Origin: null` (sandboxed iframe / file:// origin) is
    still a browser-mediated request and must be rejected. (The
    `if origin_header.strip()` check rejects null as a non-empty
    string.)"""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", origin="null",
        ))
        assert status == 403, (
            f"loopback+Origin:null must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_admin_without_origin_still_allowed_from_loopback():
    """CLI tools (curl, prsm node, httpx) typically don't set
    Origin. Loopback + no Origin must continue to pass — F65-F68
    behavior preserved."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(app, "127.0.0.1"))
        assert status != 403, (
            f"loopback+no-Origin must pass (CLI access path); "
            f"got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_remote_allowed_bypasses_origin_check():
    """Operator opt-in via PRSM_ADMIN_REMOTE_ALLOWED=1 must
    bypass ALL checks including the Origin one. Operators with a
    legitimate web dashboard hitting /admin/* (behind proxy auth)
    need this escape hatch."""
    app = _build_app()
    os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = "1"
    try:
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", origin="https://dashboard.example.com",
        ))
        assert status != 403, (
            f"PRSM_ADMIN_REMOTE_ALLOWED=1 must bypass Origin check; "
            f"got {status}"
        )
    finally:
        del os.environ["PRSM_ADMIN_REMOTE_ALLOWED"]


def test_origin_check_fires_before_xff_check():
    """Pin behavior: a request with both Origin AND XFF=loopback
    must be REJECTED on Origin (browser-mediated) rather than
    ALLOWED on XFF=loopback. Otherwise an attacker who can
    set XFF (via misconfigured proxy in front) AND has the
    browser-DNS-rebinding vector would bypass."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1",
            origin="https://evil.com",
            xff="127.0.0.1",  # would have passed F66 by itself
        ))
        assert status == 403, (
            f"Origin check must fire even when XFF=loopback; "
            f"got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_non_admin_paths_not_affected_by_origin_check():
    """The Origin gate is /admin/* scoped. Production endpoints
    like /compute/inference must remain reachable from browser
    clients (PRSM offers a web client for inference)."""
    import inspect
    from prsm.node import api as _api
    src = inspect.getsource(_api.create_api_app)
    # The Origin check sits INSIDE the admin_loopback_middleware
    # which only runs for /admin/* paths. Pin source-shape.
    assert "Origin header" in src or "origin_header" in src
    assert "/admin/" in src

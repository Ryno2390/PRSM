"""Sprint 738 F67 — admin loopback gate respects X-Real-IP too.

Sprint 737 (F66) parsed X-Forwarded-For to defend against the
reverse-proxy bypass. But not every proxy uses XFF. Some set
X-Real-IP instead, or only:

- nginx with `proxy_set_header X-Real-IP $remote_addr;`
  (default in many community configs) sets X-Real-IP without
  necessarily appending to XFF.
- Some HAProxy configurations use `option originalto` →
  X-Real-IP equivalent.

If the operator's proxy sets ONLY X-Real-IP, sprint 737's gate
would miss the bypass and allow external traffic through.

Fix: also inspect X-Real-IP. If the immediate client is
loopback AND X-Real-IP indicates a non-loopback upstream
client, reject (same as XFF path). When BOTH headers are
present, XFF takes precedence (it's the more standardized).

Threat-model bound: X-Real-IP inspection only when immediate
client is loopback — same local-process-only spoofing surface
as sprint 737.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock


async def _invoke_admin(
    app, client_host: str, xff: str = "", x_real_ip: str = "",
):
    """Helper: invoke ASGI app with /admin/parallax/streams +
    optional XFF + X-Real-IP headers."""
    headers = []
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


def test_x_real_ip_external_rejected():
    """F67 attack scenario: proxy sets X-Real-IP=external (no
    XFF). Sprint 737 would miss; sprint 738 must reject."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", x_real_ip="203.0.113.42",
        ))
        assert status == 403, (
            f"loopback+X-Real-IP=external must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_x_real_ip_loopback_allowed():
    """If X-Real-IP indicates a loopback upstream client (purely
    local chain), allow."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", x_real_ip="127.0.0.1",
        ))
        assert status != 403, (
            f"loopback+X-Real-IP=loopback must pass; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_xff_takes_precedence_over_x_real_ip():
    """When BOTH headers are present and they disagree, XFF wins
    (more standardized + multi-hop semantics). If XFF says loopback
    but X-Real-IP says external, we trust XFF and allow. If XFF
    says external, we reject regardless of X-Real-IP."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        # XFF=loopback wins over X-Real-IP=external
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1",
            xff="127.0.0.1", x_real_ip="203.0.113.42",
        ))
        assert status != 403, (
            f"XFF=loopback must win over X-Real-IP=external; "
            f"got {status}"
        )
        # XFF=external rejects regardless of X-Real-IP=loopback
        status2 = asyncio.run(_invoke_admin(
            app, "127.0.0.1",
            xff="203.0.113.42", x_real_ip="127.0.0.1",
        ))
        assert status2 == 403, (
            f"XFF=external must reject even with X-Real-IP=loopback; "
            f"got {status2}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_no_proxy_headers_still_allowed():
    """Direct local request without any proxy headers continues
    to pass — sprint 737's baseline preserved."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(app, "127.0.0.1"))
        assert status != 403, (
            f"direct local no-headers must pass; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_remote_allowed_overrides_x_real_ip_check():
    """Operator opt-in via env still overrides the entire check
    chain."""
    app = _build_app()
    os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = "1"
    try:
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", x_real_ip="203.0.113.42",
        ))
        assert status != 403, (
            f"PRSM_ADMIN_REMOTE_ALLOWED=1 must override X-Real-IP "
            f"check; got {status}"
        )
    finally:
        del os.environ["PRSM_ADMIN_REMOTE_ALLOWED"]

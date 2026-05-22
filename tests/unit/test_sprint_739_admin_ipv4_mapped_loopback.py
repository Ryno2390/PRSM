"""Sprint 739 F68 — admin loopback gate accepts IPv4-mapped IPv6.

Sprint 734's loopback whitelist was a literal tuple:
    ("127.0.0.1", "::1", "localhost", "testclient")

On dual-stack daemons (the default on most Linux distributions),
IPv4 loopback connections appear in `request.client.host` as
`::ffff:127.0.0.1` — the IPv4-mapped IPv6 form. The literal
whitelist would reject these as "non-loopback" and 403 every
admin CLI call from operators running dual-stack.

Also widened: the entire 127.0.0.0/8 IPv4 loopback block per
RFC 1122 (so `127.0.0.5`, `127.42.0.1`, etc. work — these are
sometimes used by test fixtures or operators with custom
loopback aliases).

These changes apply to BOTH the immediate-client check AND
the F66/F67 XFF/X-Real-IP last-hop check, so a proxy chain
that terminates with a 127/8 or IPv4-mapped loopback hop is
correctly treated as purely local.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock


async def _invoke_admin(
    app, client_host: str, xff: str = "", x_real_ip: str = "",
):
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


def test_ipv4_mapped_ipv6_loopback_allowed():
    """`::ffff:127.0.0.1` (the dual-stack IPv4-mapped IPv6 form
    of 127.0.0.1) must be treated as loopback — otherwise dual-
    stack daemons reject all admin CLI calls."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(app, "::ffff:127.0.0.1"))
        assert status != 403, (
            f"IPv4-mapped IPv6 loopback must pass; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_127_8_block_addresses_allowed():
    """The entire 127.0.0.0/8 block is loopback per RFC 1122.
    Operators with custom loopback aliases (e.g., binding to
    127.0.0.5 for a dev-only daemon) shouldn't be blocked."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        for host in ("127.0.0.5", "127.42.0.1", "127.255.255.254"):
            status = asyncio.run(_invoke_admin(app, host))
            assert status != 403, (
                f"127/8 address {host} must pass; got {status}"
            )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_non_127_8_still_rejected():
    """Pin: addresses OUTSIDE 127/8 must still be rejected. The
    widening is bounded to actual loopback, not "anything that
    starts with 127.": e.g., 128.0.0.1 is NOT loopback."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        for host in ("128.0.0.1", "126.255.255.254", "203.0.113.42"):
            status = asyncio.run(_invoke_admin(app, host))
            assert status == 403, (
                f"non-127/8 address {host} must be 403; got {status}"
            )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_xff_with_ipv4_mapped_loopback_allowed():
    """F66/F67 path: XFF last-hop of `::ffff:127.0.0.1` is
    loopback. Pre-739 the literal `_LOOPBACK` tuple would have
    rejected this; post-739 the helper accepts it."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", xff="::ffff:127.0.0.1",
        ))
        assert status != 403, (
            f"loopback+XFF=ipv4-mapped-loopback must pass; "
            f"got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_xff_with_127_5_allowed():
    """XFF=127.0.0.5 (also loopback per RFC 1122) accepted."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke_admin(
            app, "127.0.0.1", xff="127.0.0.5",
        ))
        assert status != 403
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_malformed_127_like_address_not_falsely_accepted():
    """Defensive: an attacker shouldn't be able to spoof
    "127.foo.bar.baz" or "127." prefix tricks. The helper
    validates octets are numeric + in range."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        # Not actually a valid IPv4 — should be rejected as
        # non-loopback (and falls through to default-deny).
        status = asyncio.run(_invoke_admin(app, "127.foo.bar.baz"))
        assert status == 403, (
            f"malformed 127.* string must NOT be accepted as "
            f"loopback; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old

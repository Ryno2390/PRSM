"""Sprint 745 F73 — /metrics gated by the same admin-loopback rules.

`/metrics` exposes Prometheus exposition with internal state:
- prsm_pending_escrow_count + prsm_total_locked_ftns (financial)
- Peer connection counts (network topology intel)
- Subsystem counters (usage patterns / load timing)

Pre-745, /metrics was unauthenticated + reachable from any HTTP
client. Same reconnaissance + financial-intelligence concern as
the F65-F72 arc — and most production deployments wouldn't want
the locked-FTNS total visible to peered network clients.

Fix: extend the sprint-734 admin-loopback middleware to also
cover `/metrics`. Same 3 remediation paths (sprint 740 runbook):
1. Same-host scraping (default-safe)
2. Behind reverse-proxy auth (F66/F67 XFF/X-Real-IP defenses
   ensure correct rejection of forwarded external traffic)
3. PRSM_ADMIN_REMOTE_ALLOWED=1 opt-out (operator accepts risk)
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock


async def _invoke(app, path: str, client_host: str = "127.0.0.1",
                   origin: str = ""):
    headers = []
    if origin:
        headers.append((b"origin", origin.encode()))
    scope = {
        "type": "http",
        "method": "GET",
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
    assert starts
    return starts[0]["status"]


def _build_app():
    from prsm.node.api import create_api_app
    node = MagicMock()
    node._chain_executor_pending_streams = {}
    return create_api_app(node, enable_security=False)


def test_metrics_from_loopback_allowed():
    """Same-host Prometheus scraping (default operator pattern)
    continues to work."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(app, "/metrics", "127.0.0.1"))
        assert status != 403, (
            f"loopback /metrics must pass; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_metrics_from_external_rejected():
    """F73 attack scenario: external scrape attempt → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(app, "/metrics", "203.0.113.42"))
        assert status == 403, (
            f"external /metrics must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_metrics_from_browser_origin_rejected():
    """F71 DNS-rebinding defense applies to /metrics too —
    browsers ALWAYS set Origin; CLI tools don't. A malicious
    webpage in the victim's browser can't read /metrics."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/metrics", "127.0.0.1",
            origin="https://evil.com",
        ))
        assert status == 403, (
            f"browser-origin /metrics must be 403 (DNS-rebinding "
            f"defense); got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_metrics_remote_allowed_when_env_set():
    """Operator opt-out: PRSM_ADMIN_REMOTE_ALLOWED=1 lets remote
    Prometheus scrapers through. Operator accepts risk + adds
    real auth at the proxy layer."""
    app = _build_app()
    os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = "1"
    try:
        status = asyncio.run(_invoke(app, "/metrics", "203.0.113.42"))
        assert status != 403, (
            f"PRSM_ADMIN_REMOTE_ALLOWED=1 must let /metrics through; "
            f"got {status}"
        )
    finally:
        del os.environ["PRSM_ADMIN_REMOTE_ALLOWED"]


def test_non_metrics_non_admin_paths_unaffected():
    """/compute/* and similar production-traffic endpoints must
    NOT be gated by the loopback check. F73 widens to `/metrics`
    specifically, not all non-admin endpoints."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        # /compute/inference from external — should NOT be 403
        # via the loopback gate (might be 4xx for other reasons
        # like body validation, but not 403 from middleware).
        # Use POST to avoid the body-size middleware which
        # checks Content-Length — empty body is fine.
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/compute/inference",
            "query_string": b"",
            "headers": [(b"content-length", b"0")],
            "client": ("203.0.113.42", 12345),
            "server": ("127.0.0.1", 8000),
            "scheme": "http",
            "root_path": "",
        }
        received = [{"type": "http.request", "body": b"",
                     "more_body": False}]
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
        # /compute/inference might 4xx for various reasons; just
        # confirm it's NOT 403 from our middleware.
        # (Could be 422 validation, 503 executor missing, etc.)
        assert starts[0]["status"] != 403, (
            "/compute/inference must NOT be loopback-gated; got "
            f"{starts[0]['status']} (sprint 745 widening was too "
            f"broad)"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old

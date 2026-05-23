"""Sprint 748 F75 — /status + /rings/status gated by loopback rules.

`/status` is the heaviest recon leak in the codebase. Pre-748 it
returned to any HTTP client:

- `ftns_balance` (operator's literal FTNS balance — financial)
- `node_id`, `p2p_address`, `api_address` (network topology)
- Peer counts + bootstrap_telemetry (network attack surface)
- Per-subsystem provider stats (compute / storage / content /
  ledger_sync / escrow / consensus / batch_settlement / agents /
  collaboration / bittorrent)
- `ftns_onchain` summary (on-chain totals)

`/rings/status` returns Ring 1-10 init + health — same recon
class as /health/detailed.

Sibling of F73 (/metrics) + F74 (/info, /health/detailed). Same
fix: add to the loopback-gated `_GATED_PATHS` tuple.

Operator-tooling impact: same 3 remediation paths sprint-740
runbook documents (same-host scraping / reverse-proxy + auth /
PRSM_ADMIN_REMOTE_ALLOWED=1 opt-out).
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
    # /status calls `await node.get_status()`. AsyncMock makes the
    # call awaitable so the handler can complete (we're testing the
    # middleware gate, not the handler payload).
    node.get_status = AsyncMock(return_value={
        "node_id": "test-node-id", "ftns_balance": 0.0,
    })
    return create_api_app(node, enable_security=False)


def test_status_from_external_rejected():
    """Most-sensitive endpoint: external GET /status → 403 (F75)."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(app, "/status", "203.0.113.42"))
        assert status == 403, (
            f"external /status must be 403 (leaks ftns_balance + "
            f"subsystem stats); got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_status_from_loopback_passes_gate():
    """Loopback /status passes the middleware gate (may 4xx for
    other reasons like missing mock attributes; we only test that
    F75's middleware doesn't 403 us)."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(app, "/status", "127.0.0.1"))
        assert status != 403, (
            f"loopback /status must not be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_rings_status_from_external_rejected():
    """/rings/status from non-loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/rings/status", "203.0.113.42",
        ))
        assert status == 403, (
            f"external /rings/status must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_rings_status_from_loopback_passes_gate():
    """Loopback /rings/status passes the middleware gate."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/rings/status", "127.0.0.1",
        ))
        assert status != 403, (
            f"loopback /rings/status must not be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_remote_allowed_lets_status_through():
    """PRSM_ADMIN_REMOTE_ALLOWED=1 bypasses F75 gate, consistent
    with F65-F74 behavior."""
    app = _build_app()
    os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = "1"
    try:
        status = asyncio.run(_invoke(app, "/status", "203.0.113.42"))
        assert status != 403, (
            f"PRSM_ADMIN_REMOTE_ALLOWED=1 must bypass F75 gate; "
            f"got {status}"
        )
    finally:
        del os.environ["PRSM_ADMIN_REMOTE_ALLOWED"]

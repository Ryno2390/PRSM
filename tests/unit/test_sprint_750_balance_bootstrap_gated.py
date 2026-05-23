"""Sprint 750 F77 — /balance + /bootstrap/status gated by loopback.

`/balance` is the worst recon leak found yet. Pre-750 it returned:

- `wallet_id` (operator's node_id)
- `balance` (operator's literal FTNS balance)
- Last 20 `recent_transactions`:
  - tx_id, type, from_wallet, to_wallet (counterparty IDs)
  - amount + description + timestamp

For a network attacker reading /balance, this is a complete
financial profile: who the operator pays, how much, when, with
what counterparties. Worse than F75's /status (just balance) —
this also gives transaction history.

`/bootstrap/status` leaks which bootstrap servers the daemon is
using — useful for an attacker planning to attack the bootstrap
fleet (DoS the bootstrap → daemon loses peer discovery).

Sibling of F75 (/status), F76 (/peers). Same gate.
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
    node.get_status = AsyncMock(return_value={})
    # /balance reads ledger.get_balance + get_transaction_history
    node.ledger.get_balance = AsyncMock(return_value=0.0)
    node.ledger.get_transaction_history = AsyncMock(return_value=[])
    # /bootstrap/status reads discovery.get_bootstrap_status which
    # returns a dict; the endpoint expects a serializable response.
    node.discovery.get_bootstrap_status = MagicMock(return_value={
        "registered": False,
    })
    return create_api_app(node, enable_security=False)


def test_balance_from_external_rejected():
    """The worst recon leak — external GET /balance → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(app, "/balance", "203.0.113.42"))
        assert status == 403, (
            f"external /balance must be 403 (leaks balance + "
            f"tx history); got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_balance_from_loopback_passes_gate():
    """Loopback /balance passes the gate (operator CLI tooling)."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(app, "/balance", "127.0.0.1"))
        assert status != 403, (
            f"loopback /balance must not be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_bootstrap_status_from_external_rejected():
    """/bootstrap/status from non-loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/bootstrap/status", "203.0.113.42",
        ))
        assert status == 403, (
            f"external /bootstrap/status must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_bootstrap_status_from_loopback_passes_gate():
    """Loopback /bootstrap/status passes."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/bootstrap/status", "127.0.0.1",
        ))
        assert status != 403, (
            f"loopback /bootstrap/status must not be 403; "
            f"got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_remote_allowed_lets_balance_through():
    """PRSM_ADMIN_REMOTE_ALLOWED=1 bypasses F77, consistent with
    the F65-F76 arc."""
    app = _build_app()
    os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = "1"
    try:
        status = asyncio.run(_invoke(app, "/balance", "203.0.113.42"))
        assert status != 403, (
            f"PRSM_ADMIN_REMOTE_ALLOWED=1 must bypass F77 gate; "
            f"got {status}"
        )
    finally:
        del os.environ["PRSM_ADMIN_REMOTE_ALLOWED"]


def test_balance_browser_origin_rejected():
    """F71 DNS-rebinding defense applies — browser /balance → 403
    even from loopback. Prevents a malicious page in the operator's
    browser from reading their balance via DNS rebinding."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/balance",
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
            "browser-origin /balance must be 403 (DNS-rebinding "
            f"defense); got {starts[0]['status']}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old

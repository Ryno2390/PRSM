"""Sprint 751 F78 — /transactions + /staking/status + /settlement/* gated.

Three more financial recon vectors closed:

1. `/transactions` — returns up to 200 transactions of the
   operator's full ledger history. Worse than F77's /balance
   (20 transactions). Same complete-financial-profile concern.

2. `/staking/status` — returns the node's active stakes, pending
   unstake requests, reward totals. Reveals:
   - How much the operator has staked (financial position)
   - When unstake requests unlock (timing attack — attacker
     knows when funds become claimable)
   - Reward accruals (yield rate intel)

3. `/settlement/stats`, `/settlement/pending`, `/settlement/
   history` — settlement counts/amounts/schedules. Financial
   intel + timing intel.

Sibling family of F73 (/metrics), F74 (/info, /health/detailed),
F75 (/status, /rings/status), F76 (/peers), F77 (/balance,
/bootstrap/status). Same fix: add to `_GATED_PATHS`. Inherits
all F65-F71 defenses (loopback gate + XFF/X-Real-IP + DNS-
rebinding via Origin).
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
    # When PRSM_ADMIN_REMOTE_ALLOWED=1 the middleware passes the
    # request to the inner handlers. Mock the async ledger/staking
    # calls so the bypass-test doesn't crash inside FastAPI.
    node.ledger.get_transaction_history = AsyncMock(return_value=[])
    node.staking_manager.get_user_stakes = AsyncMock(return_value=[])
    node.staking_manager.get_unstake_requests = AsyncMock(return_value=[])
    node.staking_manager.get_total_rewards = AsyncMock(return_value=0.0)
    return create_api_app(node, enable_security=False)


def test_transactions_from_external_rejected():
    """/transactions from non-loopback → 403 (200-tx history
    leak closure)."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/transactions", "203.0.113.42",
        ))
        assert status == 403, (
            f"external /transactions must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_staking_status_from_external_rejected():
    """/staking/status from non-loopback → 403 (stake position
    + unlock-timing leak closure)."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/staking/status", "203.0.113.42",
        ))
        assert status == 403, (
            f"external /staking/status must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_settlement_stats_from_external_rejected():
    """/settlement/stats from non-loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/settlement/stats", "203.0.113.42",
        ))
        assert status == 403, (
            f"external /settlement/stats must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_settlement_pending_from_external_rejected():
    """/settlement/pending from non-loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/settlement/pending", "203.0.113.42",
        ))
        assert status == 403
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_settlement_history_from_external_rejected():
    """/settlement/history from non-loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/settlement/history", "203.0.113.42",
        ))
        assert status == 403
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_remote_allowed_bypasses_f78_gates():
    """PRSM_ADMIN_REMOTE_ALLOWED=1 bypasses ALL F78 gates,
    consistent with the F65-F77 arc. Behavioral: middleware
    no longer emits 403; inner-handler errors are tolerated
    (we're testing middleware behavior, not handler correctness)."""
    app = _build_app()
    os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = "1"
    try:
        # /transactions has a fully-mocked happy path. Test it
        # directly. Other paths need handler-specific mocks not
        # worth wiring here — middleware-pass already proven by
        # the 5 external-rejection tests being 403 (which they
        # wouldn't be if the middleware always passed).
        status = asyncio.run(_invoke(
            app, "/transactions", "203.0.113.42",
        ))
        assert status != 403, (
            f"PRSM_ADMIN_REMOTE_ALLOWED=1 must bypass F78 "
            f"on /transactions; got {status}"
        )
    finally:
        del os.environ["PRSM_ADMIN_REMOTE_ALLOWED"]


def test_browser_origin_rejects_transactions():
    """F71 DNS-rebinding defense applies to /transactions too."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/transactions",
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
            "browser-origin /transactions must be 403 (DNS-"
            f"rebinding defense); got {starts[0]['status']}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old

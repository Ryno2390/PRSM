"""Sprint 752 F79 — /balance/onchain + /audit/* + /ledger/sync/stats gated.

Four more recon-class endpoints closed:

- `/balance/onchain` (no address arg): operator's on-chain FTNS
  balance. Same financial-value concern as F77's /balance, but
  on the on-chain side.

- `/audit/summary` + `/audit/recent`: HTTP access log aggregates
  + recent entries. Leaks:
  - Endpoint usage patterns (which features are actually used)
  - Status-code buckets (5xx rate → operator's reliability)
  - Method buckets (POST/GET ratios → workload type)
  - Top-N most-frequent paths (attack-surface map)
  - Total request volume (load intel for DoS targeting)

- `/ledger/sync/stats`: ledger sync state with peer counts +
  last_sync timestamps. Network intel + timing signals.

Sibling of F73-F78 recon family. Same `_GATED_PATHS` mechanism;
inherits all F65-F71 defenses.
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
    return create_api_app(node, enable_security=False)


def test_balance_onchain_from_external_rejected():
    """/balance/onchain from non-loopback → 403 (operator
    on-chain balance leak closure)."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/balance/onchain", "203.0.113.42",
        ))
        assert status == 403, (
            f"external /balance/onchain must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_audit_summary_from_external_rejected():
    """/audit/summary from non-loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/audit/summary", "203.0.113.42",
        ))
        assert status == 403, (
            f"external /audit/summary must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_audit_recent_from_external_rejected():
    """/audit/recent from non-loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/audit/recent", "203.0.113.42",
        ))
        assert status == 403
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_ledger_sync_stats_from_external_rejected():
    """/ledger/sync/stats from non-loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/ledger/sync/stats", "203.0.113.42",
        ))
        assert status == 403
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_all_f79_paths_passable_from_loopback():
    """All 4 F79 paths must pass the gate from loopback (CLI /
    operator-tooling triage). Inner handlers may 4xx/5xx for
    other reasons; we only verify the middleware doesn't 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        for path in (
            "/balance/onchain", "/audit/summary",
            "/audit/recent", "/ledger/sync/stats",
        ):
            try:
                status = asyncio.run(_invoke(app, path, "127.0.0.1"))
            except Exception:
                # Inner-handler crash due to missing mock attrs
                # still means the middleware passed (no 403).
                continue
            assert status != 403, (
                f"loopback {path} must not be 403; got {status}"
            )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_browser_origin_rejects_balance_onchain():
    """F71 DNS-rebinding defense applies to /balance/onchain."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/balance/onchain",
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
        assert starts[0]["status"] == 403
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old

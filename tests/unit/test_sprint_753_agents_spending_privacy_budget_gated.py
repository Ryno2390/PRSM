"""Sprint 753 F80 — /agents/spending + /privacy/budget gated.

Two more recon-class endpoints closed:

- `/agents/spending` — per-agent FTNS spend totals. Financial
  intel showing how much the operator pays to each agent
  service. Sibling of F77 (/balance) and F78 (/transactions).

- `/privacy/budget` — differential-privacy epsilon budget audit
  report. Reveals:
  - Total epsilon spent (privacy posture)
  - Recent budget consumption pattern (which queries ran when)
  - Remaining budget (attacker can time queries to exhaust it,
    forcing the operator to either refuse service or downgrade
    privacy tier)

Explicit non-inclusion: `/agents` (bare list) is service-
discovery surface in PRSM's marketplace model. Operators
publishing services WANT them discoverable. Same posture as
minimal `/health` — public by design.

Sibling of F73-F79 recon family. Same `_GATED_PATHS` mechanism.
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


def test_agents_spending_from_external_rejected():
    """/agents/spending from non-loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/agents/spending", "203.0.113.42",
        ))
        assert status == 403, (
            f"external /agents/spending must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_privacy_budget_from_external_rejected():
    """/privacy/budget from non-loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/privacy/budget", "203.0.113.42",
        ))
        assert status == 403, (
            f"external /privacy/budget must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_agents_bare_list_remains_public():
    """Pin: /agents (bare list) is INTENTIONALLY NOT gated.
    Service-discovery surface in PRSM's marketplace — operators
    publishing services want them discoverable. This pin
    defends against accidental over-gating in a future refactor."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        # External request from a non-loopback IP — should NOT
        # 403 from the middleware (might 4xx/5xx for other
        # reasons; mock doesn't fully wire agent_registry).
        status = asyncio.run(_invoke(app, "/agents", "203.0.113.42"))
        assert status != 403, (
            f"/agents must remain public for service discovery; "
            f"got {status} — sprint 753 over-gated"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_remote_allowed_bypasses_f80():
    """PRSM_ADMIN_REMOTE_ALLOWED=1 bypasses F80 gates."""
    app = _build_app()
    os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = "1"
    try:
        for path in ("/agents/spending", "/privacy/budget"):
            try:
                status = asyncio.run(_invoke(app, path, "203.0.113.42"))
            except Exception:
                # Inner-handler crash on missing mocks still
                # proves middleware passed
                continue
            assert status != 403, (
                f"env bypass failed on {path}; got {status}"
            )
    finally:
        del os.environ["PRSM_ADMIN_REMOTE_ALLOWED"]


def test_browser_origin_rejects_privacy_budget():
    """F71 DNS-rebinding defense applies to /privacy/budget."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/privacy/budget",
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

"""Sprint 747 F74 — /info + /health/detailed gated by loopback rules.

Pre-747, both endpoints were publicly readable. Reconnaissance-
grade data leaked:

- `/info`:
  - `operator_address` (on-chain EOA — financial-intel surface,
    lets attackers cross-reference operator's mainnet activity)
  - `node_id` (P2P identity — link to public anchor entry)
  - `wired_contract_addresses` (attack-surface map for on-chain
    interaction)
  - `api_version` (CVE-matching surface — if attacker knows a
    PRSM version with a known issue, they can target it)
  - `agent_forge_wired` (subsystem state — which optional
    features are loaded)

- `/health/detailed`:
  - Per-subsystem status across 14+ subsystems including
    `ftns_ledger.connected_address` + `wired_token` + canonical-
    match boolean — same financial-intel surface as /info
  - Detailed error strings from subsystems (could leak internal
    state on failures)

Sibling of F73 (/metrics leak) — same fix pattern.

The minimal `/health` endpoint (which load balancers probe)
is INTENTIONALLY NOT gated — LBs need it reachable from a
separate health-check host.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock


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
    # /health response model requires string node_id (real-type
    # FastAPI validation); MagicMock attribute access returns a
    # MagicMock by default, which fails serialization.
    node.identity.node_id = "test-node-id"
    return create_api_app(node, enable_security=False)


def test_info_from_external_rejected():
    """/info from non-loopback → 403 (F74 closure)."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(app, "/info", "203.0.113.42"))
        assert status == 403, (
            f"external /info must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_info_from_loopback_passes():
    """/info from loopback continues to work (CLI / local
    operator tools)."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(app, "/info", "127.0.0.1"))
        assert status != 403, (
            f"loopback /info must pass; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_health_detailed_from_external_rejected():
    """/health/detailed from non-loopback → 403."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/health/detailed", "203.0.113.42",
        ))
        assert status == 403, (
            f"external /health/detailed must be 403; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_health_detailed_from_loopback_passes():
    """/health/detailed from loopback continues to work for
    local operator-tooling triage."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(
            app, "/health/detailed", "127.0.0.1",
        ))
        assert status != 403, (
            f"loopback /health/detailed must pass; got {status}"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_minimal_health_endpoint_stays_public():
    """The minimal `/health` (no /detailed suffix) MUST stay
    public — load balancers and uptime monitors need to probe it
    from external hosts. Pre-747 it was public; post-747 it
    MUST remain so."""
    app = _build_app()
    old = os.environ.pop("PRSM_ADMIN_REMOTE_ALLOWED", None)
    try:
        status = asyncio.run(_invoke(app, "/health", "203.0.113.42"))
        assert status != 403, (
            f"minimal /health must stay public (load-balancer "
            f"probe); got {status} — F74 was too broad"
        )
    finally:
        if old is not None:
            os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = old


def test_remote_allowed_lets_info_through():
    """Operator opt-out via PRSM_ADMIN_REMOTE_ALLOWED=1 bypasses
    the F74 gate, consistent with F65-F73 behavior."""
    app = _build_app()
    os.environ["PRSM_ADMIN_REMOTE_ALLOWED"] = "1"
    try:
        status = asyncio.run(_invoke(app, "/info", "203.0.113.42"))
        assert status != 403, (
            f"PRSM_ADMIN_REMOTE_ALLOWED=1 must bypass F74 gate; "
            f"got {status}"
        )
    finally:
        del os.environ["PRSM_ADMIN_REMOTE_ALLOWED"]

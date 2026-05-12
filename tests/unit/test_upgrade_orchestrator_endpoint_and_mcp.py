"""Sprint 303 — UUPS upgrade orchestrator HTTP + MCP.

POST /admin/upgrade/propose
GET  /admin/upgrade
GET  /admin/upgrade/{id}
POST /admin/upgrade/{id}/update
POST /admin/upgrade/{id}/compose-upgrade
POST /admin/upgrade/{id}/compose-rollback

prsm_upgrade MCP tool — propose | list | lookup | update |
compose_upgrade | compose_rollback.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.upgrade_orchestrator import (
    UpgradeOrchestrator, UpgradeSeverity, UpgradeStatus,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_upgrade,
)
from prsm.node.api import create_api_app


def _client(orchestrator=None, chain_id=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._upgrade_orchestrator = orchestrator
    node._upgrade_chain_id = chain_id
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _proposed(o):
    return o.propose(
        target_proxy="0x" + "aa" * 20,
        new_implementation="0x" + "bb" * 20,
        previous_implementation="0x" + "cc" * 20,
        severity=UpgradeSeverity.PLANNED,
        rationale="x",
    )


def _executed(o):
    p = _proposed(o)
    o.update_status(p.proposal_id, UpgradeStatus.REVIEWED)
    o.update_status(
        p.proposal_id, UpgradeStatus.SAFE_UPLOADED,
    )
    o.update_status(
        p.proposal_id, UpgradeStatus.EXECUTED,
        safe_tx_hash="0xabc",
    )
    return p


# ── propose ─────────────────────────────────────────


def test_propose_503_unwired():
    """When orchestrator missing, propose returns 503 (not 422).
    Provide a complete body so validation passes — the
    503 must come from the dependency check."""
    resp = _client(None).post(
        "/admin/upgrade/propose",
        json={
            "target_proxy": "0x" + "aa" * 20,
            "new_implementation": "0x" + "bb" * 20,
            "previous_implementation": "0x" + "cc" * 20,
            "severity": "planned",
            "rationale": "x",
        },
    )
    assert resp.status_code == 503


def test_propose_happy_path():
    o = UpgradeOrchestrator()
    resp = _client(o).post(
        "/admin/upgrade/propose",
        json={
            "target_proxy": "0x" + "aa" * 20,
            "new_implementation": "0x" + "bb" * 20,
            "previous_implementation": "0x" + "cc" * 20,
            "severity": "planned",
            "rationale": "reentrancy fix",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "proposed"
    assert body["severity"] == "planned"
    assert o.count() == 1


def test_propose_422_invalid_severity():
    resp = _client(UpgradeOrchestrator()).post(
        "/admin/upgrade/propose",
        json={
            "target_proxy": "0x" + "aa" * 20,
            "new_implementation": "0x" + "bb" * 20,
            "previous_implementation": "0x" + "cc" * 20,
            "severity": "panic",
            "rationale": "x",
        },
    )
    assert resp.status_code == 422


def test_propose_422_invalid_address():
    resp = _client(UpgradeOrchestrator()).post(
        "/admin/upgrade/propose",
        json={
            "target_proxy": "not-addr",
            "new_implementation": "0x" + "bb" * 20,
            "previous_implementation": "0x" + "cc" * 20,
            "severity": "planned",
            "rationale": "x",
        },
    )
    assert resp.status_code == 422


# ── list + get ──────────────────────────────────────


def test_list_filter_by_status():
    o = UpgradeOrchestrator()
    a = _proposed(o)
    _proposed(o)
    o.update_status(a.proposal_id, UpgradeStatus.REVIEWED)
    body = _client(o).get(
        "/admin/upgrade?status=reviewed",
    ).json()
    assert body["count"] == 1


def test_get_404_unknown():
    assert _client(UpgradeOrchestrator()).get(
        "/admin/upgrade/no-such",
    ).status_code == 404


def test_get_happy_path():
    o = UpgradeOrchestrator()
    p = _proposed(o)
    body = _client(o).get(
        f"/admin/upgrade/{p.proposal_id}",
    ).json()
    assert body["proposal_id"] == p.proposal_id


# ── update ──────────────────────────────────────────


def test_update_happy_path():
    o = UpgradeOrchestrator()
    p = _proposed(o)
    resp = _client(o).post(
        f"/admin/upgrade/{p.proposal_id}/update",
        json={"new_status": "reviewed"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "reviewed"


def test_update_404_unknown():
    resp = _client(UpgradeOrchestrator()).post(
        "/admin/upgrade/no-such/update",
        json={"new_status": "reviewed"},
    )
    assert resp.status_code == 404


def test_update_422_invalid_status():
    o = UpgradeOrchestrator()
    p = _proposed(o)
    resp = _client(o).post(
        f"/admin/upgrade/{p.proposal_id}/update",
        json={"new_status": "wat"},
    )
    assert resp.status_code == 422


def test_update_422_back_to_proposed():
    o = UpgradeOrchestrator()
    p = _proposed(o)
    o.update_status(p.proposal_id, UpgradeStatus.REVIEWED)
    resp = _client(o).post(
        f"/admin/upgrade/{p.proposal_id}/update",
        json={"new_status": "proposed"},
    )
    assert resp.status_code == 422


# ── compose-upgrade ─────────────────────────────────


def test_compose_upgrade_happy_path():
    o = UpgradeOrchestrator()
    p = _proposed(o)
    o.update_status(p.proposal_id, UpgradeStatus.REVIEWED)
    resp = _client(o, chain_id=8453).post(
        f"/admin/upgrade/{p.proposal_id}/compose-upgrade",
        json={},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "upgrade"
    assert body["to"] == "0x" + "aa" * 20
    assert body["chain_id"] == 8453


def test_compose_upgrade_422_not_reviewed():
    o = UpgradeOrchestrator()
    p = _proposed(o)  # still PROPOSED
    resp = _client(o).post(
        f"/admin/upgrade/{p.proposal_id}/compose-upgrade",
        json={},
    )
    assert resp.status_code == 422


def test_compose_upgrade_404_unknown():
    resp = _client(UpgradeOrchestrator()).post(
        "/admin/upgrade/nope/compose-upgrade",
        json={},
    )
    assert resp.status_code == 404


# ── compose-rollback ────────────────────────────────


def test_compose_rollback_happy_path():
    o = UpgradeOrchestrator()
    p = _executed(o)
    resp = _client(o, chain_id=8453).post(
        f"/admin/upgrade/{p.proposal_id}/compose-rollback",
        json={},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "rollback"
    assert "cc" * 20 in body["data"]


def test_compose_rollback_422_not_executed():
    o = UpgradeOrchestrator()
    p = _proposed(o)
    resp = _client(o).post(
        f"/admin/upgrade/{p.proposal_id}/compose-rollback",
        json={},
    )
    assert resp.status_code == 422


# ── MCP ─────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_upgrade" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_upgrade({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_unknown_action():
    r = await handle_prsm_upgrade({"action": "explode"})
    assert "must be" in r.lower()


@pytest.mark.asyncio
async def test_mcp_propose_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "proposal_id": "up-1",
            "opened_ts": 100.0,
            "target_proxy": "0x" + "aa" * 20,
            "new_implementation": "0x" + "bb" * 20,
            "previous_implementation": "0x" + "cc" * 20,
            "severity": "emergency",
            "rationale": "x",
            "status": "proposed",
            "init_calldata_hex": "0x",
            "reviewer_assignments": [],
            "safe_tx_hash": None,
        }),
    ) as mock_call:
        r = await handle_prsm_upgrade({
            "action": "propose",
            "target_proxy": "0x" + "aa" * 20,
            "new_implementation": "0x" + "bb" * 20,
            "previous_implementation": "0x" + "cc" * 20,
            "severity": "emergency",
            "rationale": "x",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/upgrade/propose"
    assert "up-1" in r


@pytest.mark.asyncio
async def test_mcp_list():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "records": [{
                "proposal_id": "up-abcdef12",
                "opened_ts": 100.0,
                "target_proxy": "0x" + "aa" * 20,
                "new_implementation": "0x" + "bb" * 20,
                "previous_implementation": "0x" + "cc" * 20,
                "severity": "planned",
                "rationale": "x",
                "status": "reviewed",
                "init_calldata_hex": "0x",
                "reviewer_assignments": [],
                "safe_tx_hash": None,
            }],
            "count": 1,
        }),
    ):
        r = await handle_prsm_upgrade({"action": "list"})
    assert "up-abcde" in r
    assert "reviewed" in r.lower()


@pytest.mark.asyncio
async def test_mcp_compose_upgrade_renders_safe_block():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "action": "upgrade",
            "proposal_id": "up-1",
            "to": "0x" + "aa" * 20,
            "data": "0x4f1ef286" + "0" * 200,
            "value": "0",
            "target_proxy": "0x" + "aa" * 20,
            "new_implementation": "0x" + "bb" * 20,
            "previous_implementation": "0x" + "cc" * 20,
            "severity": "emergency",
            "rationale": "reentrancy fix",
            "chain_id": 8453,
            "warning": "DESTRUCTIVE: replaces impl...",
            "explorer_url": (
                "https://basescan.org/address/0x"
                + "aa" * 20
            ),
            "instructions": "1) Open Safe...",
        }),
    ) as mock_call:
        r = await handle_prsm_upgrade({
            "action": "compose_upgrade",
            "proposal_id": "up-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/upgrade/up-1/compose-upgrade"
    assert "DESTRUCTIVE" in r.upper() or (
        "destructive" in r.lower()
    )
    assert "0x" + "aa" * 20 in r
    assert "reentrancy fix" in r


@pytest.mark.asyncio
async def test_mcp_compose_rollback():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "action": "rollback",
            "proposal_id": "up-1",
            "to": "0x" + "aa" * 20,
            "data": "0x4f1ef286" + "0" * 200,
            "value": "0",
            "target_proxy": "0x" + "aa" * 20,
            "rollback_target_implementation": (
                "0x" + "cc" * 20
            ),
            "originally_upgraded_to": "0x" + "bb" * 20,
            "severity": "emergency",
            "chain_id": 8453,
            "warning": "DESTRUCTIVE ROLLBACK: ...",
            "explorer_url": "https://basescan.org/...",
            "instructions": "1) Open Safe...",
        }),
    ) as mock_call:
        r = await handle_prsm_upgrade({
            "action": "compose_rollback",
            "proposal_id": "up-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/upgrade/up-1/compose-rollback"
    )
    assert "ROLLBACK" in r.upper()


@pytest.mark.asyncio
async def test_mcp_propose_requires_fields():
    r = await handle_prsm_upgrade({"action": "propose"})
    assert "required" in r.lower() or "missing" in r.lower()

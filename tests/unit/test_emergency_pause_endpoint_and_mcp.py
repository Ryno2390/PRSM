"""Sprint 298 — emergency pause HTTP endpoint + MCP wrapper.

GET  /admin/emergency-pause/status   — bulk status query
POST /admin/emergency-pause/compose  — composer-only tx
                                       payload for Safe upload

The composer is composer-only by design. Even an
authenticated operator submitting compose requests gains no
pause authority — the output is calldata that requires
Foundation Safe 2-of-3 multi-sig signing.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.emergency_pause_client import (
    EmergencyPauseClient, PAUSE_SELECTOR, UNPAUSE_SELECTOR,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_emergency_pause,
)
from prsm.node.api import create_api_app


class _FakeBackend:
    def __init__(self, paused_map=None):
        self.paused_map = paused_map or {}

    def call(self, addr, data):
        is_paused = self.paused_map.get(addr, False)
        return b"\x00" * 31 + (b"\x01" if is_paused else b"\x00")


def _client(pause_client=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._emergency_pause_client = pause_client
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _commissioned_pause_client(paused_map=None):
    return EmergencyPauseClient(
        contract_addresses={
            "ftns_token": "0xFFFF" + "0" * 36,
            "royalty_distributor": "0xAAAA" + "0" * 36,
            "escrow_pool": "0xBBBB" + "0" * 36,
            "stake_bond": "0xCCCC" + "0" * 36,
            "compensation_distributor": "0xDDDD" + "0" * 36,
            "storage_slashing": "0xEEEE" + "0" * 36,
            "settlement_registry": "0x1111" + "0" * 36,
            "signature_verifier": "0x2222" + "0" * 36,
            "emission_controller": "0x3333" + "0" * 36,
            "key_distribution": "0x4444" + "0" * 36,
        },
        rpc_url="https://rpc.example",
        chain_id=8453,
        backend=_FakeBackend(paused_map or {}),
    )


# ── HTTP: status ─────────────────────────────────────────


def test_status_503_when_unwired():
    resp = _client(None).get(
        "/admin/emergency-pause/status",
    )
    assert resp.status_code == 503


def test_status_returns_per_contract_dict():
    pause_client = _commissioned_pause_client()
    resp = _client(pause_client).get(
        "/admin/emergency-pause/status",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "contracts" in body
    assert "ftns_token" in body["contracts"]
    assert body["contracts"]["ftns_token"]["paused"] is False


def test_status_surfaces_paused_contract():
    pause_client = _commissioned_pause_client(
        paused_map={"0xFFFF" + "0" * 36: True},
    )
    resp = _client(pause_client).get(
        "/admin/emergency-pause/status",
    )
    body = resp.json()
    assert body["contracts"]["ftns_token"]["paused"] is True


def test_status_includes_chain_id():
    pause_client = _commissioned_pause_client()
    resp = _client(pause_client).get(
        "/admin/emergency-pause/status",
    )
    body = resp.json()
    assert body["chain_id"] == 8453


# ── HTTP: compose ────────────────────────────────────────


def test_compose_503_when_unwired():
    resp = _client(None).post(
        "/admin/emergency-pause/compose",
        json={"action": "pause", "contract_name": "ftns_token"},
    )
    assert resp.status_code == 503


def test_compose_pause_happy_path():
    pause_client = _commissioned_pause_client()
    resp = _client(pause_client).post(
        "/admin/emergency-pause/compose",
        json={"action": "pause", "contract_name": "ftns_token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "pause"
    assert body["data"] == PAUSE_SELECTOR
    assert body["to"] == "0xFFFF" + "0" * 36


def test_compose_unpause_happy_path():
    pause_client = _commissioned_pause_client()
    resp = _client(pause_client).post(
        "/admin/emergency-pause/compose",
        json={
            "action": "unpause",
            "contract_name": "ftns_token",
        },
    )
    body = resp.json()
    assert body["data"] == UNPAUSE_SELECTOR


def test_compose_422_missing_action():
    pause_client = _commissioned_pause_client()
    resp = _client(pause_client).post(
        "/admin/emergency-pause/compose",
        json={"contract_name": "ftns_token"},
    )
    assert resp.status_code == 422


def test_compose_422_missing_contract_name():
    pause_client = _commissioned_pause_client()
    resp = _client(pause_client).post(
        "/admin/emergency-pause/compose",
        json={"action": "pause"},
    )
    assert resp.status_code == 422


def test_compose_422_invalid_action():
    pause_client = _commissioned_pause_client()
    resp = _client(pause_client).post(
        "/admin/emergency-pause/compose",
        json={"action": "destroy", "contract_name": "ftns_token"},
    )
    assert resp.status_code == 422


def test_compose_422_unknown_contract():
    pause_client = _commissioned_pause_client()
    resp = _client(pause_client).post(
        "/admin/emergency-pause/compose",
        json={
            "action": "pause",
            "contract_name": "not_a_contract",
        },
    )
    assert resp.status_code == 422


def test_compose_response_warning_present():
    pause_client = _commissioned_pause_client()
    resp = _client(pause_client).post(
        "/admin/emergency-pause/compose",
        json={"action": "pause", "contract_name": "ftns_token"},
    )
    body = resp.json()
    # Operator-facing warning surfaces consequences
    assert (
        "warning" in body and len(body["warning"]) > 0
    )


# ── MCP tool ─────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_emergency_pause" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_emergency_pause({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_unknown_action():
    r = await handle_prsm_emergency_pause(
        {"action": "explode"},
    )
    assert "must be" in r.lower()


@pytest.mark.asyncio
async def test_mcp_status_renders_table():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "chain_id": 8453,
            "contracts": {
                "ftns_token": {
                    "name": "ftns_token",
                    "address": "0xFFFF" + "0" * 36,
                    "paused": False,
                    "commissioned": True,
                    "error": None,
                },
                "royalty_distributor": {
                    "name": "royalty_distributor",
                    "address": "0xAAAA" + "0" * 36,
                    "paused": True,
                    "commissioned": True,
                    "error": None,
                },
            },
        }),
    ) as mock_call:
        r = await handle_prsm_emergency_pause(
            {"action": "status"},
        )
    args = mock_call.await_args[0]
    assert args[1] == "/admin/emergency-pause/status"
    # Both contracts surface with paused-state markers
    assert "ftns_token" in r
    assert "royalty_distributor" in r
    # Paused contract gets ⚠ or PAUSED marker
    assert "PAUSED" in r or "⚠" in r


@pytest.mark.asyncio
async def test_mcp_compose_pause_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "action": "pause",
            "to": "0xFFFF" + "0" * 36,
            "data": PAUSE_SELECTOR,
            "value": "0",
            "contract_name": "ftns_token",
            "description": "PAUSE the ftns_token contract.",
            "warning": (
                "DESTRUCTIVE: pausing this contract halts "
                "user-facing operations."
            ),
            "explorer_url": (
                "https://basescan.org/address/0xFFFF"
                + "0" * 36
            ),
            "chain_id": 8453,
            "instructions": (
                "1) Open the Foundation Safe UI; ..."
            ),
        }),
    ) as mock_call:
        r = await handle_prsm_emergency_pause({
            "action": "compose_pause",
            "contract_name": "ftns_token",
        })
    body = mock_call.await_args[0][2]
    assert body["action"] == "pause"
    assert body["contract_name"] == "ftns_token"
    # Renders the tx artifact + safety scaffolding
    assert "0xFFFF" in r
    assert PAUSE_SELECTOR in r
    # Operator-safety surfaces
    assert (
        "WARNING" in r.upper() or "destructive" in r.lower()
    )
    assert "safe" in r.lower()  # mentions Foundation Safe


@pytest.mark.asyncio
async def test_mcp_compose_unpause_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "action": "unpause",
            "to": "0xFFFF" + "0" * 36,
            "data": UNPAUSE_SELECTOR,
            "value": "0",
            "contract_name": "ftns_token",
            "description": "UNPAUSE the ftns_token.",
            "warning": "DESTRUCTIVE...",
            "explorer_url": "https://basescan.org/...",
            "chain_id": 8453,
            "instructions": "1)...",
        }),
    ) as mock_call:
        r = await handle_prsm_emergency_pause({
            "action": "compose_unpause",
            "contract_name": "ftns_token",
        })
    body = mock_call.await_args[0][2]
    assert body["action"] == "unpause"
    assert UNPAUSE_SELECTOR in r


@pytest.mark.asyncio
async def test_mcp_compose_requires_contract_name():
    r = await handle_prsm_emergency_pause({
        "action": "compose_pause",
    })
    assert "contract_name" in r


@pytest.mark.asyncio
async def test_mcp_503_message_when_unwired():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "detail": "Emergency pause client not initialized.",
        }),
    ):
        r = await handle_prsm_emergency_pause(
            {"action": "status"},
        )
    assert (
        "not wired" in r.lower()
        or "not initialized" in r.lower()
    )

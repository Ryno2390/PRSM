"""Sprint 306 — $CORP capability HTTP + MCP surface.

POST /admin/corp/issuer        — register an issuer
GET  /admin/corp/issuer        — list registered issuers
POST /admin/corp/capability/redeem — verify + redeem +
                                     return remaining quota
GET  /admin/corp/capability/{id}/ledger — audit trail
GET  /admin/corp/capability/{id}/consumed — running total

prsm_corp_capability MCP tool — register_issuer | list_issuers
| redeem | get_ledger | get_consumed | keypair_gen.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.enterprise.corp_capability import (
    CorpCapabilityStore, CorpIssuer,
    generate_issuer_keypair, generate_subject_keypair,
    sign_capability, sign_redemption,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_corp_capability,
)
from prsm.node.api import create_api_app


def _client(store=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._corp_capability_store = store
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _signed(scope=("compute.inference",), quota_units=100):
    ipriv, ipub = generate_issuer_keypair()
    spriv, spub = generate_subject_keypair()
    cap = sign_capability(
        issuer_id="acme",
        issuer_privkey_b64=ipriv,
        subject_id="alice@acme",
        subject_pubkey_b64=spub,
        scope=list(scope),
        quota_units=quota_units,
        issued_at=time.time(),
        expires_at=time.time() + 3600,
    )
    issuer = CorpIssuer("acme", ipub)
    return cap, issuer, spriv


# ── /admin/corp/issuer ──────────────────────────────


def test_register_issuer_503_unwired():
    _, pub = generate_issuer_keypair()
    resp = _client(store=None).post(
        "/admin/corp/issuer",
        json={"issuer_id": "acme", "signing_pubkey_b64": pub},
    )
    assert resp.status_code == 503


def test_register_issuer_happy_path():
    store = CorpCapabilityStore()
    _, pub = generate_issuer_keypair()
    resp = _client(store=store).post(
        "/admin/corp/issuer",
        json={"issuer_id": "acme", "signing_pubkey_b64": pub},
    )
    assert resp.status_code == 200
    assert store.get_issuer("acme") is not None


def test_register_issuer_422_bad_pubkey():
    store = CorpCapabilityStore()
    resp = _client(store=store).post(
        "/admin/corp/issuer",
        json={
            "issuer_id": "acme",
            "signing_pubkey_b64": "not-base64!",
        },
    )
    assert resp.status_code == 422


def test_list_issuers_returns_registered():
    store = CorpCapabilityStore()
    _, pub = generate_issuer_keypair()
    store.register_issuer(CorpIssuer("acme", pub))
    body = _client(store=store).get(
        "/admin/corp/issuer",
    ).json()
    assert len(body["issuers"]) == 1
    assert body["issuers"][0]["issuer_id"] == "acme"


# ── /admin/corp/capability/redeem ────────────────────


def test_redeem_503_unwired():
    resp = _client(store=None).post(
        "/admin/corp/capability/redeem",
        json={"capability": {}, "request": {}},
    )
    assert resp.status_code == 503


def test_redeem_happy_path():
    store = CorpCapabilityStore()
    cap, issuer, spriv = _signed(quota_units=100)
    store.register_issuer(issuer)
    req = sign_redemption(
        subject_privkey_b64=spriv,
        capability_id=cap.capability_id,
        action="compute.inference",
        units_requested=10, nonce="n-1",
        timestamp=time.time(),
    )
    resp = _client(store=store).post(
        "/admin/corp/capability/redeem",
        json={
            "capability": cap.to_dict(),
            "request": req.to_dict(),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pass"
    assert body["remaining_quota"] == 90


def test_redeem_unknown_issuer_passes_through_to_fail():
    """Issuer not registered — endpoint returns 200 with
    status=fail so the requester sees the diagnostic."""
    store = CorpCapabilityStore()
    cap, _, spriv = _signed(quota_units=100)
    # Note: issuer NOT registered
    req = sign_redemption(
        subject_privkey_b64=spriv,
        capability_id=cap.capability_id,
        action="compute.inference",
        units_requested=10, nonce="n-1",
        timestamp=time.time(),
    )
    resp = _client(store=store).post(
        "/admin/corp/capability/redeem",
        json={
            "capability": cap.to_dict(),
            "request": req.to_dict(),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "fail"
    assert "issuer" in body["diagnostic"].lower()


def test_redeem_422_malformed_capability():
    store = CorpCapabilityStore()
    resp = _client(store=store).post(
        "/admin/corp/capability/redeem",
        json={
            "capability": {"version": "wat"},
            "request": {},
        },
    )
    assert resp.status_code == 422


# ── ledger + consumed ───────────────────────────────


def test_ledger_returns_redemption_records():
    store = CorpCapabilityStore()
    cap, issuer, spriv = _signed(quota_units=100)
    store.register_issuer(issuer)
    req = sign_redemption(
        subject_privkey_b64=spriv,
        capability_id=cap.capability_id,
        action="compute.inference",
        units_requested=15, nonce="audit-1",
        timestamp=time.time(),
    )
    store.redeem(cap, req)
    body = _client(store=store).get(
        f"/admin/corp/capability/{cap.capability_id}/ledger",
    ).json()
    assert len(body["entries"]) == 1
    assert body["entries"][0]["units_requested"] == 15


def test_consumed_returns_running_total():
    store = CorpCapabilityStore()
    cap, issuer, spriv = _signed(quota_units=100)
    store.register_issuer(issuer)
    for i in range(3):
        req = sign_redemption(
            subject_privkey_b64=spriv,
            capability_id=cap.capability_id,
            action="compute.inference",
            units_requested=5, nonce=f"n-{i}",
            timestamp=time.time(),
        )
        store.redeem(cap, req)
    body = _client(store=store).get(
        f"/admin/corp/capability/{cap.capability_id}"
        f"/consumed",
    ).json()
    assert body["consumed"] == 15


# ── MCP ─────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_corp_capability" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_corp_capability({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_unknown_action():
    r = await handle_prsm_corp_capability({"action": "boom"})
    assert "must be" in r.lower()


@pytest.mark.asyncio
async def test_mcp_keypair_gen_offline():
    """keypair_gen MUST be fully offline — the issuer / subject
    keys never need to touch the network."""
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(),
    ) as mock_call:
        r = await handle_prsm_corp_capability({
            "action": "keypair_gen",
            "kind": "issuer",
        })
    assert mock_call.await_count == 0
    assert "privkey" in r.lower()
    assert "pubkey" in r.lower()


@pytest.mark.asyncio
async def test_mcp_keypair_gen_subject():
    r = await handle_prsm_corp_capability({
        "action": "keypair_gen",
        "kind": "subject",
    })
    assert "privkey" in r.lower()


@pytest.mark.asyncio
async def test_mcp_register_issuer():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "issuer_id": "acme",
            "signing_pubkey_b64": "AAAA",
        }),
    ) as mock_call:
        r = await handle_prsm_corp_capability({
            "action": "register_issuer",
            "issuer_id": "acme",
            "signing_pubkey_b64": "AAAA",
        })
    args = mock_call.await_args[0]
    assert args[0] == "POST"
    assert args[1] == "/admin/corp/issuer"
    assert "acme" in r


@pytest.mark.asyncio
async def test_mcp_list_issuers():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "issuers": [{
                "issuer_id": "acme",
                "signing_pubkey_b64": "AAAA",
            }],
        }),
    ):
        r = await handle_prsm_corp_capability({
            "action": "list_issuers",
        })
    assert "acme" in r


@pytest.mark.asyncio
async def test_mcp_redeem_renders_result():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "status": "pass",
            "capability_id": "cap-1",
            "units_consumed_this_request": 10,
            "remaining_quota": 90,
            "diagnostic": "ok",
        }),
    ) as mock_call:
        r = await handle_prsm_corp_capability({
            "action": "redeem",
            "capability": {"capability_id": "cap-1"},
            "request": {"nonce": "n-1"},
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/corp/capability/redeem"
    assert "pass" in r.lower()
    assert "90" in r


@pytest.mark.asyncio
async def test_mcp_get_ledger():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "entries": [{
                "timestamp": 100.0,
                "action": "compute.inference",
                "units_requested": 5,
                "nonce": "n-1",
                "subject_id": "alice",
            }],
        }),
    ) as mock_call:
        r = await handle_prsm_corp_capability({
            "action": "get_ledger",
            "capability_id": "cap-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/corp/capability/cap-1/ledger"
    )
    assert "compute.inference" in r


@pytest.mark.asyncio
async def test_mcp_get_consumed():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={"consumed": 42}),
    ) as mock_call:
        r = await handle_prsm_corp_capability({
            "action": "get_consumed",
            "capability_id": "cap-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/corp/capability/cap-1/consumed"
    )
    assert "42" in r


@pytest.mark.asyncio
async def test_mcp_redeem_requires_capability_and_request():
    r = await handle_prsm_corp_capability({
        "action": "redeem",
    })
    assert "capability" in r.lower()

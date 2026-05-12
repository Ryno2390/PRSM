"""Sprint 305 — TEE policy HTTP + MCP surface.

POST /admin/tee-policy/evaluate    — run a policy against
                                     an attestation blob
                                     (passed as base64) or
                                     an explicit result dict
GET  /admin/tee-policy/node-status — current node's own
                                     attestation tier
                                     (snapshot)

prsm_tee_policy MCP tool — evaluate | node_status |
                            list_tiers.
"""
from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.compute.inference.attestation_backends import (
    SOFTWARE_TEE_ATTESTATION_PREFIX,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_tee_policy,
)
from prsm.node.api import create_api_app


def _client(node_attestation=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._tee_node_attestation_blob = node_attestation
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def _software_blob_b64() -> str:
    return base64.b64encode(
        SOFTWARE_TEE_ATTESTATION_PREFIX + b"\x00" * 32,
    ).decode()


# ── /admin/tee-policy/evaluate ──────────────────────


def test_evaluate_software_blob_passes_software_policy():
    resp = _client().post(
        "/admin/tee-policy/evaluate",
        json={
            "attestation_b64": _software_blob_b64(),
            "policy": {
                "min_attestation_tier": "software",
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pass"
    assert body["effective_tier"] == "software"


def test_evaluate_software_blob_fails_hardware_policy():
    resp = _client().post(
        "/admin/tee-policy/evaluate",
        json={
            "attestation_b64": _software_blob_b64(),
            "policy": {
                "min_attestation_tier": "hardware_unverified",
            },
        },
    )
    body = resp.json()
    assert body["status"] == "fail"


def test_evaluate_missing_blob_passes_none_policy():
    resp = _client().post(
        "/admin/tee-policy/evaluate",
        json={
            "attestation_b64": None,
            "policy": {"min_attestation_tier": "none"},
        },
    )
    body = resp.json()
    assert body["status"] == "pass"


def test_evaluate_unknown_tier_422():
    resp = _client().post(
        "/admin/tee-policy/evaluate",
        json={
            "attestation_b64": _software_blob_b64(),
            "policy": {
                "min_attestation_tier": "extra-strong",
            },
        },
    )
    assert resp.status_code == 422


def test_evaluate_malformed_base64_422():
    resp = _client().post(
        "/admin/tee-policy/evaluate",
        json={
            "attestation_b64": "not-base64!",
            "policy": {
                "min_attestation_tier": "software",
            },
        },
    )
    assert resp.status_code == 422


def test_evaluate_vendor_allowlist_passes():
    resp = _client().post(
        "/admin/tee-policy/evaluate",
        json={
            "attestation_b64": _software_blob_b64(),
            "policy": {
                "min_attestation_tier": "software",
                "allowed_vendors": ["software-fallback"],
            },
        },
    )
    assert resp.json()["status"] == "pass"


def test_evaluate_vendor_allowlist_fails():
    resp = _client().post(
        "/admin/tee-policy/evaluate",
        json={
            "attestation_b64": _software_blob_b64(),
            "policy": {
                "min_attestation_tier": "software",
                "allowed_vendors": ["intel-sgx"],
            },
        },
    )
    assert resp.json()["status"] == "fail"


# ── /admin/tee-policy/node-status ───────────────────


def test_node_status_unattested_when_no_blob():
    body = _client(node_attestation=None).get(
        "/admin/tee-policy/node-status",
    ).json()
    assert body["effective_tier"] == "none"
    assert body["vendor"] in ("", "unknown")


def test_node_status_reports_software_tier():
    blob = (
        SOFTWARE_TEE_ATTESTATION_PREFIX + b"\x00" * 32
    )
    body = _client(node_attestation=blob).get(
        "/admin/tee-policy/node-status",
    ).json()
    assert body["effective_tier"] == "software"
    assert body["vendor"] == "software-fallback"


# ── MCP ─────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_tee_policy" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_tee_policy({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_list_tiers_runs_offline():
    """list_tiers is a static lookup — must not require
    the node to be up."""
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(),
    ) as mock_call:
        r = await handle_prsm_tee_policy({
            "action": "list_tiers",
        })
    assert mock_call.await_count == 0
    assert "none" in r.lower()
    assert "software" in r.lower()
    assert "hardware_unverified" in r.lower()
    assert "hardware_verified" in r.lower()


@pytest.mark.asyncio
async def test_mcp_evaluate_calls_node():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "status": "pass",
            "effective_tier": "software",
            "min_required_tier": "software",
            "vendor": "software-fallback",
            "diagnostic": "ok",
            "error": None,
        }),
    ) as mock_call:
        r = await handle_prsm_tee_policy({
            "action": "evaluate",
            "attestation_b64": _software_blob_b64(),
            "min_attestation_tier": "software",
        })
    args = mock_call.await_args[0]
    assert args[0] == "POST"
    assert args[1] == "/admin/tee-policy/evaluate"
    assert "pass" in r.lower()


@pytest.mark.asyncio
async def test_mcp_node_status_calls_node():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "effective_tier": "software",
            "vendor": "software-fallback",
            "diagnostic": "ok",
        }),
    ) as mock_call:
        r = await handle_prsm_tee_policy({
            "action": "node_status",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/tee-policy/node-status"
    assert "software" in r.lower()


@pytest.mark.asyncio
async def test_mcp_evaluate_requires_tier():
    r = await handle_prsm_tee_policy({
        "action": "evaluate",
        "attestation_b64": _software_blob_b64(),
    })
    assert (
        "tier" in r.lower()
        or "min_attestation_tier" in r.lower()
    )

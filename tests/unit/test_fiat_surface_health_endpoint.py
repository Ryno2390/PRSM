"""Sprint 286 — fiat-surface health endpoint + MCP wrapper.

GET /admin/fiat-surface/health returns findings keyed by
severity. MCP wrapper `prsm_fiat_surface_health` renders them
for operator inspection from inside an LLM session.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_fiat_surface_health,
)
from prsm.node.api import create_api_app


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── HTTP endpoint ────────────────────────────────────────


def test_health_endpoint_clean_env(monkeypatch):
    # Defensive: clear all related env vars
    for v in [
        "KYC_VENDOR", "KYC_VENDOR_API_KEY",
        "PERSONA_WEBHOOK_SECRET", "ONFIDO_WEBHOOK_TOKEN",
        "PRSM_FIAT_COMPLIANCE_LOG_DIR",
        "PRSM_OPERATOR_JURISDICTION",
        "PRSM_KYC_WEBHOOK_VERIFY_DISABLED",
        "PRSM_FIAT_HEALTH_CHECK_BYPASS",
        "COINBASE_CDP_API_KEY_NAME",
        "COINBASE_CDP_API_KEY_PRIVATE",
        "COINBASE_CDP_PAYMASTER_ENDPOINT",
        "COINBASE_CDP_PAYMASTER_API_KEY",
        "AERODROME_USDC_FTNS_POOL_ADDRESS",
        "BASE_RPC_URL",
    ]:
        monkeypatch.delenv(v, raising=False)
    resp = _client().get("/admin/fiat-surface/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["error_count"] == 0
    assert body["warn_count"] == 0
    assert body["overall"] == "OK"
    assert body["findings"] == []


def test_health_endpoint_with_errors(monkeypatch):
    monkeypatch.setenv("KYC_VENDOR", "persona")
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.delenv("PERSONA_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv(
        "PRSM_FIAT_HEALTH_CHECK_BYPASS", raising=False,
    )
    resp = _client().get("/admin/fiat-surface/health")
    body = resp.json()
    assert body["error_count"] >= 1
    assert body["overall"] == "ERROR"
    causes = [f["cause"] for f in body["findings"]]
    assert any(
        "persona_webhook_secret" in c.lower() for c in causes
    )


def test_health_endpoint_with_warns_only(monkeypatch):
    monkeypatch.setenv("KYC_VENDOR", "persona")
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.setenv("PERSONA_WEBHOOK_SECRET", "wh_secret")
    monkeypatch.delenv(
        "PRSM_FIAT_COMPLIANCE_LOG_DIR", raising=False,
    )
    monkeypatch.delenv(
        "PRSM_OPERATOR_JURISDICTION", raising=False,
    )
    resp = _client().get("/admin/fiat-surface/health")
    body = resp.json()
    assert body["error_count"] == 0
    assert body["warn_count"] >= 1
    assert body["overall"] == "WARN"


def test_health_endpoint_bypass_downgrades(monkeypatch):
    monkeypatch.setenv("KYC_VENDOR", "persona")
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.setenv("PRSM_FIAT_HEALTH_CHECK_BYPASS", "1")
    monkeypatch.delenv("PERSONA_WEBHOOK_SECRET", raising=False)
    resp = _client().get("/admin/fiat-surface/health")
    body = resp.json()
    assert body["error_count"] == 0
    assert body["overall"] in ("OK", "WARN", "INFO")


# ── MCP tool ─────────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_fiat_surface_health" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_renders_clean():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "overall": "OK",
            "error_count": 0,
            "warn_count": 0,
            "info_count": 0,
            "findings": [],
        }),
    ) as mock_call:
        r = await handle_prsm_fiat_surface_health({})
    args = mock_call.await_args[0]
    assert args[0] == "GET"
    assert args[1] == "/admin/fiat-surface/health"
    assert "OK" in r


@pytest.mark.asyncio
async def test_mcp_renders_error_finding():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "overall": "ERROR",
            "error_count": 1,
            "warn_count": 0,
            "info_count": 0,
            "findings": [{
                "severity": "ERROR",
                "cause": (
                    "kyc_commissioned_"
                    "persona_webhook_secret_missing"
                ),
                "remediation": (
                    "Set PERSONA_WEBHOOK_SECRET to your "
                    "Persona webhook secret."
                ),
            }],
        }),
    ):
        r = await handle_prsm_fiat_surface_health({})
    assert "ERROR" in r
    assert "PERSONA_WEBHOOK_SECRET" in r
    assert "persona_webhook_secret_missing" in r.lower()


@pytest.mark.asyncio
async def test_mcp_renders_mixed_severity():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "overall": "ERROR",
            "error_count": 1,
            "warn_count": 1,
            "info_count": 0,
            "findings": [
                {
                    "severity": "ERROR",
                    "cause": "x",
                    "remediation": "fix x",
                },
                {
                    "severity": "WARN",
                    "cause": "y",
                    "remediation": "fix y",
                },
            ],
        }),
    ):
        r = await handle_prsm_fiat_surface_health({})
    assert "ERROR" in r
    assert "WARN" in r
    assert "fix x" in r
    assert "fix y" in r

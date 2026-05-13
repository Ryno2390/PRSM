"""Sprint 344 — prsm_node_health renders sprint-342/343
subsystems with their count fields inline.

Sprint 342 added FL + pipeline-inference orchestrators with
`jobs_count`; sprint 343 added 5 more stores with
`record_count`. Both fields were left in the structured payload
but never surfaced by `prsm_node_health` — AI-assisted
operators triaging via the MCP tool saw only the [ok]/[--]
marker + status text and had to drill into /health/detailed
to see the counts.

Sprint 344 extends the per-subsystem decoration switch (same
pattern used for payment_escrow.pending_count and
bootstrap_discovery.client_state) to surface the count fields
inline.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import handle_prsm_node_health


def _payload(subsystem_name: str, sub_info: dict):
    return {
        "status": "healthy",
        "node_id": "test-node",
        "subsystems": {
            "ftns_ledger": {"available": True, "status": "ok"},
            "payment_escrow": {"available": True, "status": "ok"},
            subsystem_name: sub_info,
        },
    }


@pytest.mark.parametrize("name", [
    "federated_learning_orchestrator",
    "pipeline_inference_orchestrator",
])
def test_orchestrator_jobs_count_rendered_inline(name):
    payload = _payload(name, {
        "available": True, "status": "ok",
        "jobs_count": 7,
    })
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_node_health({}))
    assert name in out
    assert "jobs=7" in out, (
        f"expected jobs_count surfaced inline; got: {out}"
    )


@pytest.mark.parametrize("name", [
    "content_filter_store",
    "disclosure_intake",
    "incident_response",
    "corp_capability_store",
    "upgrade_orchestrator",
])
def test_store_record_count_rendered_inline(name):
    payload = _payload(name, {
        "available": True, "status": "ok",
        "record_count": 12,
    })
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_node_health({}))
    assert name in out
    assert "records=12" in out


def test_not_wired_renders_cleanly_without_count_fields():
    """Opt-out (not_wired) → no jobs_count / record_count in
    payload → no decoration; generic path renders cleanly."""
    payload = _payload("federated_learning_orchestrator", {
        "available": False, "status": "not_wired",
    })
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_node_health({}))
    assert "federated_learning_orchestrator" in out
    assert "not_wired" in out
    assert "jobs=" not in out
    assert "records=" not in out


def test_error_status_surfaces_reason():
    """Wired but probe raised → status=error + error field
    surfaced by generic error-rendering path."""
    payload = _payload("upgrade_orchestrator", {
        "available": False, "status": "error",
        "error": "disk full",
    })
    with patch("prsm.mcp_server._call_node_api",
               new=AsyncMock(return_value=payload)):
        out = asyncio.run(handle_prsm_node_health({}))
    assert "upgrade_orchestrator" in out
    assert "disk full" in out

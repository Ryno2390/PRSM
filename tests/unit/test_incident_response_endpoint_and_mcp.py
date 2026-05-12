"""Sprint 301 — incident response HTTP + MCP surface.

POST /admin/incident/open            — open new incident
GET  /admin/incident                 — list (filtered)
GET  /admin/incident/{id}            — single record + timeline
POST /admin/incident/{id}/advance    — advance phase
POST /admin/incident/{id}/event      — record free-form event
GET  /admin/incident/{id}/recommendations — playbook actions
GET  /admin/incident/{id}/comms-template  — comms text

prsm_incident MCP tool (action: open | list | lookup |
advance | event | recommend | comms).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.incident_response import (
    IncidentPhase, IncidentResponse, IncidentSeverity,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_incident,
)
from prsm.node.api import create_api_app


def _client(ir=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._incident_response = ir
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


# ── open ────────────────────────────────────────────


def test_open_503_when_unwired():
    resp = _client(None).post(
        "/admin/incident/open",
        json={"severity": "s1", "summary": "x"},
    )
    assert resp.status_code == 503


def test_open_happy_path():
    ir = IncidentResponse()
    resp = _client(ir).post(
        "/admin/incident/open",
        json={
            "severity": "s0",
            "summary": "reentrancy drain in progress",
            "affected_contracts": ["royalty_distributor"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["severity"] == "s0"
    assert body["current_phase"] == "detected"
    assert body["incident_id"]
    assert ir.count() == 1


def test_open_422_invalid_severity():
    resp = _client(IncidentResponse()).post(
        "/admin/incident/open",
        json={"severity": "extinction", "summary": "x"},
    )
    assert resp.status_code == 422


def test_open_422_empty_summary():
    resp = _client(IncidentResponse()).post(
        "/admin/incident/open",
        json={"severity": "s1", "summary": ""},
    )
    assert resp.status_code == 422


# ── list + get ──────────────────────────────────────


def test_list_503_when_unwired():
    assert _client(None).get(
        "/admin/incident",
    ).status_code == 503


def test_list_empty():
    body = _client(IncidentResponse()).get(
        "/admin/incident",
    ).json()
    assert body["records"] == []
    assert body["count"] == 0


def test_list_filter_by_severity():
    ir = IncidentResponse()
    ir.open(
        severity=IncidentSeverity.S0,
        summary="a", affected_contracts=[],
    )
    ir.open(
        severity=IncidentSeverity.S2,
        summary="b", affected_contracts=[],
    )
    body = _client(ir).get(
        "/admin/incident?severity=s0",
    ).json()
    assert body["count"] == 1
    assert body["records"][0]["summary"] == "a"


def test_list_filter_by_phase():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x", affected_contracts=[],
    )
    ir.advance_phase(r.incident_id, IncidentPhase.TRIAGED)
    ir.open(
        severity=IncidentSeverity.S1,
        summary="y", affected_contracts=[],
    )
    body = _client(ir).get(
        "/admin/incident?phase=triaged",
    ).json()
    assert body["count"] == 1
    assert body["records"][0]["summary"] == "x"


def test_get_404_unknown():
    assert _client(IncidentResponse()).get(
        "/admin/incident/no-such",
    ).status_code == 404


def test_get_happy_path():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x", affected_contracts=[],
    )
    body = _client(ir).get(
        f"/admin/incident/{r.incident_id}",
    ).json()
    assert body["incident_id"] == r.incident_id
    assert len(body["timeline"]) == 1


# ── advance + event ─────────────────────────────────


def test_advance_happy_path():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x", affected_contracts=[],
    )
    resp = _client(ir).post(
        f"/admin/incident/{r.incident_id}/advance",
        json={
            "new_phase": "triaged",
            "note": "investigating",
            "actor": "oncall",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_phase"] == "triaged"
    assert body["timeline"][-1]["actor"] == "oncall"


def test_advance_422_backwards():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x", affected_contracts=[],
    )
    ir.advance_phase(r.incident_id, IncidentPhase.TRIAGED)
    resp = _client(ir).post(
        f"/admin/incident/{r.incident_id}/advance",
        json={"new_phase": "detected"},
    )
    assert resp.status_code == 422


def test_advance_404_unknown():
    resp = _client(IncidentResponse()).post(
        "/admin/incident/no-such/advance",
        json={"new_phase": "triaged"},
    )
    assert resp.status_code == 404


def test_advance_422_invalid_phase():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x", affected_contracts=[],
    )
    resp = _client(ir).post(
        f"/admin/incident/{r.incident_id}/advance",
        json={"new_phase": "wat"},
    )
    assert resp.status_code == 422


def test_event_happy_path():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x", affected_contracts=[],
    )
    resp = _client(ir).post(
        f"/admin/incident/{r.incident_id}/event",
        json={
            "note": "paused royalty_distributor",
            "actor": "multisig",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["timeline"]) == 2
    assert (
        body["timeline"][-1]["note"]
        == "paused royalty_distributor"
    )


def test_event_404_unknown():
    resp = _client(IncidentResponse()).post(
        "/admin/incident/no-such/event",
        json={"note": "x"},
    )
    assert resp.status_code == 404


def test_event_422_empty_note():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S1,
        summary="x", affected_contracts=[],
    )
    resp = _client(ir).post(
        f"/admin/incident/{r.incident_id}/event",
        json={"note": ""},
    )
    assert resp.status_code == 422


# ── recommendations + comms ─────────────────────────


def test_recommendations_happy_path():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S0,
        summary="drain",
        affected_contracts=["royalty_distributor"],
    )
    body = _client(ir).get(
        f"/admin/incident/{r.incident_id}/recommendations",
    ).json()
    assert body["severity"] == "s0"
    assert body["current_phase"] == "detected"
    blob = " ".join(body["recommendations"]).lower()
    assert "pause" in blob


def test_recommendations_404_unknown():
    assert _client(IncidentResponse()).get(
        "/admin/incident/no-such/recommendations",
    ).status_code == 404


def test_comms_template_happy_path():
    ir = IncidentResponse()
    r = ir.open(
        severity=IncidentSeverity.S0,
        summary="drain on royalty_distributor",
        affected_contracts=["royalty_distributor"],
    )
    body = _client(ir).get(
        f"/admin/incident/{r.incident_id}/comms-template",
    ).json()
    assert "drain on royalty_distributor" in body["text"]
    assert body["severity"] == "s0"


def test_comms_template_404_unknown():
    assert _client(IncidentResponse()).get(
        "/admin/incident/no-such/comms-template",
    ).status_code == 404


# ── Public playbook endpoint ────────────────────────


def test_public_playbook_returns_all_severities():
    """Vision §14: playbook is PUBLIC."""
    body = _client(IncidentResponse()).get(
        "/admin/incident/playbook",
    ).json()
    # All four severities should be present at DETECTED
    severities = {
        entry["severity"] for entry in body["decision_tree"]
    }
    assert {"s0", "s1", "s2", "s3"} <= severities


# ── MCP tool ────────────────────────────────────────


def test_mcp_tool_registered():
    assert "prsm_incident" in TOOL_HANDLERS


@pytest.mark.asyncio
async def test_mcp_missing_action():
    r = await handle_prsm_incident({})
    assert "action" in r.lower()


@pytest.mark.asyncio
async def test_mcp_unknown_action():
    r = await handle_prsm_incident({"action": "boom"})
    assert "must be" in r.lower()


@pytest.mark.asyncio
async def test_mcp_list_renders_table():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "records": [{
                "incident_id": "abc12345-6789",
                "opened_ts": 100.0,
                "severity": "s0",
                "summary": "drain",
                "affected_contracts": ["royalty_distributor"],
                "current_phase": "detected",
                "timeline": [],
                "related_disclosure_id": None,
            }],
            "count": 1,
        }),
    ):
        r = await handle_prsm_incident({"action": "list"})
    assert "abc12345" in r
    assert "s0" in r.lower()
    assert "drain" in r


@pytest.mark.asyncio
async def test_mcp_open_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "incident_id": "new-id-1",
            "opened_ts": 100.0,
            "severity": "s1",
            "summary": "x",
            "affected_contracts": [],
            "current_phase": "detected",
            "timeline": [],
            "related_disclosure_id": None,
        }),
    ) as mock_call:
        r = await handle_prsm_incident({
            "action": "open",
            "severity": "s1",
            "summary": "x",
            "affected_contracts": [],
        })
    args = mock_call.await_args[0]
    assert args[0] == "POST"
    assert args[1] == "/admin/incident/open"
    assert "new-id-1" in r


@pytest.mark.asyncio
async def test_mcp_open_requires_severity():
    r = await handle_prsm_incident({
        "action": "open", "summary": "x",
    })
    assert "severity" in r.lower()


@pytest.mark.asyncio
async def test_mcp_lookup_renders_timeline():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "incident_id": "i-1",
            "opened_ts": 100.0,
            "severity": "s1",
            "summary": "x",
            "affected_contracts": ["royalty_distributor"],
            "current_phase": "triaged",
            "timeline": [
                {
                    "ts": 100.0, "phase": "detected",
                    "note": "opened", "actor": "oncall",
                },
                {
                    "ts": 200.0, "phase": "triaged",
                    "note": "confirmed", "actor": "oncall",
                },
            ],
            "related_disclosure_id": None,
        }),
    ):
        r = await handle_prsm_incident({
            "action": "lookup", "incident_id": "i-1",
        })
    assert "i-1" in r
    assert "triaged" in r.lower()
    assert "opened" in r
    assert "confirmed" in r


@pytest.mark.asyncio
async def test_mcp_advance_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "incident_id": "i-1",
            "opened_ts": 100.0,
            "severity": "s1",
            "summary": "x",
            "affected_contracts": [],
            "current_phase": "triaged",
            "timeline": [],
            "related_disclosure_id": None,
        }),
    ) as mock_call:
        r = await handle_prsm_incident({
            "action": "advance",
            "incident_id": "i-1",
            "new_phase": "triaged",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/incident/i-1/advance"
    assert "triaged" in r.lower()


@pytest.mark.asyncio
async def test_mcp_event_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "incident_id": "i-1",
            "opened_ts": 100.0,
            "severity": "s1",
            "summary": "x",
            "affected_contracts": [],
            "current_phase": "detected",
            "timeline": [
                {
                    "ts": 100.0, "phase": "detected",
                    "note": "opened", "actor": "",
                },
                {
                    "ts": 200.0, "phase": "detected",
                    "note": "paused", "actor": "multisig",
                },
            ],
            "related_disclosure_id": None,
        }),
    ) as mock_call:
        r = await handle_prsm_incident({
            "action": "event",
            "incident_id": "i-1",
            "note": "paused",
            "actor": "multisig",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/incident/i-1/event"
    assert "paused" in r


@pytest.mark.asyncio
async def test_mcp_recommend_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "incident_id": "i-1",
            "severity": "s0",
            "current_phase": "detected",
            "recommendations": [
                "PAUSE NOW",
                "Notify multisig signers",
            ],
        }),
    ) as mock_call:
        r = await handle_prsm_incident({
            "action": "recommend",
            "incident_id": "i-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/incident/i-1/recommendations"
    )
    assert "PAUSE NOW" in r


@pytest.mark.asyncio
async def test_mcp_comms_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "incident_id": "i-1",
            "severity": "s0",
            "current_phase": "detected",
            "text": "# ⚠ PRSM Active Incident — S0\n\nfoo",
        }),
    ) as mock_call:
        r = await handle_prsm_incident({
            "action": "comms",
            "incident_id": "i-1",
        })
    args = mock_call.await_args[0]
    assert args[1] == (
        "/admin/incident/i-1/comms-template"
    )
    assert "S0" in r


@pytest.mark.asyncio
async def test_mcp_playbook_action():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "decision_tree": [
                {
                    "severity": "s0",
                    "phase": "detected",
                    "recommendations": ["PAUSE NOW"],
                },
                {
                    "severity": "s3",
                    "phase": "detected",
                    "recommendations": ["Log + monitor."],
                },
            ],
        }),
    ) as mock_call:
        r = await handle_prsm_incident({
            "action": "playbook",
        })
    args = mock_call.await_args[0]
    assert args[1] == "/admin/incident/playbook"
    assert "PAUSE NOW" in r
    assert "s0" in r.lower()

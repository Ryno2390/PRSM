"""Sprint 260 — route_filter on JobHistoryStore.list/count +
/compute/jobs + prsm_jobs_list MCP wrapper.

After sprints 251 (inference) + 252 (inference/stream) wrote
into JobHistoryStore, prsm_jobs_list returned a mixed view
(forge + inference + inference_stream). Operators wanting to
scope to a single path had no filter.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.job_history import (
    JobHistoryRecord, JobHistoryStore, JobStatus,
)


def _rec(job_id, route="forge", status=JobStatus.COMPLETED):
    return JobHistoryRecord(
        job_id=job_id,
        query="hi",
        status=status,
        started_at=100.0 + len(job_id),
        route=route,
    )


# ── Store API ────────────────────────────────────────────


def test_list_route_filter_returns_subset():
    s = JobHistoryStore()
    s.put(_rec("j1", route="forge"))
    s.put(_rec("j2", route="inference"))
    s.put(_rec("j3", route="inference_stream"))
    s.put(_rec("j4", route="forge"))
    forge_only = s.list(route_filter="forge")
    assert {r.job_id for r in forge_only} == {"j1", "j4"}


def test_count_route_filter():
    s = JobHistoryStore()
    s.put(_rec("j1", route="forge"))
    s.put(_rec("j2", route="inference"))
    s.put(_rec("j3", route="forge"))
    assert s.count(route_filter="forge") == 2
    assert s.count(route_filter="inference") == 1
    assert s.count(route_filter="missing") == 0
    assert s.count() == 3  # no filter = total


def test_list_route_and_status_compose():
    s = JobHistoryStore()
    s.put(_rec("j1", route="forge", status=JobStatus.COMPLETED))
    s.put(_rec("j2", route="forge", status=JobStatus.FAILED))
    s.put(_rec("j3", route="inference", status=JobStatus.COMPLETED))
    matches = s.list(
        route_filter="forge",
        status_filter=JobStatus.FAILED,
    )
    assert [r.job_id for r in matches] == ["j2"]


# ── Endpoint ─────────────────────────────────────────────


def _client(store):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._job_history = store
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_endpoint_route_filter_passes_through():
    s = JobHistoryStore()
    s.put(_rec("j1", route="forge"))
    s.put(_rec("j2", route="inference"))
    resp = _client(s).get("/compute/jobs?route=inference")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["jobs"][0]["job_id"] == "j2"


def test_endpoint_rejects_huge_route_string():
    resp = _client(JobHistoryStore()).get(
        "/compute/jobs?route=" + "x" * 100,
    )
    assert resp.status_code == 422


# ── MCP wrapper ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_passes_route_to_query_string():
    from prsm.mcp_server import handle_prsm_jobs_list
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "jobs": [], "total": 0, "offset": 0, "limit": 20,
        }),
    ) as mock_call:
        await handle_prsm_jobs_list({"route": "inference"})
    args, _ = mock_call.await_args
    assert "route=inference" in args[1]


@pytest.mark.asyncio
async def test_mcp_empty_route_arg_skipped():
    """Empty-string route should not be appended."""
    from prsm.mcp_server import handle_prsm_jobs_list
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "jobs": [], "total": 0, "offset": 0, "limit": 20,
        }),
    ) as mock_call:
        await handle_prsm_jobs_list({"route": ""})
    args, _ = mock_call.await_args
    assert "route=" not in args[1]

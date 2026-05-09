"""prsm_jobs_list MCP tool handler.

Wraps GET /compute/jobs to surface a paginated, filterable
operator job list via the AI side-panel.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_jobs_list,
)


class TestToolRegistration:
    def test_handler_registered(self):
        assert "prsm_jobs_list" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_jobs_list" in names

    def test_status_enum_in_schema(self):
        tool = next(t for t in TOOLS if t.name == "prsm_jobs_list")
        status_enum = tool.inputSchema["properties"]["status"]["enum"]
        assert {"in_progress", "completed", "failed", "cancelled"}.issubset(
            set(status_enum)
        )


class TestJobsListHandler:
    @pytest.mark.asyncio
    async def test_renders_job_table(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "GET"
            assert "/compute/jobs" in path
            return {
                "jobs": [
                    {
                        "job_id": "forge-aaa",
                        "status": "completed",
                        "started_at": 1_700_000_000.0,
                        "query": "Count records",
                    },
                    {
                        "job_id": "forge-bbb",
                        "status": "in_progress",
                        "started_at": 1_700_000_100.0,
                        "query": "Aggregate by tag",
                    },
                ],
                "total": 2,
                "offset": 0,
                "limit": 20,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_jobs_list({})
        assert "forge-aaa" in result
        assert "completed" in result
        assert "forge-bbb" in result
        assert "in_progress" in result
        # Pagination summary present.
        assert "1–2 of 2" in result or "1-2 of 2" in result

    @pytest.mark.asyncio
    async def test_passes_status_filter_in_query_string(self):
        captured = {}

        async def fake_call_node_api(method, path, data=None):
            captured["path"] = path
            return {"jobs": [], "total": 0, "offset": 0, "limit": 20}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            await handle_prsm_jobs_list({"status": "in_progress"})
        assert "status=in_progress" in captured["path"]

    @pytest.mark.asyncio
    async def test_pagination_hint_when_more_pages_exist(self):
        async def fake_call_node_api(method, path, data=None):
            # 50 total, returning 20 with offset 0 → 30 more.
            return {
                "jobs": [
                    {"job_id": f"forge-{i}", "status": "completed",
                     "started_at": 0.0, "query": ""}
                    for i in range(20)
                ],
                "total": 50,
                "offset": 0,
                "limit": 20,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_jobs_list({})
        # Hint to fetch next page.
        assert "offset=20" in result

    @pytest.mark.asyncio
    async def test_empty_result_renders_friendly_message(self):
        async def fake_call_node_api(method, path, data=None):
            return {"jobs": [], "total": 0, "offset": 0, "limit": 20}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_jobs_list({})
        assert "No jobs" in result


class TestJobsListErrors:
    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_jobs_list({})
        assert "cannot reach" in result.lower()

    @pytest.mark.asyncio
    async def test_503_history_not_configured(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "detail": "JobHistoryStore is not initialized on this node.",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_jobs_list({})
        assert "not configured" in result.lower() or \
            "not initialized" in result.lower()

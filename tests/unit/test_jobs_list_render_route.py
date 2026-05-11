"""Sprint 261 — prsm_jobs_list renders route per row.

Sprint 260 added route filtering. Without route in the
rendered output, operators viewing a mixed list (no route
filter) couldn't visually distinguish forge from inference
records.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import handle_prsm_jobs_list


@pytest.mark.asyncio
async def test_render_shows_route_per_row():
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "jobs": [
                {
                    "job_id": "j-forge",
                    "status": "completed",
                    "started_at": 1715000000.0,
                    "route": "forge",
                    "query": "compute me",
                },
                {
                    "job_id": "j-inference",
                    "status": "completed",
                    "started_at": 1715000010.0,
                    "route": "inference",
                    "query": "what is x",
                },
                {
                    "job_id": "j-stream",
                    "status": "in_progress",
                    "started_at": 1715000020.0,
                    "route": "inference_stream",
                    "query": "stream x",
                },
            ],
            "total": 3, "offset": 0, "limit": 20,
        }),
    ):
        result = await handle_prsm_jobs_list({})

    assert "route=forge" in result
    assert "route=inference" in result
    assert "route=inference_stream" in result


@pytest.mark.asyncio
async def test_render_missing_route_shows_question_mark():
    """Pre-251/252 records have route=None. Render gracefully."""
    with patch(
        "prsm.mcp_server._call_node_api",
        new=AsyncMock(return_value={
            "jobs": [{
                "job_id": "legacy",
                "status": "completed",
                "started_at": 1715000000.0,
                "route": None,
                "query": "old",
            }],
            "total": 1, "offset": 0, "limit": 20,
        }),
    ):
        result = await handle_prsm_jobs_list({})
    assert "route=?" in result

"""prsm_forge_submit MCP wrapper for /compute/forge."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_forge_submit,
)


class TestRegistration:
    def test_handler_registered(self):
        assert "prsm_forge_submit" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_forge_submit" in names


class TestHandler:
    @pytest.mark.asyncio
    async def test_renders_job_id_on_success(self):
        captured = {}

        async def fake_call_node_api(method, path, data=None):
            captured["method"] = method
            captured["path"] = path
            captured["data"] = data
            return {
                "job_id": "job-abc",
                "status": "PENDING",
            }

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_forge_submit(
                {"query": "What is X?"}
            )
        assert "job-abc" in result
        assert "PENDING" in result
        assert captured["method"] == "POST"
        assert captured["path"] == "/compute/forge"
        assert captured["data"]["query"] == "What is X?"

    @pytest.mark.asyncio
    async def test_passes_optional_args(self):
        captured = {}

        async def fake_call_node_api(method, path, data=None):
            captured["data"] = data
            return {"job_id": "j1", "status": "PENDING"}

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            await handle_prsm_forge_submit({
                "query": "test",
                "budget_ftns": 50.0,
                "shard_cids": ["cid1", "cid2"],
                "privacy_level": "high",
            })
        assert captured["data"]["budget_ftns"] == 50.0
        assert captured["data"]["shard_cids"] == ["cid1", "cid2"]
        assert captured["data"]["privacy_level"] == "high"

    @pytest.mark.asyncio
    async def test_idempotent_replay_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "idempotent_replay",
                "job_id": "job-prior",
            }

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_forge_submit({"query": "x"})
        assert "Idempotent replay" in result
        assert "job-prior" in result

    @pytest.mark.asyncio
    async def test_agent_forge_not_enabled_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {"detail": "agent_forge not available"}

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_forge_submit({"query": "x"})
        assert "PRSM_QUERY_ORCHESTRATOR_ENABLED" in result

    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")

        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_forge_submit({"query": "x"})
        assert "Cannot reach PRSM node" in result

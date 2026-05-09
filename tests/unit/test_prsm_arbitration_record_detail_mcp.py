"""prsm_arbitration_record_detail MCP tool handler.

Wraps GET /content/arbitration/queue/{record_id} for council-side
detail-view from the AI side panel.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_arbitration_record_detail,
)


class TestToolRegistration:
    def test_handler_registered(self):
        assert "prsm_arbitration_record_detail" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_arbitration_record_detail" in names

    def test_record_id_required(self):
        tool = next(
            t for t in TOOLS
            if t.name == "prsm_arbitration_record_detail"
        )
        assert "record_id" in tool.inputSchema["required"]


class TestHandlerHappyPath:
    @pytest.mark.asyncio
    async def test_pending_record_renders_full_detail(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "GET"
            assert "/content/arbitration/queue/test-record-id" in path
            return {
                "record": {
                    "new_cid": "bafy-x",
                    "new_creator": "0x1111",
                    "candidate_parent_cid": "bafy-y",
                    "candidate_parent_creator": "0x2222",
                    "similarity": 0.88,
                    "fingerprint_kind": "image",
                    "flagged_at": 1_700_000_000,
                    "proposal_id": "prop-7",
                },
                "resolution": None,
                "status": "pending",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_arbitration_record_detail({
                "record_id": "test-record-id",
            })
        assert "bafy-x" in result
        assert "bafy-y" in result
        assert "0.8800" in result
        assert "image" in result
        assert "prop-7" in result
        assert "PENDING" in result.upper()

    @pytest.mark.asyncio
    async def test_resolved_record_includes_decision_block(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "record": {
                    "new_cid": "bafy-x",
                    "new_creator": "0x1111",
                    "candidate_parent_cid": "bafy-y",
                    "candidate_parent_creator": "0x2222",
                    "similarity": 0.92,
                    "fingerprint_kind": "audio",
                    "flagged_at": 1_700_000_000,
                    "proposal_id": None,
                },
                "resolution": {
                    "decision": "upheld_parent",
                    "by_council": ["0xCouncil1", "0xCouncil2"],
                },
                "status": "resolved",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_arbitration_record_detail({
                "record_id": "id-1",
            })
        assert "RESOLVED" in result.upper()
        assert "upheld_parent" in result
        assert "0xCouncil1" in result
        assert "0xCouncil2" in result


class TestHandlerErrors:
    @pytest.mark.asyncio
    async def test_missing_record_id_returns_user_error(self):
        result = await handle_prsm_arbitration_record_detail({})
        assert "missing" in result.lower()
        assert "prsm_arbitration_status" in result

    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_arbitration_record_detail({
                "record_id": "x",
            })
        assert "cannot reach" in result.lower()

    @pytest.mark.asyncio
    async def test_404_not_found(self):
        async def fake_call_node_api(method, path, data=None):
            return {"detail": "No arbitration record for id='missing'"}
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_arbitration_record_detail({
                "record_id": "missing",
            })
        assert "not found" in result.lower()

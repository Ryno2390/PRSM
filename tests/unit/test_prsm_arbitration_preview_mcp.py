"""prsm_arbitration_preview_resolution MCP tool handler.

Composer-only — wraps POST /content/arbitration/preview-resolution.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS,
    handle_prsm_arbitration_preview_resolution,
)


class TestRegistration:
    def test_handler_registered(self):
        assert "prsm_arbitration_preview_resolution" in TOOL_HANDLERS

    def test_required_args(self):
        tool = next(
            t for t in TOOLS
            if t.name == "prsm_arbitration_preview_resolution"
        )
        required = tool.inputSchema["required"]
        assert "record_id" in required
        assert "decision" in required
        assert "by_council" in required

    def test_decision_enum_in_schema(self):
        tool = next(
            t for t in TOOLS
            if t.name == "prsm_arbitration_preview_resolution"
        )
        decision_enum = tool.inputSchema["properties"]["decision"]["enum"]
        assert {"upheld_parent", "rejected_parent", "insufficient"} == \
            set(decision_enum)


class TestHandler:
    @pytest.mark.asyncio
    async def test_renders_no_conflict_preview(self):
        async def fake_call_node_api(method, path, data=None):
            assert method == "POST"
            assert "/content/arbitration/preview-resolution" in path
            assert data["decision"] == "upheld_parent"
            return {
                "status": "DRY_RUN",
                "record": {
                    "new_cid": "bafy-x",
                    "candidate_parent_cid": "bafy-y",
                    "similarity": 0.88,
                    "fingerprint_kind": "image",
                },
                "proposed": {
                    "decision": "upheld_parent",
                    "by_council": ["0xC1", "0xC2"],
                },
                "current_resolution": None,
                "conflict_with_existing": False,
                "note": "Composer-only.",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_arbitration_preview_resolution({
                "record_id": "rec-1",
                "decision": "upheld_parent",
                "by_council": ["0xC1", "0xC2"],
            })
        assert "DRY_RUN" in result
        assert "bafy-x" in result
        assert "0xC1" in result
        assert "0xC2" in result
        assert "CONFLICT" not in result.upper()

    @pytest.mark.asyncio
    async def test_renders_conflict_warning(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "DRY_RUN",
                "record": {
                    "new_cid": "bafy-x",
                    "candidate_parent_cid": "bafy-y",
                    "similarity": 0.88,
                    "fingerprint_kind": "image",
                },
                "proposed": {
                    "decision": "rejected_parent",
                    "by_council": ["0xC1"],
                },
                "current_resolution": {
                    "decision": "upheld_parent",
                    "by_council": ["0xC2"],
                },
                "conflict_with_existing": True,
                "note": "Composer-only.",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_arbitration_preview_resolution({
                "record_id": "rec-1",
                "decision": "rejected_parent",
                "by_council": ["0xC1"],
            })
        assert "CONFLICT" in result
        # Existing resolution surfaced.
        assert "upheld_parent" in result
        # Reconcile-with-council hint present.
        assert "council" in result.lower()

    @pytest.mark.asyncio
    async def test_renders_no_op_for_idempotent_re_resolve(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "status": "DRY_RUN",
                "record": {
                    "new_cid": "bafy-x",
                    "candidate_parent_cid": "bafy-y",
                    "similarity": 0.88,
                    "fingerprint_kind": "image",
                },
                "proposed": {
                    "decision": "upheld_parent",
                    "by_council": ["0xC1"],
                },
                "current_resolution": {
                    "decision": "upheld_parent",
                    "by_council": ["0xC1"],
                },
                "conflict_with_existing": False,
                "note": "Composer-only.",
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_arbitration_preview_resolution({
                "record_id": "rec-1",
                "decision": "upheld_parent",
                "by_council": ["0xC1"],
            })
        # No-op hint surfaced.
        assert "no-op" in result.lower() or "no conflict" in result.lower()


class TestErrors:
    @pytest.mark.asyncio
    async def test_missing_args(self):
        result = await handle_prsm_arbitration_preview_resolution({
            "record_id": "x",
        })
        assert "missing" in result.lower()

    @pytest.mark.asyncio
    async def test_node_unreachable(self):
        async def boom(method, path, data=None):
            raise RuntimeError("connection refused")
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=boom,
        ):
            result = await handle_prsm_arbitration_preview_resolution({
                "record_id": "x",
                "decision": "upheld_parent",
                "by_council": ["0xC1"],
            })
        assert "cannot reach" in result.lower()

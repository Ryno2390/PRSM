"""Sprint 272 — prsm_takedown_notices MCP tool.

Wraps Foundation takedown-notice intake endpoints behind a
single action-selector tool. Per Vision §14 / R9-SCOPING-1 §8
this surface is information distribution only — operators
voluntarily act via sprint-269/270 prsm_content_filter.

Actions:
  - list:    GET  /admin/takedown-notices
  - lookup:  GET  /admin/takedown-notices/{notice_id}
  - record:  POST /admin/takedown-notice
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_takedown_notices,
)


def test_tool_registered():
    assert "prsm_takedown_notices" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        r = await handle_prsm_takedown_notices({})
        assert "action" in r.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        r = await handle_prsm_takedown_notices(
            {"action": "explode"},
        )
        assert "must be" in r.lower()


class TestList:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "notices": [], "total": 0,
                "offset": 0, "limit": 50,
            }),
        ) as mock_call:
            r = await handle_prsm_takedown_notices(
                {"action": "list"},
            )
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1].startswith("/admin/takedown-notices?")
        assert "0 of 0" in r
        assert "(none)" in r

    @pytest.mark.asyncio
    async def test_list_renders_notices(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "notices": [
                    {
                        "notice_id": "abcdef0123456789",
                        "status": "received",
                        "target_cid": "bafy-target",
                        "jurisdiction": "US-DMCA",
                        "basis": "DMCA §512(c)",
                    },
                ],
                "total": 1, "offset": 0, "limit": 50,
            }),
        ):
            r = await handle_prsm_takedown_notices(
                {"action": "list"},
            )
        assert "abcdef01" in r
        assert "received" in r
        assert "bafy-target" in r
        assert "US-DMCA" in r

    @pytest.mark.asyncio
    async def test_list_passes_filters_in_query(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "notices": [], "total": 0,
                "offset": 0, "limit": 50,
            }),
        ) as mock_call:
            await handle_prsm_takedown_notices({
                "action": "list",
                "status": "received",
                "target_cid": "bafy-x",
                "limit": 10, "offset": 5,
            })
        path = mock_call.await_args[0][1]
        assert "status=received" in path
        assert "target_cid=bafy-x" in path
        assert "limit=10" in path
        assert "offset=5" in path

    @pytest.mark.asyncio
    async def test_list_not_initialized_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "Takedown notice ring not initialized.",
            }),
        ):
            r = await handle_prsm_takedown_notices(
                {"action": "list"},
            )
        assert "not wired" in r.lower()
        assert "PRSM_TAKEDOWN_NOTICE_LOG_DIR" in r


class TestLookup:
    @pytest.mark.asyncio
    async def test_lookup_requires_notice_id(self):
        r = await handle_prsm_takedown_notices(
            {"action": "lookup"},
        )
        assert "notice_id" in r

    @pytest.mark.asyncio
    async def test_lookup_renders_notice(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "notice_id": "abc-123",
                "timestamp": 100.0,
                "status": "received",
                "target_cid": "bafy-target",
                "sender": "legal@ex.com",
                "jurisdiction": "US-DMCA",
                "basis": "DMCA §512(c)",
                "notice_text": "body",
            }),
        ) as mock_call:
            r = await handle_prsm_takedown_notices(
                {"action": "lookup", "notice_id": "abc-123"},
            )
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/admin/takedown-notices/abc-123"
        assert "abc-123" in r
        assert "bafy-target" in r
        assert "legal@ex.com" in r

    @pytest.mark.asyncio
    async def test_lookup_missing_notice_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "no notice with id='missing'",
            }),
        ):
            r = await handle_prsm_takedown_notices(
                {"action": "lookup", "notice_id": "missing"},
            )
        assert "no notice" in r.lower()


class TestRecord:
    @pytest.mark.asyncio
    async def test_record_requires_fields(self):
        r = await handle_prsm_takedown_notices(
            {"action": "record"},
        )
        assert "target_cid" in r
        assert "sender" in r
        assert "jurisdiction" in r
        assert "basis" in r

    @pytest.mark.asyncio
    async def test_record_partial_missing(self):
        r = await handle_prsm_takedown_notices({
            "action": "record",
            "target_cid": "bafy1",
            "sender": "x",
            # missing jurisdiction + basis
        })
        assert "jurisdiction" in r
        assert "basis" in r
        assert "target_cid" not in r  # already provided

    @pytest.mark.asyncio
    async def test_record_happy_path(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "notice_id": "new-id-123",
                "target_cid": "bafy1",
                "status": "received",
            }),
        ) as mock_call:
            r = await handle_prsm_takedown_notices({
                "action": "record",
                "target_cid": "bafy1",
                "sender": "legal@ex.com",
                "jurisdiction": "US-DMCA",
                "basis": "DMCA §512(c)",
                "notice_text": "full body",
            })
        call_args = mock_call.await_args[0]
        assert call_args[0] == "POST"
        assert call_args[1] == "/admin/takedown-notice"
        body_dict = call_args[2]
        assert body_dict["target_cid"] == "bafy1"
        assert body_dict["sender"] == "legal@ex.com"
        assert body_dict["jurisdiction"] == "US-DMCA"
        assert "new-id-123" in r
        assert "received" in r


# Sprint 273 — apply_to_filter bridge action


class TestApplyToFilter:
    @pytest.mark.asyncio
    async def test_requires_notice_id(self):
        r = await handle_prsm_takedown_notices(
            {"action": "apply_to_filter"},
        )
        assert "notice_id" in r

    @pytest.mark.asyncio
    async def test_happy_path(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "notice_id": "abc-123",
                "target_cid": "bafy-blocked",
                "added": 1,
                "notice_status": "acknowledged",
            }),
        ) as mock_call:
            r = await handle_prsm_takedown_notices({
                "action": "apply_to_filter",
                "notice_id": "abc-123",
            })
        call_args = mock_call.await_args[0]
        assert call_args[0] == "POST"
        assert call_args[1] == (
            "/admin/content-filter/from-notice/abc-123"
        )
        assert "bafy-blocked" in r
        assert "acknowledged" in r

    @pytest.mark.asyncio
    async def test_idempotent_already_blocked(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "notice_id": "abc-123",
                "target_cid": "bafy-blocked",
                "added": 0,
                "notice_status": "acknowledged",
            }),
        ):
            r = await handle_prsm_takedown_notices({
                "action": "apply_to_filter",
                "notice_id": "abc-123",
            })
        assert "already" in r.lower()
        assert "acknowledged" in r

    @pytest.mark.asyncio
    async def test_missing_notice_message(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "no notice with id='missing'",
            }),
        ):
            r = await handle_prsm_takedown_notices({
                "action": "apply_to_filter",
                "notice_id": "missing",
            })
        assert "no notice" in r.lower()

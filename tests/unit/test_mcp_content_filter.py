"""Sprint 270 — prsm_content_filter MCP tool.

Wraps the sprint-269 admin CRUD endpoints behind a single
action-selector tool. Mirrors prsm_agent_admin (sprint 221) +
prsm_settler_admin (sprint 234) patterns.

Actions:
  - list:        GET /admin/content-filter
  - add_cids:    POST /admin/content-filter/cids
  - remove_cid:  DELETE /admin/content-filter/cids/{cid}
  - add_tags:    POST /admin/content-filter/tags
  - remove_tag:  DELETE /admin/content-filter/tags/{tag}
  - set_action:  POST /admin/content-filter/action
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from prsm.mcp_server import (
    TOOL_HANDLERS, handle_prsm_content_filter,
)


def test_tool_registered():
    assert "prsm_content_filter" in TOOL_HANDLERS


class TestValidation:
    @pytest.mark.asyncio
    async def test_missing_action_rejected(self):
        r = await handle_prsm_content_filter({})
        assert "action" in r.lower()

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self):
        r = await handle_prsm_content_filter({"action": "explode"})
        assert "must be" in r.lower()


class TestList:
    @pytest.mark.asyncio
    async def test_list_renders_state(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "blocked_content_ids": ["bafy1", "bafy2"],
                "blocked_model_tags": ["safety"],
                "blocked_input_patterns": ["^evil"],
                "action_on_match": "log_and_refuse",
                "count_cids": 2,
                "count_tags": 1,
                "count_patterns": 1,
            }),
        ) as mock_call:
            r = await handle_prsm_content_filter({"action": "list"})
        args, _ = mock_call.await_args
        assert args[0] == "GET"
        assert args[1] == "/admin/content-filter"
        assert "bafy1" in r
        assert "safety" in r
        assert "log_and_refuse" in r

    @pytest.mark.asyncio
    async def test_list_503_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "Content filter store not initialized.",
            }),
        ):
            r = await handle_prsm_content_filter({"action": "list"})
        assert "not wired" in r.lower()


class TestAddCids:
    @pytest.mark.asyncio
    async def test_add_cids_happy(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"added": 2, "total": 5}),
        ) as mock_call:
            r = await handle_prsm_content_filter({
                "action": "add_cids",
                "cids": ["bafy1", "bafy2"],
            })
        args, _ = mock_call.await_args
        assert args[0] == "POST"
        assert args[1] == "/admin/content-filter/cids"
        body = args[2]
        assert body["cids"] == ["bafy1", "bafy2"]
        assert "added=2" in r
        assert "total=5" in r

    @pytest.mark.asyncio
    async def test_add_cids_missing_arg(self):
        r = await handle_prsm_content_filter({"action": "add_cids"})
        assert "cids" in r.lower()

    @pytest.mark.asyncio
    async def test_add_cids_not_list(self):
        r = await handle_prsm_content_filter({
            "action": "add_cids", "cids": "bafy1",
        })
        assert "list" in r.lower()


class TestRemoveCid:
    @pytest.mark.asyncio
    async def test_remove_cid_happy(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "removed": "bafy1", "total": 4,
            }),
        ) as mock_call:
            r = await handle_prsm_content_filter({
                "action": "remove_cid", "cid": "bafy1",
            })
        args, _ = mock_call.await_args
        assert args[0] == "DELETE"
        assert args[1] == "/admin/content-filter/cids/bafy1"
        assert "removed bafy1" in r.lower()

    @pytest.mark.asyncio
    async def test_remove_cid_missing_arg(self):
        r = await handle_prsm_content_filter({
            "action": "remove_cid",
        })
        assert "cid" in r.lower()

    @pytest.mark.asyncio
    async def test_remove_cid_404_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "detail": "cid='ghost' not in blocklist",
            }),
        ):
            r = await handle_prsm_content_filter({
                "action": "remove_cid", "cid": "ghost",
            })
        assert "not in blocklist" in r.lower()


class TestAddTags:
    @pytest.mark.asyncio
    async def test_add_tags_happy(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={"added": 1, "total": 3}),
        ) as mock_call:
            r = await handle_prsm_content_filter({
                "action": "add_tags", "tags": ["safety"],
            })
        args, _ = mock_call.await_args
        assert args[1] == "/admin/content-filter/tags"
        body = args[2]
        assert body["tags"] == ["safety"]
        assert "added=1" in r


class TestRemoveTag:
    @pytest.mark.asyncio
    async def test_remove_tag_happy(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "removed": "safety", "total": 2,
            }),
        ) as mock_call:
            r = await handle_prsm_content_filter({
                "action": "remove_tag", "tag": "safety",
            })
        args, _ = mock_call.await_args
        assert args[0] == "DELETE"
        assert "/admin/content-filter/tags/safety" in args[1]


class TestSetAction:
    @pytest.mark.asyncio
    async def test_set_action_happy(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(return_value={
                "action_on_match": "log_and_refuse",
            }),
        ) as mock_call:
            r = await handle_prsm_content_filter({
                "action": "set_action",
                "filter_action": "log_and_refuse",
            })
        args, _ = mock_call.await_args
        assert args[1] == "/admin/content-filter/action"
        body = args[2]
        assert body["action"] == "log_and_refuse"
        assert "log_and_refuse" in r

    @pytest.mark.asyncio
    async def test_set_action_missing_arg(self):
        r = await handle_prsm_content_filter({
            "action": "set_action",
        })
        assert "filter_action" in r.lower()


class TestNetworkError:
    @pytest.mark.asyncio
    async def test_unreachable_friendly(self):
        with patch(
            "prsm.mcp_server._call_node_api",
            new=AsyncMock(side_effect=RuntimeError("conn refused")),
        ):
            r = await handle_prsm_content_filter({"action": "list"})
        assert "running" in r.lower() or "failed" in r.lower()

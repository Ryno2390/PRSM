"""HeartbeatRecordedRing + GET /admin/heartbeat-history + MCP wrapper.

Symmetric to slash_event_log: separate ring, separate concern.
Wired to StorageSlashingWatcher via heartbeat_log= kwarg.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.storage_slashing import HeartbeatRecordedEvent
from prsm.node.api import create_api_app
from prsm.node.heartbeat_log import HeartbeatRecordedRing
from prsm.node.node import _build_storage_slashing_watcher_or_none
from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_heartbeat_history,
)


def _node(*, with_log=True):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node._heartbeat_log = HeartbeatRecordedRing() if with_log else None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ── Ring ───────────────────────────────────────────────────────


class TestRing:
    def test_append_and_recent(self):
        ring = HeartbeatRecordedRing()
        ring.append(provider="0xPROV", onchain_timestamp=1700000000)
        results = ring.recent()
        assert len(results) == 1
        assert results[0].provider == "0xPROV"
        assert results[0].onchain_timestamp == 1700000000

    def test_most_recent_first(self):
        ring = HeartbeatRecordedRing()
        for i in range(3):
            ring.append(provider=f"0xP{i}", onchain_timestamp=i)
        results = ring.recent()
        assert [e.provider for e in results] == ["0xP2", "0xP1", "0xP0"]

    def test_bounded(self):
        ring = HeartbeatRecordedRing(max_entries=2)
        for i in range(5):
            ring.append(provider=f"0xP{i}", onchain_timestamp=i)
        assert len(ring.recent()) == 2

    def test_provider_filter(self):
        ring = HeartbeatRecordedRing()
        ring.append(provider="0xMINE", onchain_timestamp=1)
        ring.append(provider="0xOTHER", onchain_timestamp=2)
        results = ring.recent(provider="0xMINE")
        assert len(results) == 1


# ── Endpoint ───────────────────────────────────────────────────


class TestEndpoint:
    def test_503_when_not_wired(self):
        node = _node(with_log=False)
        resp = _client(node).get("/admin/heartbeat-history")
        assert resp.status_code == 503

    def test_returns_recent(self):
        node = _node()
        node._heartbeat_log.append(
            provider="0xPROV", onchain_timestamp=1700000000,
        )
        resp = _client(node).get("/admin/heartbeat-history")
        body = resp.json()
        assert body["total"] == 1
        assert body["entries"][0]["provider"] == "0xPROV"
        assert body["entries"][0]["onchain_timestamp"] == 1700000000

    def test_provider_filter(self):
        node = _node()
        node._heartbeat_log.append(provider="0xMINE", onchain_timestamp=1)
        node._heartbeat_log.append(provider="0xOTHER", onchain_timestamp=2)
        resp = _client(node).get(
            "/admin/heartbeat-history?provider=0xMINE"
        )
        body = resp.json()
        assert len(body["entries"]) == 1
        assert body["entries"][0]["provider"] == "0xMINE"

    def test_invalid_limit_422(self):
        node = _node()
        resp = _client(node).get("/admin/heartbeat-history?limit=0")
        assert resp.status_code == 422


# ── Wiring ────────────────────────────────────────────────────


@pytest.fixture
def opted_in_env():
    with patch.dict(os.environ, {
        "PRSM_STORAGE_SLASHING_WATCHER_ENABLED": "1",
    }):
        yield


def test_heartbeat_recorded_routed_to_ring(opted_in_env):
    ring = HeartbeatRecordedRing()
    watcher = _build_storage_slashing_watcher_or_none(
        client=MagicMock(),
        heartbeat_log=ring,
    )
    event = HeartbeatRecordedEvent(
        provider="0xPROV", timestamp=1700000000,
    )
    watcher._on_recorded(event)

    entries = ring.recent()
    assert len(entries) == 1
    assert entries[0].provider == "0xPROV"
    assert entries[0].onchain_timestamp == 1700000000


def test_no_ring_arg_does_not_crash(opted_in_env):
    watcher = _build_storage_slashing_watcher_or_none(
        client=MagicMock(),
        heartbeat_log=None,
    )
    event = HeartbeatRecordedEvent(provider="0xP", timestamp=0)
    watcher._on_recorded(event)


def test_ring_failure_isolated_from_callback(opted_in_env):
    ring = MagicMock()
    ring.append.side_effect = RuntimeError("ring boom")
    watcher = _build_storage_slashing_watcher_or_none(
        client=MagicMock(),
        heartbeat_log=ring,
    )
    event = HeartbeatRecordedEvent(provider="0xP", timestamp=0)
    watcher._on_recorded(event)


# ── MCP ───────────────────────────────────────────────────────


class TestMcp:
    def test_handler_registered(self):
        assert "prsm_heartbeat_history" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_heartbeat_history" in names

    @pytest.mark.asyncio
    async def test_renders_recent(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "entries": [
                    {
                        "timestamp": 1700000000.0,
                        "provider": "0xPROVIDER",
                        "onchain_timestamp": 1700000000,
                    },
                ],
                "total": 1, "offset": 0, "limit": 20,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_heartbeat_history({})
        assert "PRSM Heartbeats" in result

    @pytest.mark.asyncio
    async def test_empty_friendly(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "entries": [], "total": 0, "offset": 0, "limit": 20,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_heartbeat_history({})
        assert "No heartbeats" in result

"""DistributedEventRing + /admin/distribution-history + MCP.

Symmetric to slash_event_log + heartbeat_log: separate ring
for Distributed events. Wired to CompensationDistributorWatcher.
"""
from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prsm.economy.web3.compensation_distributor import DistributedEvent
from prsm.node.api import create_api_app
from prsm.node.distribution_log import DistributedEventRing
from prsm.node.node import (
    _build_compensation_distributor_watcher_or_none,
)
from prsm.mcp_server import (
    TOOL_HANDLERS, TOOLS, handle_prsm_distribution_history,
)


def _node(*, with_log=True):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node._distribution_log = (
        DistributedEventRing() if with_log else None
    )
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ── Ring ───────────────────────────────────────────────────────


class TestRing:
    def test_append_and_recent(self):
        ring = DistributedEventRing()
        ring.append(to_creator=100, to_operator=50, to_grant=25)
        results = ring.recent()
        assert len(results) == 1
        assert results[0].to_creator == 100
        assert results[0].to_operator == 50
        assert results[0].to_grant == 25

    def test_to_dict_includes_total(self):
        ring = DistributedEventRing()
        ring.append(to_creator=100, to_operator=50, to_grant=25)
        d = ring.recent()[0].to_dict()
        assert d["total_distributed"] == 175

    def test_most_recent_first(self):
        ring = DistributedEventRing()
        for i in range(3):
            ring.append(
                to_creator=i, to_operator=0, to_grant=0,
            )
        results = ring.recent()
        assert [e.to_creator for e in results] == [2, 1, 0]


# ── Endpoint ───────────────────────────────────────────────────


class TestEndpoint:
    def test_503_when_not_wired(self):
        resp = _client(_node(with_log=False)).get(
            "/admin/distribution-history",
        )
        assert resp.status_code == 503

    def test_returns_recent(self):
        node = _node()
        node._distribution_log.append(
            to_creator=100, to_operator=50, to_grant=25,
        )
        resp = _client(node).get("/admin/distribution-history")
        body = resp.json()
        assert body["total"] == 1
        assert body["entries"][0]["to_creator"] == 100
        assert body["entries"][0]["total_distributed"] == 175

    def test_invalid_limit_422(self):
        resp = _client(_node()).get(
            "/admin/distribution-history?limit=0",
        )
        assert resp.status_code == 422


# ── Wiring ────────────────────────────────────────────────────


@pytest.fixture
def opted_in_env():
    with patch.dict(os.environ, {
        "PRSM_COMPENSATION_DISTRIBUTOR_WATCHER_ENABLED": "1",
    }):
        yield


def test_distributed_routed_to_ring(opted_in_env):
    ring = DistributedEventRing()
    watcher = _build_compensation_distributor_watcher_or_none(
        client=MagicMock(),
        distribution_log=ring,
    )
    event = DistributedEvent(
        to_creator=100, to_operator=50, to_grant=25,
    )
    asyncio.run(watcher._invoke_cb(event))

    entries = ring.recent()
    assert len(entries) == 1
    assert entries[0].to_creator == 100


def test_no_ring_does_not_crash(opted_in_env):
    watcher = _build_compensation_distributor_watcher_or_none(
        client=MagicMock(),
        distribution_log=None,
    )
    event = DistributedEvent(to_creator=0, to_operator=0, to_grant=0)
    asyncio.run(watcher._invoke_cb(event))


def test_ring_failure_isolated(opted_in_env):
    ring = MagicMock()
    ring.append.side_effect = RuntimeError("ring boom")
    watcher = _build_compensation_distributor_watcher_or_none(
        client=MagicMock(),
        distribution_log=ring,
    )
    event = DistributedEvent(to_creator=0, to_operator=0, to_grant=0)
    asyncio.run(watcher._invoke_cb(event))


# ── MCP ───────────────────────────────────────────────────────


class TestMcp:
    def test_handler_registered(self):
        assert "prsm_distribution_history" in TOOL_HANDLERS

    def test_tool_definition_present(self):
        names = [t.name for t in TOOLS]
        assert "prsm_distribution_history" in names

    @pytest.mark.asyncio
    async def test_renders_recent(self):
        async def fake_call_node_api(method, path, data=None):
            return {
                "entries": [
                    {
                        "timestamp": 1700000000.0,
                        "to_creator": 1_000_000_000_000_000_000,  # 1.0
                        "to_operator": 500_000_000_000_000_000,    # 0.5
                        "to_grant": 100_000_000_000_000_000,       # 0.1
                        "total_distributed": 1_600_000_000_000_000_000,
                    },
                ],
                "total": 1, "offset": 0, "limit": 20,
            }
        with patch(
            "prsm.mcp_server._call_node_api",
            side_effect=fake_call_node_api,
        ):
            result = await handle_prsm_distribution_history({})
        assert "PRSM Distributions" in result
        assert "1.0000" in result  # to_creator FTNS

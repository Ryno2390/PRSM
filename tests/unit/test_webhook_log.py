"""WebhookLogRing + GET /admin/webhook-history."""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.webhook_delivery import WebhookDeliverer, DeliveryResult
from prsm.node.webhook_log import WebhookLogRing


def _node(*, with_log=True):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._royalty_distributor_client = None
    node._webhook_log = WebhookLogRing() if with_log else None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# WebhookLogRing primitive
# ──────────────────────────────────────────────────────────────────────


class TestRing:
    def test_append_and_recent(self):
        ring = WebhookLogRing()
        ring.append(
            event="daemon.crashed",
            url="https://hook.example.com",
            success=True,
            attempts=1,
            status_code=200,
        )
        results = ring.recent()
        assert len(results) == 1
        assert results[0].event == "daemon.crashed"
        assert results[0].success is True

    def test_most_recent_first(self):
        ring = WebhookLogRing()
        for i in range(3):
            ring.append(
                event=f"e{i}", url="https://x.com",
                success=True, attempts=1,
            )
        results = ring.recent()
        assert [e.event for e in results] == ["e2", "e1", "e0"]

    def test_bounded_by_max_entries(self):
        ring = WebhookLogRing(max_entries=2)
        for i in range(5):
            ring.append(
                event=f"e{i}", url="https://x.com",
                success=True, attempts=1,
            )
        results = ring.recent()
        assert len(results) == 2
        assert [e.event for e in results] == ["e4", "e3"]


# ──────────────────────────────────────────────────────────────────────
# WebhookDeliverer auto-records to ring
# ──────────────────────────────────────────────────────────────────────


class TestDelivererAutoRecord:
    @pytest.mark.asyncio
    async def test_success_recorded(self):
        ring = WebhookLogRing()
        deliverer = WebhookDeliverer(log_ring=ring)

        async def fake_post(url, body, headers):
            return 200, "ok"

        await deliverer.deliver(
            url="https://hook.example.com",
            event="test.event",
            payload={"x": 1},
            post_fn=fake_post,
        )
        results = ring.recent()
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].status_code == 200
        assert results[0].event == "test.event"

    @pytest.mark.asyncio
    async def test_failure_recorded(self):
        ring = WebhookLogRing()

        async def no_sleep(s):
            return None

        deliverer = WebhookDeliverer(
            log_ring=ring, max_attempts=2,
        )
        deliverer._sleep = no_sleep

        async def fake_post(url, body, headers):
            return 503, "down"

        await deliverer.deliver(
            url="https://hook.example.com",
            event="test.event",
            payload={"x": 1},
            post_fn=fake_post,
        )
        results = ring.recent()
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].attempts == 2

    @pytest.mark.asyncio
    async def test_non_retryable_recorded(self):
        ring = WebhookLogRing()
        deliverer = WebhookDeliverer(log_ring=ring)

        async def fake_post(url, body, headers):
            return 400, "bad request"

        await deliverer.deliver(
            url="https://hook.example.com",
            event="test.event",
            payload={"x": 1},
            post_fn=fake_post,
        )
        results = ring.recent()
        assert len(results) == 1
        assert results[0].success is False
        assert "non-retryable" in (results[0].error or "")


# ──────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────


class TestEndpoint:
    def test_503_when_not_wired(self):
        node = _node(with_log=False)
        resp = _client(node).get("/admin/webhook-history")
        assert resp.status_code == 503

    def test_returns_recent_entries(self):
        node = _node()
        node._webhook_log.append(
            event="daemon.crashed", url="https://x.com",
            success=True, attempts=1, status_code=200,
        )
        resp = _client(node).get("/admin/webhook-history")
        body = resp.json()
        assert body["total"] == 1
        assert body["entries"][0]["event"] == "daemon.crashed"

    def test_pagination(self):
        node = _node()
        for i in range(5):
            node._webhook_log.append(
                event=f"e{i}", url="https://x.com",
                success=True, attempts=1,
            )
        resp = _client(node).get(
            "/admin/webhook-history?limit=2&offset=1"
        )
        body = resp.json()
        # Most-recent first, offset 1 skips e4 → start at e3
        assert [e["event"] for e in body["entries"]] == ["e3", "e2"]

    def test_invalid_limit_returns_422(self):
        node = _node()
        resp = _client(node).get("/admin/webhook-history?limit=0")
        assert resp.status_code == 422

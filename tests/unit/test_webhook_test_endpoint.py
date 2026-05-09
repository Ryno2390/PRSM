"""POST /admin/webhook-test — operator-side smoke test of webhook
delivery configuration. Synthesizes a webhook.test event +
dispatches to the configured URL so operators see success/failure
without waiting for a real daemon crash."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.webhook_delivery import DeliveryResult


def _node(*, with_webhook=True, success=True):
    node = MagicMock()
    node.identity.node_id = "test-node"
    if not with_webhook:
        node._webhook_deliverer = None
        node._daemon_watchdog = None
        return node

    deliverer = MagicMock()
    deliverer.deliver = AsyncMock(
        return_value=DeliveryResult(
            success=success,
            status_code=200 if success else 503,
            attempts=1 if success else 3,
            error=None if success else "simulated failure",
        ),
    )
    node._webhook_deliverer = deliverer
    # Watchdog holds the URL + secret config.
    watchdog = MagicMock()
    watchdog._webhook_url = "https://hook.example.com/in"
    watchdog._webhook_secret = "ops-secret"
    node._daemon_watchdog = watchdog
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


class TestWebhookTestAvailability:
    def test_503_when_webhook_not_configured(self):
        node = _node(with_webhook=False)
        resp = _client(node).post("/admin/webhook-test")
        assert resp.status_code == 503


class TestWebhookTestSuccess:
    def test_200_with_delivery_result_shape(self):
        node = _node(with_webhook=True, success=True)
        resp = _client(node).post("/admin/webhook-test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["status_code"] == 200
        assert body["attempts"] == 1

    def test_dispatches_to_configured_url(self):
        node = _node()
        _client(node).post("/admin/webhook-test")
        # Deliver was called with the watchdog's URL.
        node._webhook_deliverer.deliver.assert_awaited_once()
        call = node._webhook_deliverer.deliver.await_args
        assert call.kwargs["url"] == "https://hook.example.com/in"
        # event name is webhook.test (not daemon.crashed — this is
        # explicitly a smoke test).
        assert call.kwargs["event"] == "webhook.test"
        # Secret threaded.
        assert call.kwargs["secret"] == "ops-secret"


class TestWebhookTestFailure:
    def test_returns_200_with_success_false_on_delivery_failure(self):
        """The endpoint itself returns 200 even on delivery
        failure — the failure detail is in the response body
        so operator can triage. Returning 503 would conflate
        "endpoint broken" with "webhook delivery failed."""
        node = _node(success=False)
        resp = _client(node).post("/admin/webhook-test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["error"] == "simulated failure"
        assert body["attempts"] == 3

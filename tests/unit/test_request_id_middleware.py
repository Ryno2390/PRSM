"""X-Request-ID correlation middleware.

Production-debugging feature: every response carries an
X-Request-ID header. If the request supplies one (e.g., from
an upstream load balancer), the server echoes it back; if not,
the server generates a UUID. Threading through logs lets
operators correlate a complaint to specific log lines without
guessing on timestamp.
"""
from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._royalty_distributor_client = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# Server generates an ID when none provided
# ──────────────────────────────────────────────────────────────────────


class TestServerGenerated:
    def test_response_has_x_request_id_header(self):
        node = _node()
        resp = _client(node).get("/health")
        assert "x-request-id" in {k.lower() for k in resp.headers.keys()}

    def test_generated_id_is_uuid_shaped(self):
        node = _node()
        resp = _client(node).get("/health")
        rid = resp.headers["x-request-id"]
        # UUID4: 8-4-4-4-12 hex digits.
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            rid,
            flags=re.IGNORECASE,
        ) is not None

    def test_distinct_ids_across_requests(self):
        node = _node()
        client = _client(node)
        resp1 = client.get("/health")
        resp2 = client.get("/health")
        # Two separate requests get two distinct IDs.
        assert resp1.headers["x-request-id"] != resp2.headers["x-request-id"]


# ──────────────────────────────────────────────────────────────────────
# Client-supplied ID is echoed back
# ──────────────────────────────────────────────────────────────────────


class TestClientSupplied:
    def test_supplied_id_echoed_back(self):
        node = _node()
        custom_id = "client-correlation-abc-123"
        resp = _client(node).get(
            "/health",
            headers={"X-Request-ID": custom_id},
        )
        assert resp.headers["x-request-id"] == custom_id

    def test_lowercase_header_name_also_accepted(self):
        """HTTP headers are case-insensitive; FastAPI handles that
        but we verify the round-trip works."""
        node = _node()
        resp = _client(node).get(
            "/health",
            headers={"x-request-id": "lowercase-abc"},
        )
        assert resp.headers["x-request-id"] == "lowercase-abc"


# ──────────────────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_overly_long_supplied_id_truncated(self):
        """Defensive: cap supplied X-Request-ID at 128 chars to
        defend against log-poisoning via gigantic IDs."""
        node = _node()
        bomb = "A" * 10_000
        resp = _client(node).get(
            "/health",
            headers={"X-Request-ID": bomb},
        )
        rid = resp.headers["x-request-id"]
        assert len(rid) <= 128

    def test_empty_supplied_id_falls_back_to_generated(self):
        node = _node()
        resp = _client(node).get(
            "/health",
            headers={"X-Request-ID": ""},
        )
        rid = resp.headers["x-request-id"]
        # Empty string ignored; generated UUID present.
        assert rid != ""
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            rid,
            flags=re.IGNORECASE,
        ) is not None

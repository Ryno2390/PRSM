"""SlashEventRing + GET /admin/slash-history.

In-memory ring buffer of on-chain slash events observed by the
StorageSlashingWatcher. Two event flavors recorded:

  * proof_failure_slashed     — proof verification failed
  * heartbeat_missing_slashed — operator missed heartbeat window

Watcher callbacks (already wired in node.py) get a wrapped sink
that also appends to this ring. Block/tx info isn't on the
event-callback signature; operators correlate to on-chain via
slash_id_hex.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.slash_event_log import SlashEventRing


def _node(*, with_log=True):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._webhook_log = None
    node._slash_event_log = SlashEventRing() if with_log else None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


# ──────────────────────────────────────────────────────────────────────
# SlashEventRing primitive
# ──────────────────────────────────────────────────────────────────────


class TestRing:
    def test_append_and_recent(self):
        ring = SlashEventRing()
        ring.append(
            kind="proof_failure_slashed",
            provider="0xPROV",
            challenger="0xCHAL",
            slash_id=b"\x01" * 32,
            extras={"shard_id": "0xabcd"},
        )
        results = ring.recent()
        assert len(results) == 1
        assert results[0].kind == "proof_failure_slashed"
        assert results[0].provider == "0xPROV"
        assert results[0].challenger == "0xCHAL"
        assert results[0].slash_id_hex.startswith("0x")
        assert results[0].extras["shard_id"] == "0xabcd"

    def test_kind_enum_validated(self):
        ring = SlashEventRing()
        with pytest.raises(ValueError):
            ring.append(
                kind="not_a_real_kind",
                provider="0xX", challenger="0xY",
                slash_id=b"\x00" * 32,
            )

    def test_most_recent_first(self):
        ring = SlashEventRing()
        for i in range(3):
            ring.append(
                kind="heartbeat_missing_slashed",
                provider=f"0xPROV{i}", challenger="0xC",
                slash_id=bytes([i] * 32),
            )
        results = ring.recent()
        assert [e.provider for e in results] == [
            "0xPROV2", "0xPROV1", "0xPROV0",
        ]

    def test_bounded_by_max_entries(self):
        ring = SlashEventRing(max_entries=2)
        for i in range(5):
            ring.append(
                kind="proof_failure_slashed",
                provider=f"0xP{i}", challenger="0xC",
                slash_id=bytes([i] * 32),
            )
        results = ring.recent()
        assert len(results) == 2

    def test_filter_by_provider(self):
        ring = SlashEventRing()
        ring.append(
            kind="proof_failure_slashed",
            provider="0xMINE", challenger="0xC",
            slash_id=b"\x01" * 32,
        )
        ring.append(
            kind="proof_failure_slashed",
            provider="0xOTHER", challenger="0xC",
            slash_id=b"\x02" * 32,
        )
        results = ring.recent(provider="0xMINE")
        assert len(results) == 1
        assert results[0].provider == "0xMINE"


# ──────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────


class TestEndpoint:
    def test_503_when_not_wired(self):
        node = _node(with_log=False)
        resp = _client(node).get("/admin/slash-history")
        assert resp.status_code == 503

    def test_returns_recent_entries(self):
        node = _node()
        node._slash_event_log.append(
            kind="proof_failure_slashed",
            provider="0xPROV", challenger="0xCHAL",
            slash_id=b"\x01" * 32,
            extras={"shard_id": "0xabcd"},
        )
        resp = _client(node).get("/admin/slash-history")
        body = resp.json()
        assert body["total"] == 1
        assert body["entries"][0]["kind"] == "proof_failure_slashed"
        assert body["entries"][0]["challenger"] == "0xCHAL"
        assert body["entries"][0]["extras"]["shard_id"] == "0xabcd"

    def test_provider_filter(self):
        node = _node()
        node._slash_event_log.append(
            kind="proof_failure_slashed",
            provider="0xMINE", challenger="0xC",
            slash_id=b"\x01" * 32,
        )
        node._slash_event_log.append(
            kind="proof_failure_slashed",
            provider="0xOTHER", challenger="0xC",
            slash_id=b"\x02" * 32,
        )
        resp = _client(node).get(
            "/admin/slash-history?provider=0xMINE"
        )
        body = resp.json()
        assert len(body["entries"]) == 1
        assert body["entries"][0]["provider"] == "0xMINE"

    def test_invalid_limit_returns_422(self):
        node = _node()
        resp = _client(node).get("/admin/slash-history?limit=0")
        assert resp.status_code == 422

    def test_pagination(self):
        node = _node()
        for i in range(5):
            node._slash_event_log.append(
                kind="heartbeat_missing_slashed",
                provider=f"0xP{i}", challenger="0xC",
                slash_id=bytes([i] * 32),
            )
        resp = _client(node).get(
            "/admin/slash-history?limit=2&offset=1"
        )
        body = resp.json()
        assert len(body["entries"]) == 2
        # Most-recent first; offset=1 skips most-recent (0xP4)
        assert body["entries"][0]["provider"] == "0xP3"

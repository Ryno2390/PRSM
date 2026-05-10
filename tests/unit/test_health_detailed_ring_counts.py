"""Ring buffer counts surfaced in /health/detailed (sprint 94).

Operators see at a glance whether the 4 dashboard rings
(webhook + slash + heartbeat + distribution) are populated +
whether persistence is enabled.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.distribution_log import DistributedEventRing
from prsm.node.heartbeat_log import HeartbeatRecordedRing
from prsm.node.slash_event_log import SlashEventRing
from prsm.node.webhook_log import WebhookLogRing


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._payment_escrow = None
    node._job_history = None
    node._provenance_client = None
    node._royalty_distributor_client = None
    node._storage_slashing_client = None
    node._compensation_distributor_client = None
    node._key_distribution_client = None
    node._webhook_log = None
    node._slash_event_log = None
    node._heartbeat_log = None
    node._distribution_log = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def test_ring_counts_zero_when_empty():
    node = _node()
    node._slash_event_log = SlashEventRing()
    node._heartbeat_log = HeartbeatRecordedRing()
    node._distribution_log = DistributedEventRing()
    node._webhook_log = WebhookLogRing()
    resp = _client(node).get("/health/detailed")
    body = resp.json()
    for ring_name in (
        "webhook_log", "slash_event_log",
        "heartbeat_log", "distribution_log",
    ):
        sub = body["subsystems"][ring_name]
        assert sub["available"] is True
        assert sub["count"] == 0
        assert sub["persisted"] is False


def test_ring_count_reflects_appends():
    node = _node()
    node._slash_event_log = SlashEventRing()
    node._slash_event_log.append(
        kind="proof_failure_slashed",
        provider="0xP", challenger="0xC",
        slash_id=b"\x01" * 32,
    )
    node._slash_event_log.append(
        kind="heartbeat_missing_slashed",
        provider="0xP", challenger="0xC",
        slash_id=b"\x02" * 32,
    )
    resp = _client(node).get("/health/detailed")
    sub = resp.json()["subsystems"]["slash_event_log"]
    assert sub["count"] == 2


def test_persisted_flag_when_persist_dir_set(tmp_path: Path):
    node = _node()
    node._slash_event_log = SlashEventRing(persist_dir=tmp_path)
    resp = _client(node).get("/health/detailed")
    sub = resp.json()["subsystems"]["slash_event_log"]
    assert sub["persisted"] is True


def test_not_wired_when_none():
    node = _node()  # All rings None
    resp = _client(node).get("/health/detailed")
    body = resp.json()
    for ring_name in (
        "webhook_log", "slash_event_log",
        "heartbeat_log", "distribution_log",
    ):
        sub = body["subsystems"][ring_name]
        assert sub["available"] is False
        assert sub["status"] == "not_wired"

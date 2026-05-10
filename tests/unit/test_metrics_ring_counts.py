"""Ring buffer counts in /metrics Prometheus exposition (sprint 97).

4 new gauges:
  prsm_webhook_log_count
  prsm_slash_event_log_count
  prsm_heartbeat_log_count
  prsm_distribution_log_count
"""
from __future__ import annotations

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
    node._webhook_log = None
    node._slash_event_log = None
    node._heartbeat_log = None
    node._distribution_log = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def test_metrics_emits_zero_counts_for_empty_rings():
    node = _node()
    node._webhook_log = WebhookLogRing()
    node._slash_event_log = SlashEventRing()
    node._heartbeat_log = HeartbeatRecordedRing()
    node._distribution_log = DistributedEventRing()
    text = _client(node).get("/metrics").text
    assert "prsm_webhook_log_count 0" in text
    assert "prsm_slash_event_log_count 0" in text
    assert "prsm_heartbeat_log_count 0" in text
    assert "prsm_distribution_log_count 0" in text


def test_metrics_emits_actual_counts():
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
    node._distribution_log = DistributedEventRing()
    node._distribution_log.append(
        to_creator=100, to_operator=50, to_grant=25,
    )
    text = _client(node).get("/metrics").text
    assert "prsm_slash_event_log_count 2" in text
    assert "prsm_distribution_log_count 1" in text


def test_metrics_omits_unwired_rings():
    node = _node()  # All rings None
    text = _client(node).get("/metrics").text
    # Unwired rings produce NO gauge (not zero — they're absent)
    assert "prsm_webhook_log_count" not in text
    assert "prsm_slash_event_log_count" not in text
    assert "prsm_heartbeat_log_count" not in text
    assert "prsm_distribution_log_count" not in text


def test_help_and_type_emitted():
    node = _node()
    node._slash_event_log = SlashEventRing()
    text = _client(node).get("/metrics").text
    assert "# HELP prsm_slash_event_log_count" in text
    assert "# TYPE prsm_slash_event_log_count gauge" in text


def test_metrics_per_ring_isolation():
    """One ring's count() raising shouldn't take down others."""
    node = _node()
    bad_ring = MagicMock()
    bad_ring.count = MagicMock(
        side_effect=RuntimeError("simulated"),
    )
    node._webhook_log = bad_ring
    # Working ring still emits
    node._slash_event_log = SlashEventRing()
    text = _client(node).get("/metrics").text
    # webhook count omitted (bad ring) but slash count still emitted
    assert "prsm_webhook_log_count" not in text
    assert "prsm_slash_event_log_count 0" in text

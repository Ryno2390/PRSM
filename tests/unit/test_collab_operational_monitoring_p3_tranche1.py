"""P3 Tranche 1 tests for collaboration operational metrics and alert wiring."""

from datetime import datetime

import pytest

from prsm.core.monitoring.alerts import AlertManager
from prsm.core.monitoring.metrics import CollaborationTelemetryMetrics, MetricValue, MetricsCollector


class _StubComponent:
    def __init__(self, payload):
        self._payload = payload

    def get_telemetry_snapshot(self):
        return self._payload


@pytest.mark.asyncio
async def test_collab_telemetry_metric_bridge_exports_expected_signals() -> None:
    """Bridge should map existing bounded telemetry into canonical metric names."""
    transport = _StubComponent(
        {
            "handshake_success_total": 8,
            "handshake_failure_total": 2,
            "handshake_failure_reasons": {"replay_nonce": 2, "missing_public_key": 1},
            "dispatch_success_total": 18,
            "dispatch_failure_total": 2,
            "dispatch_failure_reasons": {"handler_exception": 2},
        }
    )
    gossip = _StubComponent(
        {
            "publish_total": 10,
            "forward_total": 7,
            "drop_total": 3,
            "drop_by_reason": {"ttl_exhausted": 2, "missing_subtype": 1},
        }
    )
    collab = _StubComponent(
        {
            "protocol_transition_total": 9,
            "terminal_outcome_total": 6,
        }
    )
    manager = _StubComponent(
        {
            "dispatch_success_total": 5,
            "dispatch_failure_total": 1,
        }
    )

    metric = CollaborationTelemetryMetrics(
        transport_provider=lambda: transport,
        gossip_provider=lambda: gossip,
        agent_collab_provider=lambda: collab,
        collab_manager_provider=lambda: manager,
    )

    values = await metric.collect()
    by_name = {}
    for value in values:
        by_name.setdefault(value.name, []).append(value)

    assert by_name["collab_transport_handshake_failure_total"][0].value == 2
    assert by_name["collab_transport_handshake_failure_rate"][0].value == pytest.approx(0.2)
    assert by_name["collab_transport_handshake_replay_nonce_total"][0].value == 2
    assert by_name["collab_transport_dispatch_failure_rate"][0].value == pytest.approx(0.1)
    assert by_name["collab_gossip_drop_rate"][0].value == pytest.approx(3 / 10)
    assert by_name["collab_protocol_stalled_total"][0].value == 3
    assert by_name["collab_manager_dispatch_failure_rate"][0].value == pytest.approx(1 / 6)

    labeled_handshake = by_name["collab_transport_handshake_failures_by_reason_total"]
    assert any(m.labels == {"reason": "replay_nonce"} and m.value == 2 for m in labeled_handshake)
    assert any(m.labels == {"reason": "missing_public_key"} and m.value == 1 for m in labeled_handshake)


@pytest.mark.asyncio
async def test_collab_telemetry_replay_delta_tracks_increment_only() -> None:
    """Replay nonce delta should reflect monotonic increases between collection cycles."""
    transport = _StubComponent(
        {
            "handshake_success_total": 1,
            "handshake_failure_total": 1,
            "handshake_failure_reasons": {"replay_nonce": 1},
            "dispatch_success_total": 1,
            "dispatch_failure_total": 0,
        }
    )

    metric = CollaborationTelemetryMetrics(transport_provider=lambda: transport)

    first = await metric.collect()
    first_delta = [m for m in first if m.name == "collab_transport_handshake_replay_nonce_delta"][0]
    assert first_delta.value == 1

    transport._payload["handshake_failure_reasons"]["replay_nonce"] = 4
    second = await metric.collect()
    second_delta = [m for m in second if m.name == "collab_transport_handshake_replay_nonce_delta"][0]
    assert second_delta.value == 3

    # No increase should produce 0 delta
    third = await metric.collect()
    third_delta = [m for m in third if m.name == "collab_transport_handshake_replay_nonce_delta"][0]
    assert third_delta.value == 0


def test_collaboration_alert_rules_registered_with_expected_metric_bindings() -> None:
    """Alert manager should provision dedicated collaboration/trust rules."""
    manager = AlertManager()
    manager.setup_collaboration_rules()

    expected = {
        "collab_handshake_failure_rate_high": "collab_transport_handshake_failure_rate",
        "collab_replay_nonce_spike": "collab_transport_handshake_replay_nonce_delta",
        "collab_dispatch_failure_rate_high": "collab_transport_dispatch_failure_rate",
        "collab_stalled_protocols_detected": "collab_protocol_stalled_total",
    }

    for rule_name, metric_name in expected.items():
        assert rule_name in manager.rules
        assert manager.rules[rule_name].condition.metric_name == metric_name


@pytest.mark.asyncio
async def test_metrics_collector_registers_collaboration_bridge() -> None:
    """Collector helper should register the collaboration telemetry metric cleanly."""
    collector = MetricsCollector(collection_interval=999)
    transport = _StubComponent({"handshake_success_total": 0, "handshake_failure_total": 0})

    metric = collector.register_collaboration_telemetry(transport_provider=lambda: transport)
    assert metric.name == "collaboration_telemetry"
    assert collector.registry.get_metric("collaboration_telemetry") is metric

    collected = await collector.registry.collect_all()
    assert any(isinstance(item, MetricValue) for item in collected)
    assert any(item.name == "collab_transport_handshake_failure_total" for item in collected)
    assert all(isinstance(item.timestamp, datetime) for item in collected)


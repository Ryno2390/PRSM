"""P3 Tranche 2 tests for collaboration telemetry export wiring and alert tuning."""

from pathlib import Path

import pytest

from prsm.core.monitoring.alerts import AlertManager
from prsm.core.monitoring.metrics import MetricsCollector, PROMETHEUS_AVAILABLE


class _StubComponent:
    def __init__(self, payload):
        self._payload = payload

    def get_telemetry_snapshot(self):
        return self._payload


@pytest.mark.asyncio
async def test_collaboration_metrics_are_exposed_through_collector_export_path() -> None:
    """Collector should bridge collab telemetry into exporter-visible metric state."""
    collector = MetricsCollector(collection_interval=999)
    transport = _StubComponent(
        {
            "handshake_success_total": 12,
            "handshake_failure_total": 3,
            "handshake_failure_reasons": {"replay_nonce": 3},
            "dispatch_success_total": 30,
            "dispatch_failure_total": 2,
            "dispatch_failure_reasons": {"handler_exception": 2},
        }
    )
    gossip = _StubComponent({"publish_total": 5, "forward_total": 3, "drop_total": 2})
    collab = _StubComponent({"protocol_transition_total": 10, "terminal_outcome_total": 8})
    manager = _StubComponent({"dispatch_success_total": 9, "dispatch_failure_total": 1})

    collector.register_collaboration_telemetry(
        transport_provider=lambda: transport,
        gossip_provider=lambda: gossip,
        agent_collab_provider=lambda: collab,
        collab_manager_provider=lambda: manager,
    )

    metrics = await collector.collect_once()
    names = {m.name for m in metrics}
    assert "collab_transport_handshake_failure_rate" in names
    assert "collab_manager_dispatch_failure_rate" in names
    assert "collab_protocol_stalled_ratio" in names

    if PROMETHEUS_AVAILABLE:
        exported = collector.get_prometheus_metrics()
        assert "collab_transport_handshake_failure_rate" in exported
        assert "collab_protocol_stalled_ratio" in exported


@pytest.mark.asyncio
async def test_collaboration_reason_label_cardinality_is_bounded() -> None:
    """Unrecognized reason labels should be collapsed into `other`."""
    collector = MetricsCollector(collection_interval=999)
    transport = _StubComponent(
        {
            "handshake_success_total": 1,
            "handshake_failure_total": 2,
            "handshake_failure_reasons": {
                "replay_nonce": 1,
                "totally_new_reason": 4,
            },
            "dispatch_success_total": 1,
            "dispatch_failure_total": 2,
            "dispatch_failure_reasons": {
                "timeout": 1,
                "unbounded_dispatch_reason": 7,
            },
        }
    )
    gossip = _StubComponent(
        {
            "publish_total": 1,
            "forward_total": 1,
            "drop_total": 2,
            "drop_by_reason": {"deduplicated": 1, "new_drop_reason": 5},
        }
    )

    collector.register_collaboration_telemetry(
        transport_provider=lambda: transport,
        gossip_provider=lambda: gossip,
    )

    metrics = await collector.collect_once()

    handshake_reasons = {
        m.labels.get("reason")
        for m in metrics
        if m.name == "collab_transport_handshake_failures_by_reason_total"
    }
    dispatch_reasons = {
        m.labels.get("reason")
        for m in metrics
        if m.name == "collab_transport_dispatch_failures_by_reason_total"
    }
    gossip_reasons = {
        m.labels.get("reason")
        for m in metrics
        if m.name == "collab_gossip_drop_by_reason_total"
    }

    assert handshake_reasons == {"replay_nonce", "other"}
    assert dispatch_reasons == {"timeout", "other"}
    assert gossip_reasons == {"deduplicated", "other"}


def test_collaboration_alert_rule_thresholds_are_tuned_conservatively() -> None:
    """Collaboration alert rules should use conservative practical defaults."""
    manager = AlertManager()
    manager.setup_collaboration_rules()

    assert manager.rules["collab_handshake_failure_rate_high"].condition.threshold == pytest.approx(0.15)
    assert manager.rules["collab_handshake_failure_rate_high"].condition.duration.total_seconds() == 300
    assert manager.rules["collab_replay_nonce_spike"].condition.threshold == 5
    assert manager.rules["collab_replay_nonce_spike"].condition.duration.total_seconds() == 120
    assert manager.rules["collab_dispatch_failure_rate_high"].condition.threshold == pytest.approx(0.10)
    assert manager.rules["collab_manager_dispatch_failure_rate_high"].condition.threshold == pytest.approx(0.08)
    assert manager.rules["collab_stalled_protocols_detected"].condition.threshold == 2
    assert manager.rules["collab_stalled_protocol_ratio_high"].condition.threshold == pytest.approx(0.25)


def test_monitoring_configs_include_collaboration_series_for_dashboards() -> None:
    """Prometheus/exporter config should include collaboration-compatible surfaces."""
    exporter_config = Path("config/metrics-exporter.yml").read_text(encoding="utf-8")
    assert "collaboration: true" in exporter_config

    recording_rules = Path("config/prometheus/recording_rules.yml").read_text(encoding="utf-8")
    assert "- name: prsm.collaboration.recording" in recording_rules
    assert "record: prsm:collab_handshake_failure_rate_5m" in recording_rules
    assert "expr: avg_over_time(collab_transport_handshake_failure_rate[5m])" in recording_rules
    assert "record: prsm:collab_replay_nonce_delta_max_5m" in recording_rules
    assert "expr: max_over_time(collab_transport_handshake_replay_nonce_delta[5m])" in recording_rules
    assert "record: prsm:collab_transport_dispatch_failure_rate_5m" in recording_rules
    assert "record: prsm:collab_manager_dispatch_failure_rate_5m" in recording_rules
    assert "record: prsm:collab_protocol_stalled_ratio_5m" in recording_rules


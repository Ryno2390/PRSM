"""Sprint 389 — bootstrap server /metrics Prometheus content-negotiation.

Two distinct gaps in pre-sprint-389 `prsm/bootstrap/server.py`:

1. Path-naming convention mismatch — `/metrics` returns JSON,
   `/prometheus` returns Prometheus exposition format. The
   universal Prometheus convention is `/metrics` with text/plain.
   So a default scrape config pointed at this bootstrap server
   gets back JSON that the Prometheus parser silently rejects.
   Operator runs a deployment, points Grafana at it, sees zero
   panels — silent observability failure.

2. Incomplete metric coverage — `BootstrapMetrics.to_dict()`
   exposes 15 fields (incl. 2 dicts: peers_by_region and
   peers_by_capability). The Prometheus surface only renders 8
   of those flat counters, and zero of the labeled-gauge dict
   fields. Operators tracking regional peer distribution or
   rate-limit rejections can't query them.

This sprint:
- Adds Accept-header content negotiation on `/metrics` —
  text/plain or openmetrics-text → Prometheus; default → JSON
  (backwards-compat for any operator script polling `/metrics`
  as JSON today).
- Keeps `/prometheus` as an explicit-always-Prometheus alias.
- Surfaces the 6 missing flat metrics
  (rejected_connections, bytes_sent, bytes_received,
  avg_response_time_ms) AND the 2 dict-fields as labeled
  gauges (prsm_bootstrap_peers_by_region{region="..."},
  prsm_bootstrap_peers_by_capability{capability="..."}).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

# fastapi.testclient lives in starlette; both ship with fastapi
fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

from prsm.bootstrap.config import BootstrapConfig
from prsm.bootstrap.server import BootstrapServer


@pytest.fixture
def server() -> BootstrapServer:
    cfg = BootstrapConfig(
        domain="test.prsm-network.com",
        host="127.0.0.1",
        port=8765,
        api_port=8000,
        ssl_enabled=False,
        max_peers=100,
        persist_peers=False,
        metrics_enabled=True,
        log_level="DEBUG",
        region="test-region",
    )
    return BootstrapServer(cfg)


@pytest.fixture
def client(server: BootstrapServer) -> TestClient:
    """Construct app via the new _build_api_app testable seam."""
    app = server._build_api_app()
    return TestClient(app)


# ── Path-naming convention: /metrics with Prometheus Accept ──


class TestMetricsContentNegotiation:
    def test_metrics_plain_text_returns_prometheus(
        self, server, client,
    ):
        server.metrics.total_connections = 42
        resp = client.get("/metrics", headers={"Accept": "text/plain"})
        assert resp.status_code == 200
        # Prometheus exposition lines are # HELP / # TYPE / metric
        assert "# HELP prsm_bootstrap_total_connections" in resp.text
        assert "prsm_bootstrap_total_connections 42" in resp.text
        # Content-Type carries the canonical Prometheus mime
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_openmetrics_accept_returns_prometheus(
        self, server, client,
    ):
        """Prometheus scrape default `Accept` header is
        `application/openmetrics-text;...,text/plain;...` —
        the openmetrics variant should also trigger
        Prometheus exposition."""
        server.metrics.active_connections = 7
        resp = client.get(
            "/metrics",
            headers={
                "Accept": (
                    "application/openmetrics-text;"
                    "version=1.0.0;charset=utf-8,"
                    "text/plain;version=0.0.4;q=0.5,*/*;q=0.1"
                ),
            },
        )
        assert resp.status_code == 200
        assert "prsm_bootstrap_active_connections 7" in resp.text

    def test_metrics_json_accept_returns_json(
        self, server, client,
    ):
        """Backwards-compat: anyone explicitly asking for
        JSON gets JSON."""
        server.metrics.total_connections = 11
        resp = client.get(
            "/metrics", headers={"Accept": "application/json"},
        )
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        data = resp.json()
        assert data["total_connections"] == 11

    def test_metrics_no_accept_header_returns_json(
        self, server, client,
    ):
        """Backwards-compat: no Accept header → JSON
        (pre-sprint-389 default behavior preserved)."""
        server.metrics.total_connections = 99
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        assert resp.json()["total_connections"] == 99


# ── /prometheus alias remains for explicit consumers ─────


class TestPrometheusAlias:
    def test_prometheus_endpoint_always_returns_prometheus(
        self, server, client,
    ):
        """The explicit `/prometheus` path is unchanged —
        backwards-compat for any operator who already wired
        their scrape config to it."""
        server.metrics.total_connections = 3
        resp = client.get("/prometheus")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert "prsm_bootstrap_total_connections 3" in resp.text


# ── Missing flat metrics now surfaced ────────────────────


class TestExpandedFlatMetrics:
    def test_rejected_connections_exposed(self, server, client):
        server.metrics.rejected_connections = 5
        resp = client.get("/prometheus")
        assert "prsm_bootstrap_rejected_connections 5" in resp.text
        # And it's typed as counter (rate-limit + ban hits accumulate)
        assert (
            "# TYPE prsm_bootstrap_rejected_connections counter"
            in resp.text
        )

    def test_bytes_sent_and_received_exposed(
        self, server, client,
    ):
        server.metrics.bytes_sent = 12345
        server.metrics.bytes_received = 67890
        resp = client.get("/prometheus")
        assert "prsm_bootstrap_bytes_sent 12345" in resp.text
        assert "prsm_bootstrap_bytes_received 67890" in resp.text

    def test_avg_response_time_ms_exposed(self, server, client):
        server.metrics.avg_response_time_ms = 42.5
        resp = client.get("/prometheus")
        assert "prsm_bootstrap_avg_response_time_ms 42.5" in resp.text


# ── Labeled-gauge surface for dict-shaped fields ─────────


class TestLabeledGauges:
    def test_peers_by_region_labeled_gauge(
        self, server, client,
    ):
        server.metrics.peers_by_region = {
            "us-east-1": 3,
            "eu-west-1": 7,
            "ap-northeast-1": 2,
        }
        resp = client.get("/prometheus")
        assert (
            'prsm_bootstrap_peers_by_region{region="us-east-1"} 3'
            in resp.text
        )
        assert (
            'prsm_bootstrap_peers_by_region{region="eu-west-1"} 7'
            in resp.text
        )
        assert (
            'prsm_bootstrap_peers_by_region{region="ap-northeast-1"} 2'
            in resp.text
        )
        # Single HELP + TYPE for the metric family — not per label
        assert resp.text.count("# HELP prsm_bootstrap_peers_by_region") == 1

    def test_peers_by_capability_labeled_gauge(
        self, server, client,
    ):
        server.metrics.peers_by_capability = {
            "compute": 8,
            "storage": 4,
        }
        resp = client.get("/prometheus")
        assert (
            'prsm_bootstrap_peers_by_capability{capability="compute"} 8'
            in resp.text
        )
        assert (
            'prsm_bootstrap_peers_by_capability{capability="storage"} 4'
            in resp.text
        )

    def test_empty_label_dict_emits_no_lines(
        self, server, client,
    ):
        """No-peers state: the labeled-gauge family should
        be absent entirely (canonical Prometheus pattern —
        absence = no current value)."""
        server.metrics.peers_by_region = {}
        server.metrics.peers_by_capability = {}
        resp = client.get("/prometheus")
        assert "prsm_bootstrap_peers_by_region{" not in resp.text
        assert "prsm_bootstrap_peers_by_capability{" not in resp.text

    def test_label_value_escapes_quotes_and_backslash(
        self, server, client,
    ):
        """Defensive label-value escaping — matches the
        sprint-377 operator-node convention."""
        server.metrics.peers_by_region = {
            'weird"region': 1,
            'back\\slash': 2,
        }
        resp = client.get("/prometheus")
        # " escaped to \"
        assert (
            'prsm_bootstrap_peers_by_region{region="weird\\"region"} 1'
            in resp.text
        )
        # \ escaped to \\
        assert (
            'prsm_bootstrap_peers_by_region{region="back\\\\slash"} 2'
            in resp.text
        )


# ── Pre-existing endpoints still work ────────────────────


class TestSubsystemPrometheusGauges:
    """Sprint 394 — /health/detailed's per-subsystem state
    surfaces in the Prometheus exposition as two labeled
    gauges so operators can alert on specific stuck loops.
    """

    def test_subsystem_status_gauge_emitted(
        self, server, client,
    ):
        from datetime import datetime, timezone
        # Force every subsystem to a known state by bumping
        # the heartbeat registry directly
        now = datetime.now(timezone.utc)
        server._loop_heartbeats = {
            "peer_cleanup": now,
            "peer_backup": now,
            "federation_sync": now,
            "health_check_loop": now,
        }
        resp = client.get("/prometheus")
        text = resp.text
        # Encoding: 0 = healthy, 1 = degraded, 2 = stale
        for sub in (
            "peer_cleanup", "peer_backup",
            "federation_sync", "health_check_loop",
            "api_server",
        ):
            assert (
                f'prsm_bootstrap_subsystem_status'
                f'{{subsystem="{sub}"}} 0'
            ) in text, (
                f"missing healthy gauge for {sub} in: "
                f"{text[:500]}..."
            )

    def test_subsystem_heartbeat_age_gauge_emitted(
        self, server, client,
    ):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        server._loop_heartbeats = {
            "peer_cleanup": now - timedelta(seconds=10),
            "peer_backup": now,
            "federation_sync": now,
            "health_check_loop": now,
        }
        resp = client.get("/prometheus")
        text = resp.text
        # The age gauge for peer_cleanup should be ≥ 10
        # Find the line + parse the value
        prefix = (
            'prsm_bootstrap_subsystem_heartbeat_age_seconds'
            '{subsystem="peer_cleanup"} '
        )
        line = next(
            (l for l in text.splitlines() if l.startswith(prefix)),
            None,
        )
        assert line is not None, (
            f"missing peer_cleanup heartbeat-age gauge: "
            f"{text[:500]}..."
        )
        value = float(line.split(" ", 1)[1])
        assert value >= 10.0

    def test_subsystem_status_stale_encoded_as_2(
        self, server, client,
    ):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        # peer_cleanup interval = peer_timeout (default 300s).
        # 1800s old = 6× interval → stale.
        server._loop_heartbeats = {
            "peer_cleanup": now - timedelta(seconds=1800),
            "peer_backup": now,
            "federation_sync": now,
            "health_check_loop": now,
        }
        resp = client.get("/prometheus")
        text = resp.text
        assert (
            'prsm_bootstrap_subsystem_status'
            '{subsystem="peer_cleanup"} 2'
        ) in text

    def test_disabled_subsystem_encoded_as_1_not_2(
        self, server, client,
    ):
        """Sprint 397 — federation_sync with empty
        federation_peers reports status='disabled' (opt-out,
        not silent death). Prometheus encoding for opt-out
        is 1, matching sprint-395 operator-node-side
        convention. Encoding it as 2 would fire stale-loop
        alerts for default standalone bootstraps."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        server._loop_heartbeats = {
            "peer_cleanup": now,
            "peer_backup": now,
            "health_check_loop": now,
        }
        server.config.federation_peers = []
        resp = client.get("/prometheus")
        assert (
            'prsm_bootstrap_subsystem_status'
            '{subsystem="federation_sync"} 1'
        ) in resp.text

    def test_subsystem_help_and_type_lines_present(
        self, server, client,
    ):
        from datetime import datetime, timezone
        server._loop_heartbeats = {
            "peer_cleanup": datetime.now(timezone.utc),
        }
        resp = client.get("/prometheus")
        text = resp.text
        # HELP + TYPE for the metric families
        assert (
            "# HELP prsm_bootstrap_subsystem_status"
            in text
        )
        assert (
            "# TYPE prsm_bootstrap_subsystem_status gauge"
            in text
        )
        assert (
            "# HELP prsm_bootstrap_subsystem_heartbeat_age_seconds"
            in text
        )


class TestPreExistingEndpointsPreserved:
    def test_health_endpoint_still_works(self, server, client):
        resp = client.get("/health")
        # Bootstrap server health_check is async; FastAPI awaits
        assert resp.status_code == 200

    def test_peers_endpoint_still_works(self, server, client):
        resp = client.get("/peers")
        assert resp.status_code == 200
        assert "peers" in resp.json()

    def test_config_endpoint_still_works(self, server, client):
        resp = client.get("/config")
        assert resp.status_code == 200
        # The bootstrap config dict carries the domain
        assert (
            resp.json().get("domain") == "test.prsm-network.com"
        )


# ── Build-app seam ───────────────────────────────────────


class TestBuildApiAppSeam:
    def test_build_api_app_returns_fastapi_app(self, server):
        """Sprint 389 introduced `_build_api_app` as the
        explicit testable seam separate from
        `_run_api_server`, which is an io-loop entrypoint."""
        app = server._build_api_app()
        assert app is not None
        # FastAPI apps expose `.routes`
        assert hasattr(app, "routes")

    def test_app_exposes_metrics_and_prometheus_routes(
        self, server,
    ):
        app = server._build_api_app()
        paths = {r.path for r in app.routes if hasattr(r, "path")}
        assert "/metrics" in paths
        assert "/prometheus" in paths
        assert "/health" in paths
        assert "/peers" in paths
        assert "/config" in paths

"""Sprint 392 — bootstrap server /health/detailed
per-subsystem readiness probe.

The bootstrap server has 4 background loops that can die
silently without affecting /health: peer_cleanup,
peer_backup, federation_sync, health_check_loop. Pre-sprint
392, if any of these loops crashed or stalled, /health
still returned "healthy" because /health only checks
self.running, which is the top-level lifecycle flag.

This sprint adds:
  - `self._loop_heartbeats: Dict[str, datetime]` —
    last-successful-iteration timestamp per loop
  - Each loop bumps its key after a successful iteration
  - `health_check_detailed()` aggregates per-subsystem
    state and returns {subsystem_name: {alive,
    last_heartbeat_age_seconds, status, expected_interval}}
  - New `/health/detailed` HTTP route consuming it

Subsystem status thresholds (relative to expected loop
interval):
  - heartbeat_age < 2 × interval → healthy
  - 2 ≤ age < 5 × interval        → degraded
  - age ≥ 5 × interval OR missing → stale
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

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
        peer_timeout=60,
        heartbeat_interval=10,
        health_check_interval=30,
        peer_db_backup_interval=300,
        federation_sync_interval=900,
        persist_peers=False,
        metrics_enabled=True,
        log_level="DEBUG",
    )
    return BootstrapServer(cfg)


@pytest.fixture
def client(server) -> TestClient:
    return TestClient(server._build_api_app())


# ── Initial state — loop heartbeats registry ─────────────


def test_loop_heartbeats_initialized_on_construction(server):
    """Every BootstrapServer instance has the registry."""
    assert hasattr(server, "_loop_heartbeats")
    assert isinstance(server._loop_heartbeats, dict)


def test_heartbeat_recorder_method_exists(server):
    """Helper that loops call after a successful iteration."""
    assert hasattr(server, "_record_loop_heartbeat")
    # Method is callable + accepts a string label
    server._record_loop_heartbeat("peer_cleanup")
    assert "peer_cleanup" in server._loop_heartbeats
    # Stored value is a datetime
    assert isinstance(
        server._loop_heartbeats["peer_cleanup"],
        datetime,
    )


# ── /health/detailed route + canonical subsystem coverage ─


def test_health_detailed_route_exists(server, client):
    resp = client.get("/health/detailed")
    assert resp.status_code == 200


def test_health_detailed_reports_all_canonical_subsystems(
    server, client,
):
    """The four background loops + the API server itself
    all surface in /health/detailed."""
    resp = client.get("/health/detailed")
    body = resp.json()
    subs = body.get("subsystems") or {}
    for name in (
        "peer_cleanup",
        "peer_backup",
        "federation_sync",
        "health_check_loop",
        "api_server",
    ):
        assert name in subs, (
            f"missing subsystem {name!r}; got {list(subs.keys())}"
        )


def test_health_detailed_includes_aggregate_status(
    server, client,
):
    resp = client.get("/health/detailed")
    body = resp.json()
    assert "status" in body
    assert body["status"] in ("healthy", "degraded", "unhealthy")


# ── Subsystem status math ────────────────────────────────


def test_subsystem_never_heartbeated_is_stale(server, client):
    """A loop that hasn't ticked yet (server just started)
    surfaces as stale — not falsely healthy."""
    server._loop_heartbeats.clear()
    resp = client.get("/health/detailed")
    sub = resp.json()["subsystems"]["peer_cleanup"]
    assert sub["status"] == "stale"
    assert sub["last_heartbeat_age_seconds"] is None
    assert sub["alive"] is False


def test_subsystem_recent_heartbeat_is_healthy(
    server, client,
):
    """A loop that ticked moments ago is healthy."""
    server._loop_heartbeats["peer_cleanup"] = (
        datetime.now(timezone.utc)
    )
    resp = client.get("/health/detailed")
    sub = resp.json()["subsystems"]["peer_cleanup"]
    assert sub["status"] == "healthy"
    assert sub["alive"] is True
    assert sub["last_heartbeat_age_seconds"] is not None
    assert sub["last_heartbeat_age_seconds"] < 5.0


def test_subsystem_2x_interval_is_degraded(server, client):
    """Heartbeat age between 2× and 5× expected interval =
    degraded. peer_cleanup runs every peer_timeout (60s) →
    age 150s (2.5× interval) is degraded, not stale."""
    server._loop_heartbeats["peer_cleanup"] = (
        datetime.now(timezone.utc) - timedelta(seconds=150)
    )
    resp = client.get("/health/detailed")
    sub = resp.json()["subsystems"]["peer_cleanup"]
    assert sub["status"] == "degraded"


def test_subsystem_5x_interval_is_stale(server, client):
    """Age > 5× expected interval = stale.
    peer_cleanup interval=60s → 350s is stale."""
    server._loop_heartbeats["peer_cleanup"] = (
        datetime.now(timezone.utc) - timedelta(seconds=350)
    )
    resp = client.get("/health/detailed")
    sub = resp.json()["subsystems"]["peer_cleanup"]
    assert sub["status"] == "stale"


# ── Aggregate status reflects worst subsystem ────────────


def test_aggregate_healthy_when_all_subsystems_healthy(
    server, client,
):
    now = datetime.now(timezone.utc)
    for name in (
        "peer_cleanup", "peer_backup", "federation_sync",
        "health_check_loop",
    ):
        server._loop_heartbeats[name] = now
    resp = client.get("/health/detailed")
    body = resp.json()
    assert body["status"] == "healthy"


def test_aggregate_degraded_when_one_subsystem_degraded(
    server, client,
):
    now = datetime.now(timezone.utc)
    for name in (
        "peer_cleanup", "peer_backup", "federation_sync",
        "health_check_loop",
    ):
        server._loop_heartbeats[name] = now
    # Force peer_backup to degraded (interval=300s, 700s old → 2.3×)
    server._loop_heartbeats["peer_backup"] = (
        now - timedelta(seconds=700)
    )
    resp = client.get("/health/detailed")
    body = resp.json()
    assert body["status"] == "degraded"


def test_aggregate_unhealthy_when_one_subsystem_stale(
    server, client,
):
    now = datetime.now(timezone.utc)
    for name in (
        "peer_cleanup", "peer_backup", "federation_sync",
        "health_check_loop",
    ):
        server._loop_heartbeats[name] = now
    # Force federation_sync to stale (interval=900s, 5000s old → 5.5×)
    server._loop_heartbeats["federation_sync"] = (
        now - timedelta(seconds=5000)
    )
    resp = client.get("/health/detailed")
    body = resp.json()
    assert body["status"] == "unhealthy"


# ── api_server semantics (always healthy if we're answering) ─


def test_api_server_subsystem_always_healthy(server, client):
    """If /health/detailed returns a 200, by definition the
    API server is alive — so api_server subsystem reads
    healthy regardless of the heartbeats dict."""
    server._loop_heartbeats.clear()
    resp = client.get("/health/detailed")
    sub = resp.json()["subsystems"]["api_server"]
    assert sub["status"] == "healthy"
    assert sub["alive"] is True


# ── Pre-existing /health backwards-compat preserved ──────


def test_health_endpoint_unchanged(server, client):
    """The flat /health endpoint still returns the
    pre-sprint-392 shape. /health/detailed is additive."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    # Pre-sprint-392 fields all still present
    for key in (
        "status", "uptime_seconds", "active_connections",
        "total_peers", "total_connections",
        "failed_connections", "messages_processed",
        "region", "version", "server_time",
    ):
        assert key in body

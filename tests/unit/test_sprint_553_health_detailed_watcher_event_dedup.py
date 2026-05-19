"""Sprint 553 — wire watcher_event_dedup into /health/detailed.

Sprint 552 added an /admin endpoint exposing per-watcher dedup
state. Sprint 553 wires the same primitive into /health/detailed,
the canonical operator monitoring surface, so monitoring/alerting
workflows see whether the sprint-549/550/551 trifecta is wired
without having to discover the /admin endpoint.

Pin tests assert:
  - When dedup store is wired (operator opted in via
    PRSM_WATCHER_STATE_PERSISTENCE_ENABLED), subsystem appears
    with status:ok + summary fields.
  - When not wired, subsystem appears with status:not_wired +
    actionable env-var hint.
  - Summary computation errors degrade to status:error without
    breaking the rest of /health/detailed (fail-soft contract).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def _stub_node(dedup_store=None):
    n = MagicMock()
    n.identity = MagicMock(node_id="stub")
    n._watcher_event_dedup_store = dedup_store
    # Other subsystem hooks default to None via MagicMock — the
    # health endpoint walks each one defensively.
    return n


def _make_app(dedup_store=None):
    from prsm.node.api import create_api_app
    return create_api_app(
        _stub_node(dedup_store=dedup_store), enable_security=False,
    )


def test_health_detailed_includes_watcher_event_dedup_when_wired(
    tmp_path,
):
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))
    store.mark_processed_event(
        "compensation_distributor", "0x" + "aa" * 32, 0,
    )

    app = _make_app(dedup_store=store)
    client = TestClient(app)
    response = client.get("/health/detailed")
    assert response.status_code == 200
    body = response.json()
    assert "subsystems" in body, body
    assert "watcher_event_dedup" in body["subsystems"], (
        f"missing watcher_event_dedup; "
        f"keys={list(body['subsystems'].keys())}"
    )
    entry = body["subsystems"]["watcher_event_dedup"]
    assert entry["available"] is True
    assert entry["status"] == "ok"
    # Surfaces the summary fields shape (rows count + per-watcher
    # rollup keyed by watcher_key).
    assert entry["total_rows_processed"] == 1
    assert "watchers" in entry
    assert (
        entry["watchers"]["compensation_distributor"][
            "rows_processed"
        ] == 1
    )


def test_health_detailed_watcher_event_dedup_not_wired():
    """No dedup store on the node → subsystem reports
    not_wired with PRSM_WATCHER_STATE_PERSISTENCE_ENABLED hint."""
    app = _make_app(dedup_store=None)
    client = TestClient(app)
    response = client.get("/health/detailed")
    assert response.status_code == 200
    body = response.json()
    assert "watcher_event_dedup" in body["subsystems"], body
    entry = body["subsystems"]["watcher_event_dedup"]
    assert entry["available"] is False
    assert entry["status"] == "not_wired"
    assert (
        "PRSM_WATCHER_STATE_PERSISTENCE_ENABLED" in entry.get(
            "hint", "",
        )
    )


def test_health_detailed_watcher_event_dedup_summary_error_failsoft(
    tmp_path,
):
    """If summary() raises, the entry surfaces status:error but
    the rest of /health/detailed still works (200 overall)."""
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )

    class _ExplodingStore(EventDedupStore):
        def summary(self):
            raise RuntimeError("disk corrupted")

    store = _ExplodingStore(str(tmp_path / "dedup.db"))
    app = _make_app(dedup_store=store)
    client = TestClient(app)
    response = client.get("/health/detailed")
    assert response.status_code == 200
    entry = response.json()["subsystems"]["watcher_event_dedup"]
    assert entry["available"] is False
    assert entry["status"] == "error"
    assert "error" in entry
    assert "disk corrupted" in entry["error"]

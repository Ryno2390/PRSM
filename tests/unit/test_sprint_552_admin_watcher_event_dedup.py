"""Sprint 552 — operator visibility for sprint-549/550/551 watcher dedup.

The audit-trifecta closed the double-dispatch bug at the watcher
level, but operators have no live way to confirm the dedup is
actually firing on their node. Sprint 552 adds:

  - ``EventDedupStore.summary()`` — per-watcher rollup
    ``{watcher_key: {rows_processed, latest_tx_hash, latest_log_index}}``.

  - ``GET /admin/watcher-event-dedup`` — surfaces the summary.
    503 if the store isn't wired (operator hasn't opted into
    PRSM_WATCHER_STATE_PERSISTENCE_ENABLED); 200 with body otherwise.

Pure read-only audit surface — no mutations. Operators use it to
verify their watcher catch-up logic is working post-restart.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ── store summary API ─────────────────────────────────────


def test_summary_empty_store_returns_empty_dict(tmp_path):
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))
    assert store.summary() == {}


def test_summary_one_watcher_one_event(tmp_path):
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))
    store.mark_processed_event(
        "compensation_distributor", "0x" + "aa" * 32, 3,
    )

    summary = store.summary()
    assert "compensation_distributor" in summary
    s = summary["compensation_distributor"]
    assert s["rows_processed"] == 1
    assert s["latest_tx_hash"] == "0x" + "aa" * 32
    assert s["latest_log_index"] == 3


def test_summary_multiple_watchers_multiple_events(tmp_path):
    """All 3 watcher_keys from the sprint-549/550/551 trifecta
    surface with independent rollups."""
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))

    # CompensationDistributor: 2 events.
    store.mark_processed_event(
        "compensation_distributor", "0x" + "01" * 32, 0,
    )
    store.mark_processed_event(
        "compensation_distributor", "0x" + "02" * 32, 1,
    )

    # KeyDistribution: 1 event.
    store.mark_processed_event(
        "key_distribution", "0x" + "03" * 32, 0,
    )

    # StorageSlashing: 3 events.
    store.mark_processed_event(
        "storage_slashing", "0x" + "04" * 32, 0,
    )
    store.mark_processed_event(
        "storage_slashing", "0x" + "05" * 32, 0,
    )
    store.mark_processed_event(
        "storage_slashing", "0x" + "06" * 32, 2,
    )

    summary = store.summary()
    assert summary["compensation_distributor"]["rows_processed"] == 2
    assert summary["key_distribution"]["rows_processed"] == 1
    assert summary["storage_slashing"]["rows_processed"] == 3


def test_summary_latest_tracks_most_recently_marked(tmp_path):
    """``latest_tx_hash`` / ``latest_log_index`` reflect the
    most-recently-marked event per watcher (insertion order)."""
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))

    store.mark_processed_event(
        "compensation_distributor", "0x" + "01" * 32, 0,
    )
    store.mark_processed_event(
        "compensation_distributor", "0x" + "02" * 32, 5,
    )

    s = store.summary()["compensation_distributor"]
    assert s["latest_tx_hash"] == "0x" + "02" * 32
    assert s["latest_log_index"] == 5


# ── /admin/watcher-event-dedup endpoint ────────────────────


def _stub_node(dedup_store=None):
    n = MagicMock()
    n.identity = MagicMock(node_id="stub")
    n._watcher_event_dedup_store = dedup_store
    return n


def _make_app(dedup_store=None):
    from prsm.node.api import create_api_app
    return create_api_app(
        _stub_node(dedup_store=dedup_store), enable_security=False,
    )


def test_admin_endpoint_503_when_store_not_wired():
    """No store on the node → 503 with operator-actionable detail
    pointing at the env var that enables persistence."""
    app = _make_app(dedup_store=None)
    client = TestClient(app)
    response = client.get("/admin/watcher-event-dedup")
    assert response.status_code == 503
    body = response.json()["detail"]
    assert "PRSM_WATCHER_STATE_PERSISTENCE_ENABLED" in body, (
        f"503 should point operators at the env var; body={body!r}"
    )


def test_admin_endpoint_200_returns_summary(tmp_path):
    """Wired store → 200 with summary body (empty when no events
    processed yet)."""
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))
    app = _make_app(dedup_store=store)
    client = TestClient(app)
    response = client.get("/admin/watcher-event-dedup")
    assert response.status_code == 200
    body = response.json()
    assert "watchers" in body
    assert body["watchers"] == {}


def test_admin_endpoint_reflects_marked_events(tmp_path):
    """Mark events → endpoint returns them in the summary."""
    from prsm.economy.web3.last_processed_block_store import (
        EventDedupStore,
    )
    store = EventDedupStore(str(tmp_path / "dedup.db"))
    store.mark_processed_event(
        "compensation_distributor", "0x" + "aa" * 32, 0,
    )
    store.mark_processed_event(
        "key_distribution", "0x" + "bb" * 32, 2,
    )

    app = _make_app(dedup_store=store)
    client = TestClient(app)
    response = client.get("/admin/watcher-event-dedup")
    assert response.status_code == 200
    body = response.json()
    assert (
        body["watchers"]["compensation_distributor"]["rows_processed"]
        == 1
    )
    assert (
        body["watchers"]["compensation_distributor"]["latest_tx_hash"]
        == "0x" + "aa" * 32
    )
    assert (
        body["watchers"]["key_distribution"]["latest_log_index"] == 2
    )

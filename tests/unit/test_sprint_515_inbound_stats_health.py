"""Sprint 515 — inbound stats endpoint + /health/detailed.inbound_monitor.

The sprint 512/513/514 inbound surfaces give per-event visibility.
Sprint 515 adds aggregate views:

  1. GET /wallet/transactions/onchain/inbound/stats —
     {recipient, count, total_inbound_ftns, first_inbound_at,
      last_inbound_at, from_block, to_block}
  2. /health/detailed.subsystems.inbound_monitor —
     {available, status, last_scanned_block,
      total_inbound_in_session, last_inbound_at}

Both surfaces are pull-only and meant for monitoring/auditing
tools, complementing sprint 514's push signal.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def test_inbound_stats_503_when_ledger_missing():
    from prsm.node.api import create_api_app
    node = MagicMock()
    node.ftns_ledger = None
    app = create_api_app(node, enable_security=False)
    client = TestClient(app)
    r = client.get(
        "/wallet/transactions/onchain/inbound/stats"
    )
    assert r.status_code == 503


def test_inbound_stats_aggregates():
    """Mocked scan returning 3 events → stats with sum +
    first/last timestamps."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    ledger = MagicMock()
    ledger._connected_address = "0x" + "a" * 40
    ledger.w3 = MagicMock()
    ledger.w3.eth.block_number = 100000
    ledger._token = MagicMock()
    # Mock get_logs returns 3 events
    logs = []
    for i, (blk, amt) in enumerate(
        [(99000, 0.5), (99500, 1.5), (99950, 2.0)],
    ):
        log = MagicMock()
        log.blockNumber = blk
        log.transactionHash = bytes([i] * 32)
        log.args.__getitem__ = lambda self, k, _i=i, _a=amt: {
            "from": "0x" + str(_i) * 40,
            "to": "0x" + "a" * 40,
            "value": int(_a * 10**18),
        }[k]
        logs.append(log)
    ledger._token.events.Transfer.get_logs.return_value = logs

    node.ftns_ledger = ledger
    app = create_api_app(node, enable_security=False)
    client = TestClient(app)
    r = client.get(
        "/wallet/transactions/onchain/inbound/stats"
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 3
    assert body["total_inbound_ftns"] == 4.0  # 0.5+1.5+2.0
    assert body["first_inbound_block"] == 99000
    assert body["last_inbound_block"] == 99950


def test_inbound_stats_empty_state():
    from prsm.node.api import create_api_app
    node = MagicMock()
    ledger = MagicMock()
    ledger._connected_address = "0x" + "a" * 40
    ledger.w3 = MagicMock()
    ledger.w3.eth.block_number = 100
    ledger._token = MagicMock()
    ledger._token.events.Transfer.get_logs.return_value = []
    node.ftns_ledger = ledger
    app = create_api_app(node, enable_security=False)
    client = TestClient(app)
    r = client.get(
        "/wallet/transactions/onchain/inbound/stats"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["total_inbound_ftns"] == 0.0
    assert body["first_inbound_block"] is None
    assert body["last_inbound_block"] is None


def test_health_detailed_includes_inbound_monitor():
    """When _inbound_monitor exists on the node, the
    /health/detailed.subsystems.inbound_monitor key must
    appear with last_scanned_block etc."""
    from prsm.node.api import create_api_app
    node = MagicMock()
    ledger = MagicMock()
    ledger._is_initialized = True
    ledger._connected_address = "0x" + "a" * 40
    ledger.contract_address = "0x" + "b" * 40
    ledger.w3 = MagicMock()
    ledger.w3.eth.get_balance.return_value = 10**15
    node.ftns_ledger = ledger
    monitor = MagicMock()
    monitor._last_scanned_block = 46160000
    node._inbound_monitor = monitor
    app = create_api_app(node, enable_security=False)
    client = TestClient(app)
    body = client.get("/health/detailed").json()
    assert "inbound_monitor" in body["subsystems"]
    sub = body["subsystems"]["inbound_monitor"]
    assert sub["available"] is True
    assert sub["last_scanned_block"] == 46160000


def test_health_detailed_inbound_monitor_not_wired():
    """No _inbound_monitor on node → subsystem reports
    not_wired (operator without env config doesn't
    silently lack monitoring)."""
    from prsm.node.api import create_api_app
    node = MagicMock()
    # MagicMock auto-attrs make this tricky; explicit None
    node._inbound_monitor = None
    ledger = MagicMock()
    ledger._is_initialized = True
    ledger._connected_address = "0x" + "a" * 40
    ledger.contract_address = "0x" + "b" * 40
    ledger.w3 = MagicMock()
    ledger.w3.eth.get_balance.return_value = 10**15
    node.ftns_ledger = ledger
    app = create_api_app(node, enable_security=False)
    client = TestClient(app)
    body = client.get("/health/detailed").json()
    sub = body["subsystems"].get("inbound_monitor", {})
    assert sub.get("status") == "not_wired"

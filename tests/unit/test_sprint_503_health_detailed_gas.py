"""Sprint 503 — /health/detailed surfaces operator_gas subsystem.

Sprint 502 shipped /wallet/gas-status as a dedicated endpoint.
Sprint 503 multiplies its reach by surfacing the same signal in
/health/detailed under `subsystems.operator_gas`, so monitoring
tools (Prometheus exporters, uptime checks, datadog, etc.)
already polling /health/detailed pick up the gas signal without
extra config.

Pin tests verify:
  - operator_gas subsystem present when ftns_ledger wired
  - status mirrors /wallet/gas-status logic (ok/low/critical)
  - thresholds present
  - graceful when w3 not initialized
  - graceful when ftns_ledger missing entirely
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def _build_app(eth_wei=None, has_w3=True, has_ledger=True):
    from prsm.node.api import create_api_app

    node = MagicMock()
    if has_ledger:
        ledger = MagicMock()
        ledger._connected_address = (
            "0x4acdE458766C704B2511583572303e77109cFFE8"
        )
        ledger._is_initialized = True
        ledger.contract_address = (
            "0x5276a3756C85f2E9e46f6D34386167a209aa16e5"
        )
        if has_w3:
            ledger.w3 = MagicMock()
            ledger.w3.eth.get_balance.return_value = eth_wei
        else:
            ledger.w3 = None
        node.ftns_ledger = ledger
    else:
        node.ftns_ledger = None
    return create_api_app(node, enable_security=False)


def test_health_detailed_includes_operator_gas_when_healthy():
    """0.001 ETH → status=ok with full schema."""
    app = _build_app(eth_wei=10**15)
    client = TestClient(app)
    body = client.get("/health/detailed").json()
    assert "operator_gas" in body["subsystems"]
    gas = body["subsystems"]["operator_gas"]
    assert gas["available"] is True
    assert gas["status"] == "ok"
    assert gas["eth_balance"] == 0.001
    assert gas["low_threshold_eth"] == 0.0005
    assert gas["critical_threshold_eth"] == 0.0001


def test_health_detailed_operator_gas_low_status():
    """0.0003 ETH → status=low."""
    app = _build_app(eth_wei=3 * 10**14)
    client = TestClient(app)
    body = client.get("/health/detailed").json()
    assert body["subsystems"]["operator_gas"]["status"] == "low"


def test_health_detailed_operator_gas_critical_status():
    """0.00005 ETH → status=critical."""
    app = _build_app(eth_wei=5 * 10**13)
    client = TestClient(app)
    body = client.get("/health/detailed").json()
    assert (
        body["subsystems"]["operator_gas"]["status"]
        == "critical"
    )


def test_health_detailed_operator_gas_unavailable_no_w3():
    """If w3 not initialized, subsystem must report
    unavailable, not crash."""
    app = _build_app(eth_wei=0, has_w3=False)
    client = TestClient(app)
    body = client.get("/health/detailed").json()
    gas = body["subsystems"]["operator_gas"]
    assert gas["available"] is False
    assert gas["status"] == "unavailable"


def test_health_detailed_operator_gas_not_wired_no_ledger():
    """If no ftns_ledger at all, subsystem must still
    appear with status=not_wired so monitoring tools can
    alert."""
    app = _build_app(has_ledger=False)
    client = TestClient(app)
    body = client.get("/health/detailed").json()
    assert "operator_gas" in body["subsystems"]
    assert (
        body["subsystems"]["operator_gas"]["status"]
        == "not_wired"
    )

"""Sprint 501 — endpoint scope field reflects persistence.

The async fixtures in test_sprint_501_onchain_tx_persistence.py
trigger asyncio.AUTO mode which makes sync TestClient flaky in
the same file. This pin lives in its own file so the TestClient
runs in plain sync context.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def test_endpoint_scope_persistent_when_db_path_set():
    """When ledger.is_persistent=True, the `scope` field
    must contain "persistent" AND the db_path so operators
    know where their audit trail lives."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    ledger = MagicMock()
    ledger._connected_address = "0xAAAA"
    ledger._transactions = []
    ledger.is_persistent = True
    ledger.db_path = "/some/path/onchain_tx.db"
    node.ftns_ledger = ledger
    app = create_api_app(node, enable_security=False)

    client = TestClient(app)
    r = client.get("/wallet/transactions/onchain")
    assert r.status_code == 200
    body = r.json()
    assert "persistent" in body["scope"].lower()
    assert "/some/path/onchain_tx.db" in body["scope"]


def test_endpoint_scope_in_memory_when_no_db_path():
    """When ledger.is_persistent=False, scope still
    documents in-memory honestly + points to the fix."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    ledger = MagicMock()
    ledger._connected_address = "0xAAAA"
    ledger._transactions = []
    ledger.is_persistent = False
    ledger.db_path = None
    node.ftns_ledger = ledger
    app = create_api_app(node, enable_security=False)

    client = TestClient(app)
    r = client.get("/wallet/transactions/onchain")
    body = r.json()
    assert "in-memory" in body["scope"].lower()
    assert "db_path" in body["scope"]

"""Sprint 512 — endpoint-contract pin for inbound transfer scan.

Sibling to test_sprint_512_inbound_transfer_scan.py — split to
avoid asyncio.AUTO mode interaction with sync TestClient (same
pattern as sprints 501, 510).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def test_endpoint_503_when_ledger_missing():
    from prsm.node.api import create_api_app

    node = MagicMock()
    node.ftns_ledger = None
    app = create_api_app(node, enable_security=False)
    client = TestClient(app)
    r = client.get("/wallet/transactions/onchain/inbound")
    assert r.status_code == 503


def test_endpoint_503_when_w3_not_initialized():
    """503 when ledger exists but w3/token/address missing."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    ledger = MagicMock()
    ledger.w3 = None
    ledger._connected_address = None
    ledger._token = None
    node.ftns_ledger = ledger
    app = create_api_app(node, enable_security=False)
    client = TestClient(app)
    r = client.get("/wallet/transactions/onchain/inbound")
    assert r.status_code == 503


def test_endpoint_happy_path_returns_transfers():
    """Mocked ledger + scan_inbound_transfers → canonical
    response envelope."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    ledger = MagicMock()
    ledger.w3 = MagicMock()
    ledger.w3.eth.block_number = 46200000
    ledger._connected_address = "0x" + "a" * 40
    ledger._token = MagicMock()
    log = MagicMock()
    log.blockNumber = 46159546
    log.transactionHash = b"\xde\xad\xbe\xef" + b"\x00" * 28
    log.args.__getitem__ = lambda self, k: {
        "from": "0x91b0e6F85A371D82De94eD13A3812d9f5A4E5791",
        "to": "0x" + "a" * 40,
        "value": 2 * 10**18,
    }[k]
    ledger._token.events.Transfer.get_logs.return_value = [log]
    node.ftns_ledger = ledger
    app = create_api_app(node, enable_security=False)
    client = TestClient(app)
    r = client.get("/wallet/transactions/onchain/inbound")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["recipient"] == "0x" + "a" * 40
    assert body["count"] == 1
    assert body["transfers"][0]["amount_ftns"] == 2.0

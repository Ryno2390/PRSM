"""Sprint 199 — /ledger/transfer financial-input validation.

Pre-fix surface gaps:

  1. `amount=NaN`  — `nan <= 0` is False, passes the only guard,
                     reaches signed_transfer (financial UB).
  2. `amount=inf`  — same: `inf <= 0` is False, passes through.

Post-fix: both return 422 with structured detail. The `to_wallet`
string length is mitigated by FastAPI's URL-length default
(httpx already rejects >1MB query strings client-side; nginx /
uvicorn enforce server-side), so it's NOT in scope for sprint 199.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client():
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node.ledger_sync = MagicMock()
    node.ledger_sync.signed_transfer = AsyncMock(return_value=None)
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_nan_amount_rejected():
    resp = _client().post(
        "/ledger/transfer",
        params={"to_wallet": "0xabc", "amount": "NaN"},
    )
    assert resp.status_code == 422


def test_inf_amount_rejected():
    resp = _client().post(
        "/ledger/transfer",
        params={"to_wallet": "0xabc", "amount": "Infinity"},
    )
    assert resp.status_code == 422


def test_negative_inf_amount_rejected():
    resp = _client().post(
        "/ledger/transfer",
        params={"to_wallet": "0xabc", "amount": "-Infinity"},
    )
    # Either 400 (amount <= 0) or 422 (sprint 199 finite check).
    # We accept either; the bug is silently passing it through.
    assert resp.status_code in (400, 422)


def test_typical_transfer_passes_validation():
    resp = _client().post(
        "/ledger/transfer",
        params={"to_wallet": "0xabc123", "amount": 1.0},
    )
    # Sanity: ordinary call should not hit a 422/414.
    assert resp.status_code not in (422, 414)

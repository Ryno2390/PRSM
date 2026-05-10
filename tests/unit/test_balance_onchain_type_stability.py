"""Sprint 162 — /balance/onchain `escrowed_ftns` is float-stable.

Pre-fix when the requester had no pending escrows, the response
returned `escrowed_ftns: 0` (int) while sibling fields were
floats:

  balance_ftns:             float = 0.0
  claimable_royalties_ftns: float = 0.0
  total_ftns:               float = 0.0
  escrowed_ftns:            int = 0    ← inconsistent

Operator scripts iterating these fields uniformly (e.g.,
multiplying by usd_rate, summing across keys) could break or
produce subtle rounding artifacts.

Fix: sum(..., start=0.0) keeps the type float regardless of
whether `pending` is empty or populated.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _node():
    node = MagicMock()
    node.identity.node_id = "test-node"
    # ftns_ledger present, balance 0
    ftns = MagicMock()
    ftns._is_initialized = True
    ftns._connected_address = "0x" + "ab" * 20
    ftns.get_balance = AsyncMock(return_value=0.0)
    ftns._decimals = 18
    node.ftns_ledger = ftns

    # PaymentEscrow present, no pending escrows
    pe = MagicMock()
    pe.list_escrows_by_requester = MagicMock(return_value=[])
    node._payment_escrow = pe
    node._royalty_distributor_client = None
    return node


def _client(node):
    return TestClient(create_api_app(node, enable_security=False))


def test_escrowed_ftns_float_when_empty():
    """Sprint 162 — empty pending list returns 0.0 (float),
    NOT 0 (int)."""
    resp = _client(_node()).get("/balance/onchain")
    body = resp.json()
    # JSON has no int/float distinction, so assert via the
    # presence of `.0` in the raw response payload.
    raw = resp.text
    assert '"escrowed_ftns":0.0' in raw or '"escrowed_ftns": 0.0' in raw, (
        f"escrowed_ftns must be float-formatted; raw response: {raw}"
    )


def test_escrowed_ftns_float_when_populated():
    """Sprint 162 invariant — populated path also stays float."""
    node = _node()
    e = MagicMock()
    e.amount = 1.5
    node._payment_escrow.list_escrows_by_requester.return_value = [e]
    resp = _client(node).get("/balance/onchain")
    raw = resp.text
    assert '"escrowed_ftns":1.5' in raw or '"escrowed_ftns": 1.5' in raw

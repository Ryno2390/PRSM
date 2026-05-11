"""Sprint 250 — GET /compute/receipts endpoint."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app
from prsm.node.receipt_store import ReceiptStore


def _client(store=None):
    node = MagicMock()
    node.identity.node_id = "test-node"
    node.ftns_ledger = None
    node._receipt_store = store
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_returns_paginated_list():
    s = ReceiptStore()
    for i in range(3):
        s.put(f"j{i}", {"job_id": f"j{i}", "model_id": "m1"})
    resp = _client(s).get("/compute/receipts?limit=2&offset=0")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["receipts"]) == 2
    # Newest first
    assert body["receipts"][0]["job_id"] == "j2"


def test_model_id_filter_passes_through():
    s = ReceiptStore()
    s.put("a", {"job_id": "a", "model_id": "m1"})
    s.put("b", {"job_id": "b", "model_id": "m2"})
    s.put("c", {"job_id": "c", "model_id": "m1"})
    resp = _client(s).get("/compute/receipts?model_id=m1")
    assert resp.status_code == 200
    body = resp.json()
    assert {r["job_id"] for r in body["receipts"]} == {"a", "c"}


def test_503_when_store_unwired():
    resp = _client(None).get("/compute/receipts")
    assert resp.status_code == 503


def test_422_on_bad_limit():
    resp = _client(ReceiptStore()).get(
        "/compute/receipts?limit=2000",
    )
    assert resp.status_code == 422


def test_422_on_negative_offset():
    resp = _client(ReceiptStore()).get(
        "/compute/receipts?offset=-1",
    )
    assert resp.status_code == 422


def test_empty_store_returns_zero():
    resp = _client(ReceiptStore()).get("/compute/receipts")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["receipts"] == []

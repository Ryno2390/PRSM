"""Sprint 510 — endpoint-contract pin for chain-filtered TX history.

Sibling file to test_sprint_510_f39_chain_id_filter.py — split
to avoid asyncio.AUTO mode interaction with sync TestClient
(same pattern as sprint 501).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def test_endpoint_serves_only_pre_filtered_transactions():
    """The /wallet/transactions/onchain endpoint reads
    ledger._transactions which is now sprint-510-filtered
    at _init_persistence time. End-to-end: a Sepolia daemon
    sees only Sepolia rows, never mainnet."""
    from prsm.node.api import create_api_app

    node = MagicMock()
    ledger = MagicMock()
    ledger._connected_address = "0xAAAA"
    ledger.is_persistent = True
    ledger.db_path = "/x"
    only_this_chain_tx = MagicMock()
    only_this_chain_tx.tx_hash = "0x" + "22" * 32
    only_this_chain_tx.status = "confirmed"
    only_this_chain_tx.block_number = 10000001
    only_this_chain_tx.from_addr = "0xCCCC"
    only_this_chain_tx.to_addr = "0xDDDD"
    only_this_chain_tx.amount_ftns = 0.5
    only_this_chain_tx.created_at = 1700001000.0
    only_this_chain_tx.job_id = "sepolia-1"
    ledger._transactions = [only_this_chain_tx]
    node.ftns_ledger = ledger
    app = create_api_app(node, enable_security=False)
    client = TestClient(app)
    body = client.get("/wallet/transactions/onchain").json()
    assert body["count"] == 1
    assert body["transactions"][0]["job_id"] == "sepolia-1"

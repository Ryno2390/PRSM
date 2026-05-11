"""Sprint 264 — faucet caps tunable via env.

Pre-fix /ftns/faucet hardcoded 100 FTNS/request + 1000 FTNS/
wallet. Operators on stress-test or restrictive testnets had
no way to adjust without forking. Mirrors the operator-tunable
env pattern (PRSM_MAX_QUERY_BYTES, PRSM_AGGREGATOR_SHARE_BPS,
PRSM_ESCROW_TIMEOUT_SEC, etc.) across the rest of the codebase.

Env vars (both have sane defaults preserving sprint-181
behavior):
  PRSM_FAUCET_MAX_PER_REQUEST  default 100
  PRSM_FAUCET_MAX_PER_WALLET   default 1000

Both fail-soft to defaults on non-numeric / zero / negative
values, matching the rest of the codebase's env-handling
discipline.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.node.api import create_api_app


def _client(balance=0.0, env=None, monkeypatch=None):
    node = MagicMock()
    node.identity.node_id = "wallet-self"
    node.ftns_ledger = None
    node.ledger = MagicMock()
    node.ledger.get_balance = AsyncMock(return_value=balance)
    node.ledger.credit = AsyncMock()
    if env and monkeypatch:
        for k, v in env.items():
            monkeypatch.setenv(k, v)
    return TestClient(
        create_api_app(node, enable_security=False),
        raise_server_exceptions=False,
    )


def test_default_caps_preserve_sprint_181_behavior(monkeypatch):
    monkeypatch.delenv("PRSM_FAUCET_MAX_PER_REQUEST", raising=False)
    monkeypatch.delenv("PRSM_FAUCET_MAX_PER_WALLET", raising=False)
    # 100 cap: requesting 200 returns 100 (clamped)
    resp = _client(balance=0).post(
        "/ftns/faucet", json={"amount": 200},
    )
    assert resp.status_code == 200
    assert resp.json()["granted"] == 100


def test_per_request_cap_env_override(monkeypatch):
    monkeypatch.setenv("PRSM_FAUCET_MAX_PER_REQUEST", "50")
    resp = _client(balance=0, monkeypatch=monkeypatch).post(
        "/ftns/faucet", json={"amount": 200},
    )
    body = resp.json()
    assert body["granted"] == 50


def test_per_wallet_cap_env_override(monkeypatch):
    monkeypatch.setenv("PRSM_FAUCET_MAX_PER_WALLET", "500")
    # Wallet has 500 FTNS — should hit the 500 cap
    resp = _client(balance=500.0, monkeypatch=monkeypatch).post(
        "/ftns/faucet", json={"amount": 100},
    )
    assert resp.status_code == 429
    assert "500" in resp.json()["detail"]


def test_bad_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(
        "PRSM_FAUCET_MAX_PER_REQUEST", "not_a_number",
    )
    resp = _client(balance=0, monkeypatch=monkeypatch).post(
        "/ftns/faucet", json={"amount": 200},
    )
    # Falls back to 100 default
    assert resp.json()["granted"] == 100


def test_zero_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PRSM_FAUCET_MAX_PER_REQUEST", "0")
    resp = _client(balance=0, monkeypatch=monkeypatch).post(
        "/ftns/faucet", json={"amount": 200},
    )
    # Falls back to 100 default (zero would brick the faucet)
    assert resp.json()["granted"] == 100


def test_negative_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PRSM_FAUCET_MAX_PER_REQUEST", "-5")
    resp = _client(balance=0, monkeypatch=monkeypatch).post(
        "/ftns/faucet", json={"amount": 200},
    )
    assert resp.json()["granted"] == 100

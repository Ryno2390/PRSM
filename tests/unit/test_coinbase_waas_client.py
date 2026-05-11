"""Sprint 276 — CoinbaseWaaSClient.

Coinbase Wallet-as-a-Service adapter. Per Vision §14 mitigation
("Embedded wallets by default … indistinguishable from web2
onboarding") the goal is to make crypto invisible: user creates
PRSM account, server-side MPC wallet is provisioned via Coinbase
CDP, user never sees a seed phrase.

This v1 scaffold:
  - Returns PENDING_COMMISSION records when CDP env keys are
    absent (mirrors the offramp quote pattern).
  - When commissioned (keys set + backend wired), defers wallet
    creation to a dependency-injected backend. The real backend
    will wrap the Coinbase CDP SDK; tests use a fake.
  - Persists wallet records as JSON files in PRSM_WAAS_STORE_DIR
    when set, so a re-provision request for the same user_id is
    idempotent across restarts.
"""
from __future__ import annotations

import json

import pytest

from prsm.economy.web3.coinbase_waas_client import (
    CoinbaseWaaSClient, WaasWalletRecord,
)


class FakeBackend:
    """Test backend mirroring the real CDP SDK surface.

    Generates deterministic addresses so tests can assert exact
    values. Production backend will hit Coinbase CDP."""

    def __init__(self):
        self.calls = []

    def create_wallet(self, user_id, email):
        self.calls.append((user_id, email))
        return {
            "wallet_id": f"wallet-{user_id}",
            "address": f"0x{user_id:0>40}",
            "network": "base-mainnet",
        }


# ── PENDING_COMMISSION (no env keys) ─────────────────────


def test_from_env_returns_uncommissioned_when_keys_missing(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("COINBASE_CDP_API_KEY_NAME", raising=False)
    monkeypatch.delenv(
        "COINBASE_CDP_API_KEY_PRIVATE", raising=False,
    )
    monkeypatch.setenv("PRSM_WAAS_STORE_DIR", str(tmp_path))
    c = CoinbaseWaaSClient.from_env()
    assert c is not None
    assert c.is_commissioned() is False


def test_provision_uncommissioned_returns_pending_record():
    c = CoinbaseWaaSClient()
    record = c.provision_wallet(
        user_id="alice", email="alice@example.com",
    )
    assert record.user_id == "alice"
    assert record.status == "PENDING_COMMISSION"
    assert record.wallet_id is None
    assert record.address is None


def test_provision_uncommissioned_does_not_hit_backend():
    fake = FakeBackend()
    c = CoinbaseWaaSClient(backend=fake)  # backend set but no keys
    c.provision_wallet(
        user_id="alice", email="alice@example.com",
    )
    assert fake.calls == []  # commissioned check gates SDK call


# ── COMMISSIONED (keys present) ──────────────────────────


def test_provision_commissioned_creates_wallet_via_backend():
    fake = FakeBackend()
    c = CoinbaseWaaSClient(
        api_key_name="key", api_key_private="priv",
        backend=fake,
    )
    assert c.is_commissioned() is True
    record = c.provision_wallet(
        user_id="alice", email="alice@example.com",
    )
    assert record.status == "PROVISIONED"
    assert record.wallet_id == "wallet-alice"
    assert record.address.startswith("0x")
    assert record.network == "base-mainnet"
    assert fake.calls == [("alice", "alice@example.com")]


def test_provision_idempotent_for_same_user_id():
    fake = FakeBackend()
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p", backend=fake,
    )
    first = c.provision_wallet(user_id="alice", email="a@x.io")
    second = c.provision_wallet(user_id="alice", email="a@x.io")
    assert first.wallet_id == second.wallet_id
    # Backend invoked only once — second call returns cached.
    assert len(fake.calls) == 1


def test_provision_validates_required_fields():
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        backend=FakeBackend(),
    )
    with pytest.raises(ValueError):
        c.provision_wallet(user_id="", email="a@x.io")
    with pytest.raises(ValueError):
        c.provision_wallet(user_id="alice", email="")


# ── Lookup + list ────────────────────────────────────────


def test_get_wallet_returns_none_for_unknown_user():
    c = CoinbaseWaaSClient()
    assert c.get_wallet("nobody") is None


def test_get_wallet_returns_record_after_provision():
    fake = FakeBackend()
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p", backend=fake,
    )
    c.provision_wallet(user_id="alice", email="a@x.io")
    record = c.get_wallet("alice")
    assert record is not None
    assert record.user_id == "alice"
    assert record.status == "PROVISIONED"


def test_list_wallets_empty():
    c = CoinbaseWaaSClient()
    assert c.list_wallets() == []


def test_list_wallets_returns_provisioned_records():
    fake = FakeBackend()
    c = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p", backend=fake,
    )
    c.provision_wallet(user_id="alice", email="a@x.io")
    c.provision_wallet(user_id="bob", email="b@x.io")
    ids = sorted(r.user_id for r in c.list_wallets())
    assert ids == ["alice", "bob"]


# ── Persistence ──────────────────────────────────────────


def test_persistence_round_trip(tmp_path):
    fake = FakeBackend()
    c1 = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        persist_dir=tmp_path, backend=fake,
    )
    c1.provision_wallet(user_id="alice", email="a@x.io")
    # New instance loads from disk
    c2 = CoinbaseWaaSClient(
        api_key_name="k", api_key_private="p",
        persist_dir=tmp_path, backend=FakeBackend(),
    )
    record = c2.get_wallet("alice")
    assert record is not None
    assert record.wallet_id == "wallet-alice"


def test_persistence_corrupt_file_fail_soft(tmp_path):
    (tmp_path / "garbage.json").write_text("{not valid json")
    c = CoinbaseWaaSClient(persist_dir=tmp_path)
    assert c.list_wallets() == []


def test_record_to_dict_round_trip():
    r = WaasWalletRecord(
        user_id="alice", email="a@x.io",
        wallet_id="wallet-alice", address="0xabc",
        network="base-mainnet", status="PROVISIONED",
        created_at=100.0,
    )
    d = r.to_dict()
    assert d["user_id"] == "alice"
    assert d["status"] == "PROVISIONED"
    restored = WaasWalletRecord.from_dict(d)
    assert restored == r

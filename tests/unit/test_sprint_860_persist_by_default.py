"""Sprint 860 — KYC + WaaS persist by default; :memory: opt-out.

Closes the observability gap surfaced during sp859: WaaS
`wallet_count` returned 0 even though 3 real Base mainnet wallets
had been provisioned in the same session. Cause: the in-memory
dict didn't persist across daemon restarts because
PRSM_WAAS_STORE_DIR + PRSM_KYC_STORE_DIR defaulted to None.

Sp860 flips the default to ``~/.prsm/{waas-wallets,kyc-records}/``
so records persist by default. Operators opt out via the env-var
sentinel ``:memory:`` (Python sqlite-style).

Pin tests:
  - Unset env → persist at ~/.prsm/{waas-wallets,kyc-records}
  - Env=":memory:" → no persistence (in-memory only)
  - Explicit path → respected (existing operator override path)
  - Created dir survives roundtrip (write + reload + same records)
  - Default dirs don't clobber existing operator-set paths
"""
from __future__ import annotations

from pathlib import Path

import pytest

from prsm.economy.web3.coinbase_waas_client import CoinbaseWaaSClient
from prsm.economy.web3.kyc_client import KYCClient


# ── WaaS default-persist behavior ────────────────────────────

def test_waas_from_env_persists_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("PRSM_WAAS_STORE_DIR", raising=False)
    # Redirect HOME so the test doesn't write to the real
    # ~/.prsm/ directory.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("COINBASE_CDP_API_KEY_NAME", raising=False)
    monkeypatch.delenv("COINBASE_CDP_API_KEY_PRIVATE", raising=False)
    c = CoinbaseWaaSClient.from_env()
    expected = tmp_path / ".prsm" / "waas-wallets"
    assert c._persist_dir == expected
    # mkdir parents=True ran in __init__
    assert expected.exists()


def test_waas_from_env_memory_sentinel_opts_out(
    monkeypatch, tmp_path,
):
    monkeypatch.setenv("PRSM_WAAS_STORE_DIR", ":memory:")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("COINBASE_CDP_API_KEY_NAME", raising=False)
    monkeypatch.delenv("COINBASE_CDP_API_KEY_PRIVATE", raising=False)
    c = CoinbaseWaaSClient.from_env()
    assert c._persist_dir is None
    # No directory created
    assert not (tmp_path / ".prsm" / "waas-wallets").exists()


def test_waas_from_env_explicit_path_respected(
    monkeypatch, tmp_path,
):
    custom = tmp_path / "custom-waas-dir"
    monkeypatch.setenv("PRSM_WAAS_STORE_DIR", str(custom))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("COINBASE_CDP_API_KEY_NAME", raising=False)
    monkeypatch.delenv("COINBASE_CDP_API_KEY_PRIVATE", raising=False)
    c = CoinbaseWaaSClient.from_env()
    assert c._persist_dir == custom
    assert custom.exists()
    # Default path NOT created (operator override beats default)
    assert not (tmp_path / ".prsm" / "waas-wallets").exists()


# ── KYC default-persist behavior ─────────────────────────────

def test_kyc_from_env_persists_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("PRSM_KYC_STORE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KYC_VENDOR", raising=False)
    monkeypatch.delenv("KYC_VENDOR_API_KEY", raising=False)
    c = KYCClient.from_env()
    expected = tmp_path / ".prsm" / "kyc-records"
    assert c._persist_dir == expected
    assert expected.exists()


def test_kyc_from_env_memory_sentinel_opts_out(monkeypatch, tmp_path):
    monkeypatch.setenv("PRSM_KYC_STORE_DIR", ":memory:")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KYC_VENDOR", raising=False)
    monkeypatch.delenv("KYC_VENDOR_API_KEY", raising=False)
    c = KYCClient.from_env()
    assert c._persist_dir is None
    assert not (tmp_path / ".prsm" / "kyc-records").exists()


def test_kyc_from_env_explicit_path_respected(monkeypatch, tmp_path):
    custom = tmp_path / "custom-kyc-dir"
    monkeypatch.setenv("PRSM_KYC_STORE_DIR", str(custom))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KYC_VENDOR", raising=False)
    monkeypatch.delenv("KYC_VENDOR_API_KEY", raising=False)
    c = KYCClient.from_env()
    assert c._persist_dir == custom


# ── Roundtrip via persistence ────────────────────────────────

def test_waas_roundtrip_via_default_persist_dir(
    monkeypatch, tmp_path,
):
    """A wallet provisioned in one client instance is visible in
    a second instance constructed via from_env() — the actual
    operator-facing guarantee sp860 closes."""
    monkeypatch.delenv("PRSM_WAAS_STORE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    # Provision requires commissioned env (any non-empty key fine
    # since the backend is injected directly).
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "kid_test")
    monkeypatch.setenv("COINBASE_CDP_API_KEY_PRIVATE", "p_test")

    # Inject a fake backend so provision_wallet returns immediately
    # without hitting CDP.
    class _FakeBackend:
        def create_wallet(self, user_id, email):
            return {
                "wallet_id": f"w_{user_id}",
                "address": "0x" + "11" * 20,
                "network": "base-mainnet",
            }

    c1 = CoinbaseWaaSClient.from_env(backend=_FakeBackend())
    rec1 = c1.provision_wallet("alice", "a@x.io")
    assert rec1.status == "PROVISIONED"

    # Fresh from_env reload — should see alice's wallet.
    c2 = CoinbaseWaaSClient.from_env(backend=_FakeBackend())
    rec2 = c2.get_wallet("alice")
    assert rec2 is not None
    assert rec2.address == "0x" + "11" * 20
    assert rec2.user_id == "alice"


def test_kyc_roundtrip_via_default_persist_dir(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("PRSM_KYC_STORE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KYC_VENDOR", "persona")
    monkeypatch.setenv("KYC_VENDOR_API_KEY", "k")
    monkeypatch.setenv("PERSONA_TEMPLATE_ID", "itmpl_x")

    class _FakeBackend:
        def initiate_session(self, user_id, email, level):
            return {
                "vendor_ref": "inq_" + user_id,
                "session_url": "https://withpersona.com/v/x",
                "status": "INITIATED",
            }

    c1 = KYCClient.from_env(backend=_FakeBackend())
    rec1 = c1.initiate("bob", "b@x.io", "basic")
    assert rec1.status == "INITIATED"

    # Reload — bob's record persists.
    c2 = KYCClient.from_env(backend=_FakeBackend())
    rec2 = c2.get_status("bob")
    assert rec2 is not None
    assert rec2.vendor_ref == "inq_bob"

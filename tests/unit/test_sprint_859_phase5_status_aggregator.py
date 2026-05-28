"""Sprint 859 — Phase 5 readiness aggregator pin tests.

Defends the canonical schema + the READY/PARTIAL/NOT_READY rollup
logic. Each per-surface status block must keep its load-bearing
fields stable so operator dashboards + the CLI surface that
consumes this can render without breaking on schema drift.

Pin tests:
  - 5 surfaces present + canonical names
  - NOT_READY when all clients are None
  - PARTIAL when some surfaces live
  - READY when all surfaces live
  - live_surface_count + total_surface_count match
  - Each surface has commissioned + adapter_wired + live_exec + notes
  - Onramp is env-driven (no client) — picks up CDP keys + Pay App ID
  - Aerodrome surface reports pool_configured + pool_address
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.phase5_status import aggregate_phase5_status


class _FakeKYC:
    def __init__(self, *, commissioned=True, wired=True,
                 vendor="persona"):
        self._c, self._w = commissioned, wired
        self._vendor = vendor

    def is_commissioned(self): return self._c
    def adapter_wired(self): return self._w


class _FakeWaaS:
    def __init__(self, *, commissioned=True, wired=True, wallets=0):
        self._c, self._w = commissioned, wired
        self._wallets = ["w"] * wallets

    def is_commissioned(self): return self._c
    def adapter_wired(self): return self._w
    def list_wallets(self): return self._wallets


class _FakePaymaster:
    def __init__(
        self, *, commissioned=True, wired=True, sponsorships=0,
    ):
        self._c, self._w, self._s = commissioned, wired, sponsorships

    def spend_summary(self):
        return {
            "commissioned": self._c,
            "adapter_wired": self._w,
            "sponsorships": self._s,
            "total_sponsored_wei": 0,
            "endpoint": "https://x",
            "policy_id": None,
        }


class _FakeAerodrome:
    def __init__(self, *, configured=False, pool="0xPOOL"):
        self._configured = configured
        self.pool_address = pool

    def is_configured(self): return self._configured


# ── Canonical schema ─────────────────────────────────────────

def test_aggregate_returns_5_surfaces(monkeypatch):
    monkeypatch.delenv("COINBASE_CDP_API_KEY_NAME", raising=False)
    monkeypatch.delenv("COINBASE_CDP_API_KEY_PRIVATE", raising=False)
    monkeypatch.delenv("COINBASE_PAY_APP_ID", raising=False)
    r = aggregate_phase5_status()
    assert set(r["surfaces"].keys()) == {
        "kyc", "waas", "paymaster", "onramp", "aerodrome",
    }


def test_each_surface_has_required_fields(monkeypatch):
    monkeypatch.delenv("COINBASE_CDP_API_KEY_NAME", raising=False)
    monkeypatch.delenv("COINBASE_CDP_API_KEY_PRIVATE", raising=False)
    r = aggregate_phase5_status()
    for name, surface in r["surfaces"].items():
        assert "commissioned" in surface, (
            f"{name} missing commissioned"
        )
        assert "adapter_wired" in surface, (
            f"{name} missing adapter_wired"
        )
        assert "live_exec" in surface, f"{name} missing live_exec"
        assert "notes" in surface, f"{name} missing notes"
        assert isinstance(surface["notes"], str)


# ── overall rollup ───────────────────────────────────────────

def test_overall_not_ready_when_all_clients_none(monkeypatch):
    monkeypatch.delenv("COINBASE_CDP_API_KEY_NAME", raising=False)
    monkeypatch.delenv("COINBASE_CDP_API_KEY_PRIVATE", raising=False)
    r = aggregate_phase5_status()
    assert r["overall"] == "NOT_READY"
    assert r["live_surface_count"] == 0
    assert r["total_surface_count"] == 5


def test_overall_partial_when_some_live(monkeypatch):
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "k")
    monkeypatch.setenv("COINBASE_CDP_API_KEY_PRIVATE", "p")
    r = aggregate_phase5_status(
        kyc_client=_FakeKYC(),
        waas_client=_FakeWaaS(),
        # paymaster + aerodrome left None
    )
    assert r["overall"] == "PARTIAL"
    assert 0 < r["live_surface_count"] < r["total_surface_count"]


def test_overall_ready_when_all_live(monkeypatch):
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "k")
    monkeypatch.setenv("COINBASE_CDP_API_KEY_PRIVATE", "p")
    r = aggregate_phase5_status(
        kyc_client=_FakeKYC(),
        waas_client=_FakeWaaS(),
        # Paymaster live_exec requires sponsorships>0
        paymaster_client=_FakePaymaster(sponsorships=1),
        aerodrome_client=_FakeAerodrome(configured=True),
    )
    assert r["overall"] == "READY"
    assert r["live_surface_count"] == 5


# ── Onramp env-driven path ───────────────────────────────────

def test_onramp_picks_up_cdp_keys(monkeypatch):
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "k")
    monkeypatch.setenv("COINBASE_CDP_API_KEY_PRIVATE", "p")
    monkeypatch.delenv("COINBASE_PAY_APP_ID", raising=False)
    r = aggregate_phase5_status()
    assert r["surfaces"]["onramp"]["commissioned"] is True
    assert r["surfaces"]["onramp"]["adapter_wired"] is True
    assert r["surfaces"]["onramp"]["live_exec"] is True
    # Pay App ID separate flag — false here
    assert r["surfaces"]["onramp"]["pay_app_id_set"] is False
    assert r["surfaces"]["onramp"]["secure_init"] is True


def test_onramp_dark_without_cdp_keys(monkeypatch):
    monkeypatch.delenv("COINBASE_CDP_API_KEY_NAME", raising=False)
    monkeypatch.delenv("COINBASE_CDP_API_KEY_PRIVATE", raising=False)
    r = aggregate_phase5_status()
    assert r["surfaces"]["onramp"]["commissioned"] is False
    assert r["surfaces"]["onramp"]["live_exec"] is False


def test_onramp_pay_app_id_separate_signal(monkeypatch):
    monkeypatch.setenv("COINBASE_CDP_API_KEY_NAME", "k")
    monkeypatch.setenv("COINBASE_CDP_API_KEY_PRIVATE", "p")
    monkeypatch.setenv("COINBASE_PAY_APP_ID", "f4a5bc")
    r = aggregate_phase5_status()
    assert r["surfaces"]["onramp"]["pay_app_id_set"] is True


# ── Per-surface signal pass-through ──────────────────────────

def test_kyc_signal_passthrough():
    r = aggregate_phase5_status(
        kyc_client=_FakeKYC(commissioned=True, wired=False),
    )
    s = r["surfaces"]["kyc"]
    assert s["commissioned"] is True
    assert s["adapter_wired"] is False
    assert s["live_exec"] is False  # both must be true for live
    assert s["vendor"] == "persona"


def test_waas_signal_passthrough_wallet_count():
    r = aggregate_phase5_status(
        waas_client=_FakeWaaS(wallets=42),
    )
    assert r["surfaces"]["waas"]["wallet_count"] == 42


def test_paymaster_signal_passthrough():
    r = aggregate_phase5_status(
        paymaster_client=_FakePaymaster(
            commissioned=True, wired=True, sponsorships=7,
        ),
    )
    s = r["surfaces"]["paymaster"]
    assert s["adapter_wired"] is True
    assert s["sponsorships"] == 7
    assert s["live_exec"] is True  # > 0 sponsorships


def test_paymaster_live_exec_false_without_sponsorships():
    """The whole point of distinguishing adapter_wired from
    live_exec: env can be commissioned + adapter wired, but if NO
    real user-op has been sponsored yet, live_exec stays False.
    Flips True after sp856 ships + real submission happens."""
    r = aggregate_phase5_status(
        paymaster_client=_FakePaymaster(
            commissioned=True, wired=True, sponsorships=0,
        ),
    )
    assert r["surfaces"]["paymaster"]["adapter_wired"] is True
    assert r["surfaces"]["paymaster"]["live_exec"] is False


def test_aerodrome_pool_address_passthrough():
    r = aggregate_phase5_status(
        aerodrome_client=_FakeAerodrome(
            configured=True, pool="0xABCD",
        ),
    )
    s = r["surfaces"]["aerodrome"]
    assert s["pool_configured"] is True
    assert s["pool_address"] == "0xABCD"


def test_aerodrome_unconfigured_clear_notes():
    """Unconfigured pool surfaces the Foundation Safe ceremony
    pointer in `notes` so operators see what to do next."""
    r = aggregate_phase5_status(
        aerodrome_client=_FakeAerodrome(configured=False),
    )
    assert r["surfaces"]["aerodrome"]["live_exec"] is False
    assert "Foundation Safe" in r["surfaces"]["aerodrome"]["notes"]

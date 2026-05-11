"""Sprint 277 — PaymasterClient.

Coinbase CDP paymaster adapter. Sponsors gas on behalf of users
so their FTNS transfers feel free (no ETH required). Per Vision
§14 "Crypto-UX adoption barrier" mitigation: users should never
see "transaction fee" or hold gas tokens.

PENDING_COMMISSION pattern (mirrors sprint 276 WaaS + offramp
quote): when CDP paymaster env keys are absent, sponsor_user_op
returns a preview without touching the bundler.

Per R-2026-05-08-1 (composer-only invariant): that rule is
scoped to coinbase_offramp_initiate. Gasless FTNS transfer is
intra-network user-authorized money movement (authorization
implicit via the user's WaaS-managed key) — same risk profile
as prsm_royalty_claim, which has an execute path. So sponsor_
user_op has a dry_run-by-default execute path here.
"""
from __future__ import annotations

import pytest

from prsm.economy.web3.paymaster_client import (
    PaymasterClient, SponsorshipResult,
)


class FakePaymasterBackend:
    """Test backend mirroring CDP paymaster API surface.

    Returns deterministic gas estimates + tx hashes. Production
    backend wraps the real Coinbase paymaster endpoint."""

    def __init__(self):
        self.estimate_calls = []
        self.submit_calls = []

    def estimate_gas(self, user_op):
        self.estimate_calls.append(user_op)
        return {
            "gas_estimate_wei": 100_000,
            "max_fee_per_gas_wei": 50_000_000,
        }

    def submit_sponsored(self, user_op):
        self.submit_calls.append(user_op)
        return {
            "user_op_hash": "0xdeadbeef",
            "tx_hash": "0xabc123",
            "sponsor_amount_wei": 5_000_000_000_000_000,
        }


def _user_op(amount=100):
    return {
        "sender": "0xabc",
        "to": "0xdef",
        "ftns_amount": amount,
        "kind": "ftns_transfer",
    }


# ── PENDING_COMMISSION ───────────────────────────────────


def test_from_env_returns_uncommissioned_when_keys_missing(
    monkeypatch,
):
    monkeypatch.delenv("COINBASE_CDP_PAYMASTER_ENDPOINT", raising=False)
    monkeypatch.delenv("COINBASE_CDP_PAYMASTER_API_KEY", raising=False)
    c = PaymasterClient.from_env()
    assert c is not None
    assert c.is_commissioned() is False


def test_sponsor_uncommissioned_returns_pending():
    c = PaymasterClient()
    result = c.sponsor_user_op(_user_op(), dry_run=True)
    assert result.status == "PENDING_COMMISSION"
    assert result.tx_hash is None
    assert result.gas_estimate_wei is None


def test_sponsor_uncommissioned_does_not_hit_backend():
    fake = FakePaymasterBackend()
    c = PaymasterClient(backend=fake)
    c.sponsor_user_op(_user_op(), dry_run=False)
    assert fake.estimate_calls == []
    assert fake.submit_calls == []


# ── COMMISSIONED dry_run (estimate-only) ─────────────────


def test_dry_run_returns_estimate_no_submit():
    fake = FakePaymasterBackend()
    c = PaymasterClient(
        endpoint="https://paymaster.example",
        api_key="key", backend=fake,
    )
    assert c.is_commissioned() is True
    result = c.sponsor_user_op(_user_op(), dry_run=True)
    assert result.status == "ESTIMATED"
    assert result.gas_estimate_wei == 100_000
    assert result.tx_hash is None
    assert len(fake.estimate_calls) == 1
    assert fake.submit_calls == []


# ── COMMISSIONED execute (real submit) ───────────────────


def test_execute_submits_and_returns_tx_hash():
    fake = FakePaymasterBackend()
    c = PaymasterClient(
        endpoint="https://paymaster.example",
        api_key="key", backend=fake,
    )
    result = c.sponsor_user_op(_user_op(), dry_run=False)
    assert result.status == "SUBMITTED"
    assert result.tx_hash == "0xabc123"
    assert result.user_op_hash == "0xdeadbeef"
    assert result.sponsor_amount_wei == 5_000_000_000_000_000
    assert len(fake.submit_calls) == 1


def test_execute_tracks_cumulative_sponsor_spend():
    fake = FakePaymasterBackend()
    c = PaymasterClient(
        endpoint="https://paymaster.example",
        api_key="key", backend=fake,
    )
    c.sponsor_user_op(_user_op(), dry_run=False)
    c.sponsor_user_op(_user_op(), dry_run=False)
    assert c.total_sponsored_wei() == 10_000_000_000_000_000
    assert c.total_sponsorships() == 2


def test_execute_backend_exception_returns_failed():
    class BoomBackend:
        def estimate_gas(self, op):
            return {"gas_estimate_wei": 0, "max_fee_per_gas_wei": 0}
        def submit_sponsored(self, op):
            raise RuntimeError("bundler down")
    c = PaymasterClient(
        endpoint="x", api_key="y", backend=BoomBackend(),
    )
    result = c.sponsor_user_op(_user_op(), dry_run=False)
    assert result.status == "FAILED"
    assert result.error is not None
    assert "bundler down" in result.error
    # Did NOT pollute the spend counter
    assert c.total_sponsored_wei() == 0


# ── Validation ───────────────────────────────────────────


def test_sponsor_validates_user_op():
    c = PaymasterClient(
        endpoint="x", api_key="y",
        backend=FakePaymasterBackend(),
    )
    with pytest.raises(ValueError):
        c.sponsor_user_op({}, dry_run=True)
    with pytest.raises(ValueError):
        c.sponsor_user_op({"sender": "0xabc"}, dry_run=True)


def test_sponsor_validates_user_op_must_be_dict():
    c = PaymasterClient(
        endpoint="x", api_key="y",
        backend=FakePaymasterBackend(),
    )
    with pytest.raises(ValueError):
        c.sponsor_user_op("not a dict", dry_run=True)


# ── Spend telemetry ──────────────────────────────────────


def test_total_sponsored_starts_at_zero():
    c = PaymasterClient()
    assert c.total_sponsored_wei() == 0
    assert c.total_sponsorships() == 0


def test_spend_summary_dict():
    fake = FakePaymasterBackend()
    c = PaymasterClient(
        endpoint="x", api_key="y", backend=fake,
    )
    c.sponsor_user_op(_user_op(), dry_run=False)
    summary = c.spend_summary()
    assert summary["sponsorships"] == 1
    assert summary["total_sponsored_wei"] == 5_000_000_000_000_000
    assert summary["commissioned"] is True


# ── SponsorshipResult round-trip ─────────────────────────


def test_sponsorship_result_to_dict():
    r = SponsorshipResult(
        status="SUBMITTED",
        tx_hash="0xabc",
        user_op_hash="0xdef",
        gas_estimate_wei=100,
        sponsor_amount_wei=500,
        error=None,
    )
    d = r.to_dict()
    assert d["status"] == "SUBMITTED"
    assert d["tx_hash"] == "0xabc"
    assert d["error"] is None

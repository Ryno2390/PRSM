"""Sprint 885 — rolling tier-limit total counts settled EXECUTES, not quotes.

sp285 built the rolling-total tier check counting all fiat kinds —
but it was written BEFORE onramp/execute existed (sp853), when
quotes were the only fiat event available to count. The result was
a double bug:

  1. Non-binding price-check QUOTES burned the user's tier limit —
     request 3 quotes of $400 and you're blocked from a real $400
     purchase, even though nothing moved. (The quote endpoint's own
     response says "nothing has moved on-chain or via fiat rails.")
  2. Real EXECUTES recorded nothing toward the rolling total, so
     actual settled volume never accumulated.

Net: the "rolling limit" limited how much you could QUOTE, not
transact — backwards.

Sp885 corrects it:
  - FiatComplianceRing.total_usd_for_user counts only
    {onramp_execute, offramp_execute} — quotes are excluded.
  - The funnel CONFIRMED transition records onramp_execute with the
    ACTUAL usdc_received (settled-volume model), wired via a new
    optional compliance_ring on make_on_confirmed_callback.

Regulatory limits (FinCEN MSB $1k/$10k) are on transacted volume,
not inquiries — so settled executes are the correct denominator.
"""
from __future__ import annotations

import time

from prsm.economy.web3.fiat_compliance_ring import FiatComplianceRing
from prsm.economy.web3.onramp_funnel import (
    OnrampFunnel,
    STATUS_CONFIRMED,
)
from prsm.economy.web3.onramp_to_swap_orchestrator import (
    make_on_confirmed_callback,
)


# ── Ring: rolling total counts executes, NOT quotes ──────────

def test_quotes_do_not_count_toward_rolling_total():
    """The load-bearing correction: non-binding quotes must NOT
    burn the tier limit."""
    r = FiatComplianceRing()
    now = time.time()
    r.record(
        kind="onramp_quote", user_id="alice",
        usd_amount=400.0, ftns_amount=400.0, status="OK",
        timestamp=now,
    )
    r.record(
        kind="offramp_quote", user_id="alice",
        usd_amount=400.0, ftns_amount=400.0, status="OK",
        timestamp=now,
    )
    # Quotes recorded (audit trail) but DON'T count toward the limit.
    assert r.total_usd_for_user("alice") == 0.0


def test_executes_count_toward_rolling_total():
    r = FiatComplianceRing()
    now = time.time()
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=300.0, ftns_amount=300.0, status="CONFIRMED",
        timestamp=now,
    )
    r.record(
        kind="offramp_execute", user_id="alice",
        usd_amount=200.0, ftns_amount=200.0, status="CONFIRMED",
        timestamp=now,
    )
    assert r.total_usd_for_user("alice") == 500.0


def test_mixed_quotes_and_executes_counts_only_executes():
    """A user who quotes $5000 (price-shopping) but executes $300
    has used $300 of their limit, not $5300."""
    r = FiatComplianceRing()
    now = time.time()
    for _ in range(5):
        r.record(
            kind="onramp_quote", user_id="alice",
            usd_amount=1000.0, ftns_amount=1000.0, status="OK",
            timestamp=now,
        )
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=300.0, ftns_amount=300.0, status="CONFIRMED",
        timestamp=now,
    )
    assert r.total_usd_for_user("alice") == 300.0


def test_execute_rolling_window_still_applies():
    """Executes outside the 24h window are excluded (window logic
    unchanged — only the kind filter changed)."""
    r = FiatComplianceRing()
    now = time.time()
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=900.0, ftns_amount=900.0, status="CONFIRMED",
        timestamp=now - 25 * 3600,  # outside 24h
    )
    r.record(
        kind="onramp_execute", user_id="alice",
        usd_amount=100.0, ftns_amount=100.0, status="CONFIRMED",
        timestamp=now - 3600,  # inside
    )
    assert r.total_usd_for_user("alice") == 100.0


def test_gasless_and_kyc_still_excluded():
    """Sp285's other exclusions hold — gasless (FTNS-denominated)
    + kyc (zero-amount) never count."""
    r = FiatComplianceRing()
    now = time.time()
    r.record(
        kind="gasless_transfer_execute", user_id="alice",
        usd_amount=999.0, ftns_amount=10.0, status="OK",
        timestamp=now,
    )
    r.record(
        kind="kyc_initiate", user_id="alice",
        usd_amount=0.0, ftns_amount=0.0, status="OK",
        timestamp=now,
    )
    assert r.total_usd_for_user("alice") == 0.0


# ── Funnel CONFIRMED records onramp_execute with usdc_received ──

class _FakeAerodrome:
    def is_configured(self): return False  # envelope deferred; we
    # only care about the compliance-ring recording here
    def quote_swap(self, *a, **k): return None


class _FakeBalance:
    def __init__(self, usdc):
        self.usdc = usdc
        self.usdc_units = int(usdc * 1e6)


class _FakeReader:
    def __init__(self, usdc): self._usdc = usdc
    def get_balances(self, addr): return _FakeBalance(self._usdc)


def test_confirmed_transition_records_onramp_execute(tmp_path):
    """When the sweep CONFIRMS an intent and a compliance_ring is
    wired, an onramp_execute is recorded with the ACTUAL
    usdc_received — so the rolling total reflects settled volume."""
    ring = FiatComplianceRing()
    funnel = OnrampFunnel(persist_dir=tmp_path / "funnel")
    intent = funnel.record_intent(
        user_id="alice",
        destination_address="0x" + "11" * 20,
        expected_usd=100.0,
        session_token="tok",
    )

    on_confirmed = make_on_confirmed_callback(
        funnel=funnel,
        aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ab" * 20,
        compliance_ring=ring,
    )
    # USDC arrived ($98.5 after Coinbase spread on a $100 expected).
    funnel.sweep(
        balance_reader=_FakeReader(98.5), on_confirmed=on_confirmed,
    )

    assert funnel.get_intent(intent.intent_id).status == (
        STATUS_CONFIRMED
    )
    # The ring recorded the ACTUAL received amount, not expected.
    assert ring.total_usd_for_user("alice") == 98.5


def test_confirmed_without_ring_does_not_crash(tmp_path):
    """compliance_ring is optional — omitting it must not break the
    confirm path (backward-compat with sp871/874 callers)."""
    funnel = OnrampFunnel(persist_dir=tmp_path / "funnel")
    funnel.record_intent(
        user_id="alice",
        destination_address="0x" + "11" * 20,
        expected_usd=100.0, session_token="tok",
    )
    on_confirmed = make_on_confirmed_callback(
        funnel=funnel,
        aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ab" * 20,
        # no compliance_ring
    )
    funnel.sweep(
        balance_reader=_FakeReader(98.5), on_confirmed=on_confirmed,
    )
    # No exception = pass; intent still confirmed.
    intents = funnel.list_intents(status=STATUS_CONFIRMED)
    assert len(intents) == 1


def test_confirmed_address_only_intent_records_empty_user(tmp_path):
    """An intent with no user_id (address-only onramp) records with
    empty user_id → doesn't aggregate against any user's limit."""
    ring = FiatComplianceRing()
    funnel = OnrampFunnel(persist_dir=tmp_path / "funnel")
    funnel.record_intent(
        user_id=None,
        destination_address="0x" + "11" * 20,
        expected_usd=100.0, session_token="tok",
    )
    on_confirmed = make_on_confirmed_callback(
        funnel=funnel,
        aerodrome_client=_FakeAerodrome(),
        ftns_address="0x" + "ab" * 20,
        compliance_ring=ring,
    )
    funnel.sweep(
        balance_reader=_FakeReader(98.5), on_confirmed=on_confirmed,
    )
    # Empty user_id → total_usd_for_user("") returns 0 for any user.
    assert ring.total_usd_for_user("") == 0.0

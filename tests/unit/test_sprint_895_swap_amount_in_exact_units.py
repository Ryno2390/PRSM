"""Sprint 895 — onramp→swap uses the EXACT received base units.

sp887 value finding ("float→uint conversion loss"). build_envelope_
for_intent computed the swap's on-chain amountIn from the FLOAT
``usdc_received`` (whole-token USDC) and re-multiplied by 1e6:

    amount_in_units = int(usdc_received * (10 ** 6))

But ``usdc_received`` was itself derived by DIVIDING the exact on-chain
base-unit balance by 1e6 (``WalletBalanceReader``: usdc = usdc_units /
10**6). So the path is  int base units → float → ×1e6 → int  — a
round-trip through float64 that loses a base unit for ~1.2% of values
(e.g. 8_000_001 → 8.000001 → 8_000_000). The swap then under-spends by
the lost dust, systematically across swaps.

The fix is unforced: the funnel ALREADY persists the exact integer in
``usdc_received_units`` (set from bal.usdc_units at the CONFIRMED
transition). Use it directly for amountIn; fall back to the float-
derived value only for legacy records that predate the units field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from prsm.economy.web3.onramp_to_swap_orchestrator import (
    build_envelope_for_intent,
)


@dataclass
class _Intent:
    intent_id: str = "onramp_test"
    destination_address: str = "0x" + "11" * 20
    user_id: Optional[str] = "alice"
    usdc_received: float = 0.0
    usdc_received_units: int = 0


@dataclass
class _Quote:
    amount_out: int = 5 * 10 ** 18  # 5 FTNS
    price_impact_bps: int = 10
    fee_bps: int = 30


class _FakeAerodrome:
    """Configured pool that echoes the amount_in it was quoted on,
    so the test can assert the EXACT units flowed through."""

    def __init__(self):
        self.quoted_amount_in = None

    def is_configured(self):
        return True

    def quote_swap(self, *, amount_in, token_in):
        self.quoted_amount_in = amount_in
        return _Quote()


_FTNS = "0x" + "ab" * 20


# ── The fix: exact units, no float round-trip ────────────────

def test_amount_in_uses_exact_units_not_float_roundtrip():
    """8_000_001 base units (8.000001 USDC). The float round-trip
    yields 8_000_000 (loses 1 unit); the exact-units path must
    produce 8_000_001."""
    aero = _FakeAerodrome()
    intent = _Intent(
        usdc_received=8.000001,        # = 8_000_001 / 1e6 (lossy)
        usdc_received_units=8_000_001,  # the authoritative integer
    )
    env = build_envelope_for_intent(
        intent, aerodrome_client=aero, ftns_address=_FTNS,
    )
    assert env is not None
    assert env["args"]["amountIn"] == 8_000_001
    assert env["quote"]["amount_in_units"] == 8_000_001


def test_quote_taken_on_exact_units():
    """The QUOTE itself was lossy pre-fix (quoting on 8_000_000
    instead of 8_000_001). The pool must be quoted on the exact
    received units."""
    aero = _FakeAerodrome()
    intent = _Intent(
        usdc_received=8.000001, usdc_received_units=8_000_001,
    )
    build_envelope_for_intent(
        intent, aerodrome_client=aero, ftns_address=_FTNS,
    )
    assert aero.quoted_amount_in == 8_000_001


# ── Legacy records (no units field) fall back to the float ───

def test_falls_back_to_float_when_units_absent():
    """A pre-units intent (usdc_received_units == 0) must still
    build an envelope from the float-derived amount."""
    aero = _FakeAerodrome()
    intent = _Intent(usdc_received=8.5, usdc_received_units=0)
    env = build_envelope_for_intent(
        intent, aerodrome_client=aero, ftns_address=_FTNS,
    )
    assert env is not None
    assert env["args"]["amountIn"] == 8_500_000


def test_falls_back_when_units_attribute_missing(object_intent=None):
    """An intent object lacking the attribute entirely (oldest
    persisted shape) still works via getattr default."""
    class _LegacyIntent:
        intent_id = "legacy"
        destination_address = "0x" + "22" * 20
        user_id = "bob"
        usdc_received = 12.0
        # no usdc_received_units attribute at all

    aero = _FakeAerodrome()
    env = build_envelope_for_intent(
        _LegacyIntent(), aerodrome_client=aero, ftns_address=_FTNS,
    )
    assert env is not None
    assert env["args"]["amountIn"] == 12_000_000


# ── Regression: zero received → None (no swap) ───────────────

def test_zero_received_returns_none():
    aero = _FakeAerodrome()
    intent = _Intent(usdc_received=0.0, usdc_received_units=0)
    assert build_envelope_for_intent(
        intent, aerodrome_client=aero, ftns_address=_FTNS,
    ) is None

"""Sprint 901 — Aerodrome pool go-live verification harness.

The Aerodrome USDC↔FTNS pool seed (Vision gantt 2026-06-15) is the
single highest-leverage external blocker: it unblocks the entire
fiat→FTNS user journey (the onramp→swap leg converts USDC→FTNS through
this pool). The ceremony itself is a Foundation-Safe multi-sig action;
sp875/876 ship the tx-batch builder + runbook, but there was NO
post-seed VERIFIER — the missing final step of the ceremony lifecycle
(build batch → execute via Safe → VERIFY it actually worked).

This harness is that verifier. The instant the multi-sig executes and
the operator sets AERODROME_USDC_FTNS_POOL_ADDRESS, running it confirms
go/no-go: the pool exists, holds the right token pair, is actually
seeded (non-zero reserves), is the volatile pool the orchestrator
expects, reports the opening FTNS price, the live swap quoter returns a
usable quote, and the full onramp→swap envelope builds end-to-end. It
also prepares the exact executable first-swap envelope so the operator
can submit the first real onramp→swap→FTNS immediately.

Read-only + offline-testable now (against a fake client); runs live the
moment the pool is seeded. NO money movement — verification + tx prep.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from prsm.economy.web3.aerodrome_client import (
    AerodromePoolState,
    AerodromeQuote,
)
from prsm.economy.web3.aerodrome_pool_ceremony import (
    MAINNET_CONFIG,
    USDC_BASE_MAINNET,
    FTNS_BASE_MAINNET,
)
from prsm.economy.web3.go_live_verification import (
    run_go_live_verification,
    GoLiveReport,
)

# $0.25/FTNS opening price: 500k USDC (6dec) + 2M FTNS (18dec).
_SEED_USDC_UNITS = 500_000 * 10 ** 6
_SEED_FTNS_UNITS = 2_000_000 * 10 ** 18
_PROBE_USDC = 1_000_000  # $1.00 in USDC base units


def _seeded_state(
    *, token0=USDC_BASE_MAINNET, token1=FTNS_BASE_MAINNET,
    r0=_SEED_USDC_UNITS, r1=_SEED_FTNS_UNITS, stable=False,
):
    return AerodromePoolState(
        pool_address="0xPOOL",
        token0=token0, token1=token1,
        reserve0=r0, reserve1=r1,
        stable=stable, fee_bps=30,
        total_supply=1_000_000 * 10 ** 18, block_number=12345,
    )


def _quote(amount_out=3_960_000 * 10 ** 12):  # ~3.96 FTNS for $1
    return AerodromeQuote(
        amount_in=_PROBE_USDC, token_in=USDC_BASE_MAINNET,
        token_out=FTNS_BASE_MAINNET, amount_out=amount_out,
        price_impact_bps=15, route="aerodrome", fee_bps=30,
    )


class _FakeAero:
    def __init__(self, *, configured=True, state=None, quote=None):
        self._configured = configured
        self._state = state
        self._quote = quote
        self.pool_address = "0xPOOL" if configured else None

    def is_configured(self):
        return self._configured

    def get_pool_state(self, pool_address=None):
        return self._state

    def quote_swap(self, amount_in, token_in, pool_address=None):
        return self._quote


def _run(client, **kw):
    return run_go_live_verification(
        client, MAINNET_CONFIG,
        probe_usdc_units=_PROBE_USDC, **kw,
    )


def _status(report, check):
    f = next((f for f in report.findings if f.check == check), None)
    return f.status if f else None


# ── go=True only when the pool is genuinely seeded + swappable ─

def test_fully_seeded_pool_passes_go_live():
    client = _FakeAero(state=_seeded_state(), quote=_quote())
    report = _run(client)
    assert isinstance(report, GoLiveReport)
    assert report.go is True, report.to_dict()
    assert _status(report, "pool_configured") == "PASS"
    assert _status(report, "pool_seeded") == "PASS"
    assert _status(report, "token_pair") == "PASS"
    assert _status(report, "volatile_pool") == "PASS"
    assert _status(report, "swap_quote") == "PASS"
    assert _status(report, "onramp_swap_envelope") == "PASS"
    # The executable first-swap envelope is prepared for the operator.
    assert report.prepared_envelope is not None
    assert report.prepared_envelope["args"]["amountIn"] == _PROBE_USDC


def test_opening_price_reported():
    client = _FakeAero(state=_seeded_state(), quote=_quote())
    report = _run(client)
    price = next(
        f for f in report.findings if f.check == "opening_price"
    )
    # 500k USDC / 2M FTNS = $0.25/FTNS.
    assert "0.25" in price.detail


# ── Each not-ready condition blocks go-live (FAIL) ───────────

def test_pool_not_configured_blocks():
    report = _run(_FakeAero(configured=False))
    assert report.go is False
    assert _status(report, "pool_configured") == "FAIL"


def test_pool_state_unreadable_blocks():
    report = _run(_FakeAero(state=None, quote=_quote()))
    assert report.go is False
    assert _status(report, "pool_seeded") == "FAIL"


def test_zero_reserves_not_seeded_blocks():
    client = _FakeAero(state=_seeded_state(r0=0, r1=0), quote=_quote())
    report = _run(client)
    assert report.go is False
    assert _status(report, "pool_seeded") == "FAIL"


def test_wrong_token_pair_blocks():
    client = _FakeAero(
        state=_seeded_state(token1="0x" + "de" * 20), quote=_quote(),
    )
    report = _run(client)
    assert report.go is False
    assert _status(report, "token_pair") == "FAIL"


def test_stable_pool_blocks():
    # The orchestrator builds swapExactTokensForTokens with
    # stable=False; a seeded STABLE pool means the swap route is wrong.
    client = _FakeAero(state=_seeded_state(stable=True), quote=_quote())
    report = _run(client)
    assert report.go is False
    assert _status(report, "volatile_pool") == "FAIL"


def test_swap_quote_zero_out_blocks():
    client = _FakeAero(state=_seeded_state(), quote=_quote(amount_out=0))
    report = _run(client)
    assert report.go is False
    assert _status(report, "swap_quote") == "FAIL"


def test_swap_quote_unavailable_blocks():
    client = _FakeAero(state=_seeded_state(), quote=None)
    report = _run(client)
    assert report.go is False
    assert _status(report, "swap_quote") == "FAIL"


# ── Token-pair check is order-insensitive ────────────────────

def test_token_pair_reversed_order_passes():
    client = _FakeAero(
        state=_seeded_state(
            token0=FTNS_BASE_MAINNET, token1=USDC_BASE_MAINNET,
            r0=_SEED_FTNS_UNITS, r1=_SEED_USDC_UNITS,
        ),
        quote=_quote(),
    )
    report = _run(client)
    assert _status(report, "token_pair") == "PASS"
    assert _status(report, "pool_seeded") == "PASS"


# ── Expected-reserves cross-check (WARN, not FAIL) ───────────

def test_expected_reserves_mismatch_warns_not_blocks():
    """If the operator passes the seed amounts, a mismatch is a WARN
    (the pool may have legitimately traded) — not a hard block."""
    client = _FakeAero(state=_seeded_state(), quote=_quote())
    report = run_go_live_verification(
        client, MAINNET_CONFIG, probe_usdc_units=_PROBE_USDC,
        expected_usdc_units=_SEED_USDC_UNITS * 2,  # mismatch
        expected_ftns_units=_SEED_FTNS_UNITS,
    )
    assert _status(report, "reserves_match_seed") == "WARN"
    assert report.go is True  # WARN does not block


# ── Structured output ────────────────────────────────────────

def test_report_to_dict_shape():
    report = _run(_FakeAero(state=_seeded_state(), quote=_quote()))
    d = report.to_dict()
    assert d["go"] is True
    assert isinstance(d["findings"], list)
    assert all(
        {"check", "status", "detail"} <= set(f) for f in d["findings"]
    )

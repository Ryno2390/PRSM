"""Unit tests for prsm.economy.payments.withdrawal_orchestrator.

Per docs/2026-04-22-phase5-fiat-onramp-design-plan.md §6 Task 5.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pytest

from prsm.economy.payments.withdrawal_orchestrator import (
    InMemoryWithdrawalStore,
    KycResult,
    PayoutResult,
    SqliteWithdrawalStore,
    SwapResult,
    Withdrawal,
    WithdrawalNotFoundError,
    WithdrawalOrchestrator,
    WithdrawalState,
    WithdrawalStateError,
)


# -----------------------------------------------------------------------------
# Stub services
# -----------------------------------------------------------------------------


@dataclass
class StubKyc:
    passed: bool = True
    reason: str = ""
    calls: List[str] = field(default_factory=list)

    def check(self, user_id):
        self.calls.append(user_id)
        return KycResult(passed=self.passed, reason=self.reason or None)


@dataclass
class StubOracle:
    cents: int = 12_500  # default quote = $125.00 per call
    raise_on_call: bool = False

    def quote_ftns_to_usd_cents(self, ftns_wei):
        if self.raise_on_call:
            raise RuntimeError("oracle unavailable")
        return self.cents


@dataclass
class StubSwap:
    order_id: str = "coinbase-order-1"
    usdc: int = 125_000_000  # $125
    raise_on_call: bool = False

    def swap_ftns_to_usdc(self, ftns_wei, *, expected_usd_cents):
        if self.raise_on_call:
            raise RuntimeError("swap failed")
        return SwapResult(order_id=self.order_id, usdc_received=self.usdc)


@dataclass
class StubPayout:
    payout_id: str = "stripe-po-1"
    raise_on_call: bool = False

    def payout_usd_to_bank(self, *, amount_cents, destination_bank_token):
        if self.raise_on_call:
            raise RuntimeError("payout declined")
        return PayoutResult(payout_id=self.payout_id)


@pytest.fixture
def stubs():
    return StubKyc(), StubOracle(), StubSwap(), StubPayout()


@pytest.fixture
def clock():
    return [1_700_000_000.0]


@pytest.fixture
def orchestrator(stubs, clock):
    kyc, oracle, swap, payout = stubs
    return WithdrawalOrchestrator(
        kyc=kyc,
        oracle=oracle,
        swap=swap,
        payout=payout,
        store=InMemoryWithdrawalStore(),
        clock=lambda: clock[0],
        id_generator=lambda: "wd-test-id",
    )


def _advance_until(orch, wid, target_state, max_steps=10):
    for _ in range(max_steps):
        w = orch.advance(wid)
        if w.state == target_state:
            return w
    raise AssertionError(f"did not reach {target_state}: got {w.state}")


# -----------------------------------------------------------------------------
# Happy path
# -----------------------------------------------------------------------------


def test_request_creates_requested_withdrawal(orchestrator):
    w = orchestrator.request("alice", 10**18, "bank-token-xyz")
    assert w.state is WithdrawalState.REQUESTED
    assert w.user_id == "alice"
    assert w.ftns_wei == 10**18
    assert w.destination_bank_token == "bank-token-xyz"


def test_advance_full_happy_path(orchestrator):
    w = orchestrator.request("alice", 10**18, "bank-token")
    w = orchestrator.advance(w.id)
    assert w.state is WithdrawalState.KYC_CHECK

    w = orchestrator.advance(w.id)
    assert w.state is WithdrawalState.ORACLE_QUOTE

    w = orchestrator.advance(w.id)
    # oracle + swap are coupled in one step → SWAP_EXECUTED.
    assert w.state is WithdrawalState.SWAP_EXECUTED
    assert w.usd_quote_cents == 12_500
    assert w.coinbase_order_id == "coinbase-order-1"

    w = orchestrator.advance(w.id)
    assert w.state is WithdrawalState.STRIPE_PAYOUT
    assert w.stripe_payout_id == "stripe-po-1"

    w = orchestrator.advance(w.id)
    assert w.state is WithdrawalState.COMPLETED


def test_request_persists_record(orchestrator):
    w = orchestrator.request("alice", 10**18, "bank-token")
    loaded = orchestrator.get(w.id)
    assert loaded == w


# -----------------------------------------------------------------------------
# Failure branches
# -----------------------------------------------------------------------------


def test_kyc_failure_terminates(orchestrator, stubs):
    kyc, *_ = stubs
    kyc.passed = False
    kyc.reason = "sanctions_hit"

    w = orchestrator.request("alice", 10**18, "bank-token")
    orchestrator.advance(w.id)  # → KYC_CHECK
    w = orchestrator.advance(w.id)  # → KYC_FAILED
    assert w.state is WithdrawalState.KYC_FAILED
    assert w.error == "sanctions_hit"


def test_kyc_exception_is_captured(orchestrator, stubs):
    kyc, *_ = stubs
    kyc.passed = True  # irrelevant once exception path fires

    class Exploding:
        def check(self, user_id):
            raise RuntimeError("vendor down")

    orchestrator._kyc = Exploding()
    w = orchestrator.request("alice", 10**18, "bank-token")
    orchestrator.advance(w.id)
    w = orchestrator.advance(w.id)
    assert w.state is WithdrawalState.KYC_FAILED
    assert "vendor down" in (w.error or "")


def test_oracle_failure_terminates(orchestrator, stubs):
    _, oracle, *_ = stubs
    oracle.raise_on_call = True
    w = orchestrator.request("alice", 10**18, "bank-token")
    orchestrator.advance(w.id)  # REQUESTED → KYC_CHECK
    orchestrator.advance(w.id)  # → ORACLE_QUOTE
    w = orchestrator.advance(w.id)  # oracle raises → QUOTE_FAILED
    assert w.state is WithdrawalState.QUOTE_FAILED


def test_oracle_nonpositive_quote_rejected(orchestrator, stubs):
    _, oracle, *_ = stubs
    oracle.cents = 0
    w = orchestrator.request("alice", 10**18, "bank-token")
    orchestrator.advance(w.id)
    orchestrator.advance(w.id)
    w = orchestrator.advance(w.id)
    assert w.state is WithdrawalState.QUOTE_FAILED
    assert w.error == "nonpositive_quote"


def test_swap_failure_after_quote_stored(orchestrator, stubs):
    _, _, swap, _ = stubs
    swap.raise_on_call = True
    w = orchestrator.request("alice", 10**18, "bank-token")
    orchestrator.advance(w.id)
    orchestrator.advance(w.id)
    w = orchestrator.advance(w.id)
    assert w.state is WithdrawalState.SWAP_FAILED
    # Quote preserved for audit.
    assert w.usd_quote_cents == 12_500


def test_payout_failure_terminates(orchestrator, stubs):
    *_, payout = stubs
    payout.raise_on_call = True
    w = orchestrator.request("alice", 10**18, "bank-token")
    _advance_until(orchestrator, w.id, WithdrawalState.SWAP_EXECUTED)
    w = orchestrator.advance(w.id)
    assert w.state is WithdrawalState.PAYOUT_FAILED


# -----------------------------------------------------------------------------
# Idempotency + terminal semantics
# -----------------------------------------------------------------------------


def test_advance_on_completed_is_idempotent(orchestrator):
    w = orchestrator.request("alice", 10**18, "bank-token")
    _advance_until(orchestrator, w.id, WithdrawalState.COMPLETED)
    w_before = orchestrator.get(w.id)
    w_after = orchestrator.advance(w.id)
    assert w_before == w_after


def test_advance_on_terminal_failure_is_idempotent(orchestrator, stubs):
    kyc, *_ = stubs
    kyc.passed = False
    w = orchestrator.request("alice", 10**18, "bank-token")
    orchestrator.advance(w.id)
    orchestrator.advance(w.id)  # → KYC_FAILED
    w_before = orchestrator.get(w.id)
    w_after = orchestrator.advance(w.id)
    assert w_before == w_after


# -----------------------------------------------------------------------------
# Cancel semantics
# -----------------------------------------------------------------------------


def test_cancel_from_requested_allowed(orchestrator):
    w = orchestrator.request("alice", 10**18, "bank-token")
    w = orchestrator.cancel(w.id)
    assert w.state is WithdrawalState.CANCELED


def test_cancel_from_kyc_check_allowed(orchestrator):
    w = orchestrator.request("alice", 10**18, "bank-token")
    orchestrator.advance(w.id)  # → KYC_CHECK
    w = orchestrator.cancel(w.id)
    assert w.state is WithdrawalState.CANCELED


def test_cancel_after_swap_is_rejected(orchestrator):
    w = orchestrator.request("alice", 10**18, "bank-token")
    _advance_until(orchestrator, w.id, WithdrawalState.SWAP_EXECUTED)
    with pytest.raises(WithdrawalStateError):
        orchestrator.cancel(w.id)


def test_cancel_completed_is_rejected(orchestrator):
    w = orchestrator.request("alice", 10**18, "bank-token")
    _advance_until(orchestrator, w.id, WithdrawalState.COMPLETED)
    with pytest.raises(WithdrawalStateError):
        orchestrator.cancel(w.id)


# -----------------------------------------------------------------------------
# Validation + lookup
# -----------------------------------------------------------------------------


def test_request_rejects_non_positive_amount(orchestrator):
    with pytest.raises(Exception):
        orchestrator.request("alice", 0, "bank-token")


def test_unknown_withdrawal_raises(orchestrator):
    with pytest.raises(WithdrawalNotFoundError):
        orchestrator.get("no-such-id")


def test_unknown_withdrawal_advance_raises(orchestrator):
    with pytest.raises(WithdrawalNotFoundError):
        orchestrator.advance("no-such-id")


# -----------------------------------------------------------------------------
# SQLite store round-trip
# -----------------------------------------------------------------------------


def test_sqlite_store_round_trip(stubs, clock):
    kyc, oracle, swap, payout = stubs
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "withdrawals.sqlite"

        store = SqliteWithdrawalStore(db_path)
        orch = WithdrawalOrchestrator(
            kyc=kyc,
            oracle=oracle,
            swap=swap,
            payout=payout,
            store=store,
            clock=lambda: clock[0],
            id_generator=lambda: "persist-id",
        )

        w = orch.request("alice", 10**18, "bank-token")
        _advance_until(orch, w.id, WithdrawalState.SWAP_EXECUTED)

        # Reopen store against same file, verify state round-trips.
        store2 = SqliteWithdrawalStore(db_path)
        reloaded = store2.load("persist-id")
        assert reloaded is not None
        assert reloaded.state is WithdrawalState.SWAP_EXECUTED
        assert reloaded.usd_quote_cents == 12_500
        assert reloaded.coinbase_order_id == "coinbase-order-1"


def test_sqlite_update_is_upsert(stubs, clock):
    kyc, oracle, swap, payout = stubs
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "withdrawals.sqlite"
        store = SqliteWithdrawalStore(db_path)
        orch = WithdrawalOrchestrator(
            kyc=kyc,
            oracle=oracle,
            swap=swap,
            payout=payout,
            store=store,
            clock=lambda: clock[0],
            id_generator=lambda: "upsert-id",
        )

        w = orch.request("alice", 10**18, "bank-token")
        orch.advance(w.id)
        orch.advance(w.id)
        final = orch.advance(w.id)  # → SWAP_EXECUTED

        reloaded = store.load("upsert-id")
        assert reloaded is not None
        assert reloaded.state == final.state
        assert reloaded.updated_at == final.updated_at

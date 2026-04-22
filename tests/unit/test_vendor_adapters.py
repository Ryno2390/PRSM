"""Unit tests for Phase 5 vendor adapter scaffolds.

Per docs/2026-04-22-phase5-fiat-onramp-design-plan.md §6 Tasks 2-4.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from prsm.economy.payments.vendor_adapters import (
    CoinbaseOrderRejected,
    CoinbaseSlippageExceeded,
    CoinbaseSwapService,
    KycProviderUnavailable,
    KycServiceImpl,
    KycSession,
    KycStatus,
    StripeDestinationRejected,
    StripeError,
    StripeInsufficientFunds,
    StripePayoutService,
    StripePayoutStatus,
    StubCoinbaseClient,
    StubKycVendor,
    StubStripeClient,
)
from prsm.economy.payments.withdrawal_orchestrator import (
    InMemoryWithdrawalStore,
    WithdrawalOrchestrator,
    WithdrawalState,
)


# =============================================================================
# KYC — Task 2
# =============================================================================


class TestStubKycVendor:
    def test_create_session_starts_pending(self):
        vendor = StubKycVendor()
        session = vendor.create_session("alice")
        assert session.status is KycStatus.PENDING
        assert session.user_id == "alice"

    def test_check_status_verified_on_second_poll(self):
        vendor = StubKycVendor()
        session = vendor.create_session("alice")
        first = vendor.check_status(session.session_id)
        assert first.status is KycStatus.PENDING
        second = vendor.check_status(session.session_id)
        assert second.status is KycStatus.VERIFIED

    def test_fail_prefix_reaches_failed_state(self):
        vendor = StubKycVendor()
        session = vendor.create_session("fail-bob")
        result = vendor.check_status(session.session_id)
        assert result.status is KycStatus.FAILED
        assert result.failure_reason == "doc_rejected"

    def test_pending_prefix_never_advances(self):
        vendor = StubKycVendor()
        session = vendor.create_session("pending-carol")
        for _ in range(5):
            result = vendor.check_status(session.session_id)
            assert result.status is KycStatus.PENDING

    def test_sanctioned_prefix_triggers_is_sanctioned(self):
        vendor = StubKycVendor()
        assert vendor.is_sanctioned("sanctioned-dave") is True
        assert vendor.is_sanctioned("alice") is False

    def test_expired_session_flips_to_expired(self):
        clock = [1000.0]
        vendor = StubKycVendor(_now=lambda: clock[0])
        session = vendor.create_session("alice")
        clock[0] += vendor.session_ttl_seconds + 1
        result = vendor.check_status(session.session_id)
        assert result.status is KycStatus.EXPIRED

    def test_force_unavailable_raises(self):
        vendor = StubKycVendor()
        vendor.force_unavailable(True)
        with pytest.raises(KycProviderUnavailable):
            vendor.create_session("alice")
        with pytest.raises(KycProviderUnavailable):
            vendor.is_sanctioned("alice")


class TestKycServiceImpl:
    def test_sanctions_returns_passed_false_without_session(self):
        vendor = StubKycVendor()
        service = KycServiceImpl(vendor)
        result = service.check("sanctioned-eve")
        assert result.passed is False
        assert result.reason == "sanctions_hit"

    def test_happy_path_returns_passed_true(self):
        vendor = StubKycVendor()
        service = KycServiceImpl(vendor)
        result = service.check("alice")
        assert result.passed is True
        assert result.reason is None

    def test_failed_session_returns_passed_false_with_reason(self):
        vendor = StubKycVendor()
        service = KycServiceImpl(vendor)
        result = service.check("fail-bob")
        assert result.passed is False
        assert result.reason == "doc_rejected"

    def test_polling_exhausted_returns_passed_false(self):
        """pending-* users never verify; service exhausts poll attempts."""
        vendor = StubKycVendor()
        service = KycServiceImpl(vendor, max_poll_attempts=3)
        result = service.check("pending-carol")
        assert result.passed is False
        assert result.reason == "polling_exhausted"

    def test_vendor_unavailable_propagates(self):
        vendor = StubKycVendor()
        vendor.force_unavailable(True)
        service = KycServiceImpl(vendor)
        with pytest.raises(KycProviderUnavailable):
            service.check("alice")


# =============================================================================
# Stripe — Task 3
# =============================================================================


class TestStubStripeClient:
    def test_create_payout_default_paid(self):
        client = StubStripeClient()
        payout = client.create_payout(
            amount_cents=5000,
            destination_token="bank_happy",
            idempotency_key="k1",
        )
        assert payout.status is StripePayoutStatus.PAID
        assert payout.amount_cents == 5000

    def test_bank_reject_raises_destination_rejected(self):
        client = StubStripeClient()
        with pytest.raises(StripeDestinationRejected):
            client.create_payout(
                amount_cents=1000,
                destination_token="bank_reject",
                idempotency_key="k2",
            )

    def test_no_funds_raises_insufficient(self):
        client = StubStripeClient()
        with pytest.raises(StripeInsufficientFunds):
            client.create_payout(
                amount_cents=1000,
                destination_token="bank_no_funds",
                idempotency_key="k3",
            )

    def test_idempotency_key_dedups(self):
        """Duplicate key returns the SAME payout — Stripe's
        real Idempotency-Key contract."""
        client = StubStripeClient()
        p1 = client.create_payout(
            amount_cents=1000, destination_token="bank_x", idempotency_key="same"
        )
        p2 = client.create_payout(
            amount_cents=1000, destination_token="bank_x", idempotency_key="same"
        )
        assert p1.payout_id == p2.payout_id

    def test_bank_fail_transitions_to_failed_on_get(self):
        """create_payout succeeds, get_payout flips to FAILED
        (async bank rejection)."""
        client = StubStripeClient()
        p1 = client.create_payout(
            amount_cents=1000, destination_token="bank_fail", idempotency_key="k4"
        )
        assert p1.status is StripePayoutStatus.PAID
        p2 = client.get_payout(p1.payout_id)
        assert p2.status is StripePayoutStatus.FAILED
        assert p2.failure_reason == "async_bank_rejection"


class TestStripePayoutService:
    def test_payout_returns_payout_id_on_success(self):
        client = StubStripeClient()
        service = StripePayoutService(client)
        result = service.payout_usd_to_bank(
            amount_cents=10_000,
            destination_bank_token="bank_happy",
        )
        assert result.payout_id.startswith("po_")

    def test_payout_destination_rejected_raises(self):
        client = StubStripeClient()
        service = StripePayoutService(client)
        with pytest.raises(StripeDestinationRejected):
            service.payout_usd_to_bank(
                amount_cents=10_000, destination_bank_token="bank_reject"
            )

    def test_payout_insufficient_funds_raises(self):
        client = StubStripeClient()
        service = StripePayoutService(client)
        with pytest.raises(StripeInsufficientFunds):
            service.payout_usd_to_bank(
                amount_cents=10_000, destination_bank_token="bank_no_funds"
            )


# =============================================================================
# Coinbase — Task 4
# =============================================================================


class TestStubCoinbaseClient:
    def test_market_sell_fills_at_slipped_price(self):
        client = StubCoinbaseClient(
            mid_price_usd=Decimal("2.00"),
            actual_slippage_bps=100,  # 1%
        )
        order = client.place_market_sell(
            base_amount_wei=10**18,  # 1 FTNS
            max_slippage_bps=200,
        )
        assert order.status == "filled"
        # 1 FTNS * $2 * 0.99 = $1.98 → 1,980,000 micro.
        assert order.quote_amount_micro == 1_980_000

    def test_market_sell_rejects_when_actual_exceeds_max_slippage(self):
        client = StubCoinbaseClient(actual_slippage_bps=300)
        with pytest.raises(CoinbaseSlippageExceeded):
            client.place_market_sell(
                base_amount_wei=10**18,
                max_slippage_bps=200,
            )

    def test_reject_next_order_flag(self):
        client = StubCoinbaseClient()
        client.reject_next_order = True
        with pytest.raises(CoinbaseOrderRejected):
            client.place_market_sell(base_amount_wei=10**18, max_slippage_bps=200)
        # Flag clears after one rejection.
        order = client.place_market_sell(base_amount_wei=10**18, max_slippage_bps=200)
        assert order.status == "filled"


class TestCoinbaseSwapService:
    def test_swap_returns_swap_result(self):
        client = StubCoinbaseClient(
            mid_price_usd=Decimal("2.00"),
            actual_slippage_bps=10,
        )
        service = CoinbaseSwapService(client)
        # Expected: 1 FTNS × $2 = $2.00 = 200 cents.
        result = service.swap_ftns_to_usdc(
            ftns_wei=10**18, expected_usd_cents=200
        )
        assert result.order_id.startswith("cb_")
        # $2 × 0.999 = $1.998 → 1,998,000 micro.
        assert result.usdc_received == 1_998_000

    def test_swap_raises_when_drift_exceeds_threshold(self):
        """Exchange fill < oracle quote by >5% triggers drift exception."""
        client = StubCoinbaseClient(
            mid_price_usd=Decimal("2.00"),
            actual_slippage_bps=10,
        )
        service = CoinbaseSwapService(
            client, max_expected_quote_drift_bps=100  # 1% tolerance
        )
        # Expected quote: $10 = 1000 cents. Actual fill will be ~$9.99.
        # Drift: |999 - 1000| / 1000 = 0.1% — within tolerance, should pass.
        service.swap_ftns_to_usdc(ftns_wei=5 * 10**18, expected_usd_cents=1000)

        # Now push expected way above reality — forces drift > threshold.
        with pytest.raises(CoinbaseSlippageExceeded):
            service.swap_ftns_to_usdc(
                ftns_wei=5 * 10**18, expected_usd_cents=2000  # 100% off
            )

    def test_swap_propagates_order_rejection(self):
        client = StubCoinbaseClient()
        client.reject_next_order = True
        service = CoinbaseSwapService(client)
        with pytest.raises(CoinbaseOrderRejected):
            service.swap_ftns_to_usdc(
                ftns_wei=10**18, expected_usd_cents=200
            )


# =============================================================================
# End-to-end: WithdrawalOrchestrator with real adapters
# =============================================================================


class _FixedOracle:
    def __init__(self, cents: int) -> None:
        self._cents = cents

    def quote_ftns_to_usd_cents(self, ftns_wei: int) -> int:
        return self._cents


class TestOrchestratorWithRealAdapters:
    def test_full_pipeline_happy_path(self):
        """WithdrawalOrchestrator advances REQUESTED → KYC_CHECK →
        ORACLE_QUOTE → SWAP_EXECUTED → STRIPE_PAYOUT → COMPLETED
        using the three real adapter scaffolds."""
        kyc = KycServiceImpl(StubKycVendor())
        oracle = _FixedOracle(cents=200)   # $2 per 1 FTNS
        swap = CoinbaseSwapService(
            StubCoinbaseClient(mid_price_usd=Decimal("2.00"), actual_slippage_bps=10)
        )
        payout = StripePayoutService(StubStripeClient())
        orch = WithdrawalOrchestrator(
            kyc=kyc, oracle=oracle, swap=swap, payout=payout,
            store=InMemoryWithdrawalStore(),
        )

        w = orch.request("alice", 10**18, "bank_happy")
        # REQUESTED → KYC_CHECK
        w = orch.advance(w.id)
        assert w.state is WithdrawalState.KYC_CHECK
        # KYC_CHECK → ORACLE_QUOTE (KycServiceImpl polls StubKycVendor internally)
        w = orch.advance(w.id)
        assert w.state is WithdrawalState.ORACLE_QUOTE
        # ORACLE_QUOTE → SWAP_EXECUTED (oracle + swap coupled in one tick)
        w = orch.advance(w.id)
        assert w.state is WithdrawalState.SWAP_EXECUTED
        assert w.usd_quote_cents == 200
        assert w.coinbase_order_id and w.coinbase_order_id.startswith("cb_")
        # SWAP_EXECUTED → STRIPE_PAYOUT
        w = orch.advance(w.id)
        assert w.state is WithdrawalState.STRIPE_PAYOUT
        assert w.stripe_payout_id and w.stripe_payout_id.startswith("po_")
        # STRIPE_PAYOUT → COMPLETED
        w = orch.advance(w.id)
        assert w.state is WithdrawalState.COMPLETED

    def test_sanctions_terminates_at_kyc(self):
        kyc = KycServiceImpl(StubKycVendor())
        oracle = _FixedOracle(cents=200)
        swap = CoinbaseSwapService(StubCoinbaseClient())
        payout = StripePayoutService(StubStripeClient())
        orch = WithdrawalOrchestrator(
            kyc=kyc, oracle=oracle, swap=swap, payout=payout,
            store=InMemoryWithdrawalStore(),
        )

        w = orch.request("sanctioned-eve", 10**18, "bank_happy")
        orch.advance(w.id)  # → KYC_CHECK
        w = orch.advance(w.id)  # → KYC_FAILED
        assert w.state is WithdrawalState.KYC_FAILED
        assert w.error == "sanctions_hit"

    def test_bank_rejection_terminates_at_payout(self):
        kyc = KycServiceImpl(StubKycVendor())
        oracle = _FixedOracle(cents=200)
        swap = CoinbaseSwapService(StubCoinbaseClient(actual_slippage_bps=10))
        payout = StripePayoutService(StubStripeClient())
        orch = WithdrawalOrchestrator(
            kyc=kyc, oracle=oracle, swap=swap, payout=payout,
            store=InMemoryWithdrawalStore(),
        )

        w = orch.request("alice", 10**18, "bank_reject")
        # Advance through KYC + oracle + swap.
        for _ in range(3):
            w = orch.advance(w.id)
        assert w.state is WithdrawalState.SWAP_EXECUTED
        # Payout step flips to PAYOUT_FAILED.
        w = orch.advance(w.id)
        assert w.state is WithdrawalState.PAYOUT_FAILED
        assert "payout_error" in (w.error or "")

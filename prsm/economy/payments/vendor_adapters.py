"""Vendor-adapter scaffolds for Phase 5 Tasks 2-4.

Per docs/2026-04-22-phase5-fiat-onramp-design-plan.md §6 Tasks 2-4.

Covers the three external-vendor integration surfaces the Task 5
WithdrawalOrchestrator consumes:

  * KYC  (Task 2)  — Persona / Sumsub / Onfido (§8.1 pending decision).
  * Stripe (Task 3) — production-mode USD payouts.
  * Coinbase Exchange (Task 4) — FTNS↔USDC swap.

This module ships:

  * Vendor-client Protocols (KycVendorAdapter, StripeClient,
    CoinbaseExchangeClient) — the wire-level contract each real
    vendor implementation will satisfy.
  * In-process StubKycVendor / StubStripeClient / StubCoinbaseClient —
    deterministic test-mode implementations that let the whole
    withdrawal pipeline run end-to-end in unit tests without any
    live credentials.
  * Service impls (KycServiceImpl, StripePayoutService,
    CoinbaseSwapService) — adapt the vendor Protocols to the
    higher-level KycService / PayoutService / SwapService Protocols
    that WithdrawalOrchestrator already consumes.

Real production wiring follows the same Protocol contracts; swapping
`StubFooClient` for `LiveFooClient` is the only change once
credentials land. That's Task 2/3/4 production execution, gated on
vendor selection (§8.1) + credentials issuance.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Callable, Dict, Optional, Protocol

from prsm.economy.payments.withdrawal_orchestrator import (
    KycResult,
    PayoutResult,
    SwapResult,
)


logger = logging.getLogger(__name__)


__all__ = [
    # Errors
    "CoinbaseError",
    "CoinbaseOrderRejected",
    "CoinbaseSlippageExceeded",
    "KycProviderUnavailable",
    "KycSanctionsHit",
    "KycVerificationError",
    "StripeDestinationRejected",
    "StripeError",
    "StripeInsufficientFunds",
    # Records
    "CoinbaseOrder",
    "KycSession",
    "KycStatus",
    "StripePayout",
    "StripePayoutStatus",
    # Protocols
    "CoinbaseExchangeClient",
    "KycVendorAdapter",
    "StripeClient",
    # Stubs
    "StubCoinbaseClient",
    "StubKycVendor",
    "StubStripeClient",
    # Service impls
    "CoinbaseSwapService",
    "KycServiceImpl",
    "StripePayoutService",
]


# =============================================================================
# Errors
# =============================================================================


class KycVerificationError(Exception):
    """Base for KYC verification failures."""


class KycSanctionsHit(KycVerificationError):
    """OFAC / sanctions list match. Not retryable."""


class KycProviderUnavailable(KycVerificationError):
    """Vendor API returned 5xx / timeout. Retryable by the caller."""


class StripeError(Exception):
    """Base for Stripe integration failures."""


class StripeInsufficientFunds(StripeError):
    """Stripe connected account lacks balance. Surfacing requires
    operator intervention."""


class StripeDestinationRejected(StripeError):
    """Bank rejected the destination account (closed, frozen, etc.)."""


class CoinbaseError(Exception):
    """Base for Coinbase Exchange failures."""


class CoinbaseSlippageExceeded(CoinbaseError):
    """Fill price exceeded the configured slippage tolerance."""


class CoinbaseOrderRejected(CoinbaseError):
    """Order rejected by the exchange (bad market, halted symbol)."""


# =============================================================================
# KYC — Task 2
# =============================================================================


class KycStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class KycSession:
    session_id: str
    user_id: str
    status: KycStatus
    verification_level: str   # basic / enhanced / high
    created_at: int
    expires_at: int
    failure_reason: Optional[str] = None


class KycVendorAdapter(Protocol):
    """Vendor-specific KYC client. Real implementations wrap Persona /
    Sumsub / Onfido SDKs."""

    def create_session(self, user_id: str, level: str = "basic") -> KycSession: ...
    def check_status(self, session_id: str) -> KycSession: ...
    def is_sanctioned(self, user_id: str) -> bool: ...


@dataclass
class StubKycVendor:
    """Deterministic test-mode KYC vendor.

    Rules:
      * user_id starting with 'sanctioned-' → is_sanctioned returns True
        (KycSanctionsHit raised by the service layer).
      * user_id starting with 'unavailable-' → methods raise
        KycProviderUnavailable on demand (simulates vendor 5xx).
      * user_id starting with 'pending-' → sessions stay PENDING.
      * user_id starting with 'fail-' → sessions flip to FAILED with
        a stubbed reason.
      * All other user_ids → sessions advance PENDING → VERIFIED on the
        second check_status call (simulates async vendor review).
    """

    session_ttl_seconds: int = 15 * 60  # 15 minutes
    _now: Callable[[], float] = field(default_factory=lambda: time.time)
    _sessions: Dict[str, KycSession] = field(default_factory=dict)
    _status_check_counts: Dict[str, int] = field(default_factory=dict)
    _force_unavailable: bool = False

    def create_session(self, user_id: str, level: str = "basic") -> KycSession:
        if self._force_unavailable or user_id.startswith("unavailable-"):
            raise KycProviderUnavailable(f"stub: forced unavailable for {user_id}")
        now = int(self._now())
        session_id = f"kyc-{secrets.token_hex(8)}"
        session = KycSession(
            session_id=session_id,
            user_id=user_id,
            status=KycStatus.PENDING,
            verification_level=level,
            created_at=now,
            expires_at=now + self.session_ttl_seconds,
        )
        self._sessions[session_id] = session
        return session

    def check_status(self, session_id: str) -> KycSession:
        if self._force_unavailable:
            raise KycProviderUnavailable("stub: forced unavailable")
        session = self._sessions.get(session_id)
        if session is None:
            raise KycVerificationError(f"unknown session: {session_id}")

        # Expiry short-circuit.
        if int(self._now()) >= session.expires_at:
            expired = KycSession(
                session_id=session.session_id,
                user_id=session.user_id,
                status=KycStatus.EXPIRED,
                verification_level=session.verification_level,
                created_at=session.created_at,
                expires_at=session.expires_at,
                failure_reason="session_expired",
            )
            self._sessions[session_id] = expired
            return expired

        uid = session.user_id
        if uid.startswith("pending-"):
            return session  # stays PENDING forever
        if uid.startswith("fail-"):
            failed = KycSession(
                session_id=session.session_id,
                user_id=uid,
                status=KycStatus.FAILED,
                verification_level=session.verification_level,
                created_at=session.created_at,
                expires_at=session.expires_at,
                failure_reason="doc_rejected",
            )
            self._sessions[session_id] = failed
            return failed

        # Default happy-path: VERIFIED on second check (first check
        # returns PENDING; mimics async vendor review).
        count = self._status_check_counts.get(session_id, 0) + 1
        self._status_check_counts[session_id] = count
        if count < 2:
            return session
        verified = KycSession(
            session_id=session.session_id,
            user_id=uid,
            status=KycStatus.VERIFIED,
            verification_level=session.verification_level,
            created_at=session.created_at,
            expires_at=session.expires_at,
        )
        self._sessions[session_id] = verified
        return verified

    def is_sanctioned(self, user_id: str) -> bool:
        if self._force_unavailable:
            raise KycProviderUnavailable("stub: forced unavailable (sanctions)")
        return user_id.startswith("sanctioned-")

    def force_unavailable(self, v: bool = True) -> None:
        self._force_unavailable = v


class KycServiceImpl:
    """KycService Protocol impl that wraps a KycVendorAdapter.

    Flow:
      1. Sanctions screening (cheap, no session cost).
      2. Session create → check. Polls up to `max_poll_attempts` times
         with `poll_interval_seconds` between attempts.
      3. Returns KycResult(passed=True) on VERIFIED; (passed=False,
         reason=…) on FAILED / EXPIRED / sanctions hit; raises
         KycProviderUnavailable on persistent vendor failure.
    """

    def __init__(
        self,
        vendor: KycVendorAdapter,
        *,
        max_poll_attempts: int = 5,
    ) -> None:
        self._vendor = vendor
        self._max_polls = max_poll_attempts

    def check(self, user_id: str) -> KycResult:
        # Sanctions first — cheap + terminal.
        if self._vendor.is_sanctioned(user_id):
            return KycResult(passed=False, reason="sanctions_hit")

        session = self._vendor.create_session(user_id)
        for _ in range(self._max_polls):
            session = self._vendor.check_status(session.session_id)
            if session.status is KycStatus.VERIFIED:
                return KycResult(passed=True)
            if session.status is KycStatus.FAILED:
                return KycResult(
                    passed=False, reason=session.failure_reason or "kyc_failed"
                )
            if session.status is KycStatus.EXPIRED:
                return KycResult(passed=False, reason="session_expired")

        # Ran out of polls — treat as failure.
        return KycResult(passed=False, reason="polling_exhausted")


# =============================================================================
# Stripe — Task 3
# =============================================================================


class StripePayoutStatus(str, Enum):
    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"


@dataclass(frozen=True)
class StripePayout:
    payout_id: str
    amount_cents: int
    destination_token: str
    status: StripePayoutStatus
    created_at: int
    failure_reason: Optional[str] = None


class StripeClient(Protocol):
    def create_payout(
        self,
        amount_cents: int,
        destination_token: str,
        idempotency_key: str,
    ) -> StripePayout: ...

    def get_payout(self, payout_id: str) -> StripePayout: ...


@dataclass
class StubStripeClient:
    """Test-mode Stripe client. Respects the following conventions
    aligned with Stripe's test tokens:

      * destination_token 'bank_reject'   → StripeDestinationRejected
      * destination_token 'bank_no_funds' → StripeInsufficientFunds
      * destination_token 'bank_fail'     → payout succeeds, then
        transitions to FAILED on the next get_payout.
      * anything else                     → PAID immediately.

    Idempotency: a duplicate idempotency_key returns the SAME
    StripePayout as the first call — matches Stripe's real
    Idempotency-Key header behaviour.
    """

    _now: Callable[[], float] = field(default_factory=lambda: time.time)
    _payouts: Dict[str, StripePayout] = field(default_factory=dict)
    _idempotency_index: Dict[str, str] = field(default_factory=dict)

    def create_payout(
        self,
        amount_cents: int,
        destination_token: str,
        idempotency_key: str,
    ) -> StripePayout:
        if idempotency_key in self._idempotency_index:
            return self._payouts[self._idempotency_index[idempotency_key]]

        if destination_token == "bank_reject":
            raise StripeDestinationRejected(f"stub: destination rejected ({destination_token})")
        if destination_token == "bank_no_funds":
            raise StripeInsufficientFunds("stub: connected account has no balance")

        payout_id = f"po_{secrets.token_hex(10)}"
        payout = StripePayout(
            payout_id=payout_id,
            amount_cents=amount_cents,
            destination_token=destination_token,
            status=StripePayoutStatus.PAID,
            created_at=int(self._now()),
        )
        self._payouts[payout_id] = payout
        self._idempotency_index[idempotency_key] = payout_id
        return payout

    def get_payout(self, payout_id: str) -> StripePayout:
        payout = self._payouts.get(payout_id)
        if payout is None:
            raise StripeError(f"unknown payout: {payout_id}")
        if payout.destination_token == "bank_fail":
            failed = StripePayout(
                payout_id=payout.payout_id,
                amount_cents=payout.amount_cents,
                destination_token=payout.destination_token,
                status=StripePayoutStatus.FAILED,
                created_at=payout.created_at,
                failure_reason="async_bank_rejection",
            )
            self._payouts[payout_id] = failed
            return failed
        return payout


class StripePayoutService:
    """PayoutService Protocol impl wrapping a StripeClient.

    Generates an idempotency key from the (destination_token,
    amount_cents, second-bucket-timestamp) tuple. Duplicate calls
    within a 1-second window return the same payout — matches the
    orchestrator's idempotent-advance() semantics.
    """

    def __init__(self, client: StripeClient) -> None:
        self._client = client

    def payout_usd_to_bank(
        self,
        *,
        amount_cents: int,
        destination_bank_token: str,
    ) -> PayoutResult:
        # Idempotency key deterministic per (destination, amount, second)
        idempotency_key = (
            f"payout:{destination_bank_token}:{amount_cents}:{int(time.time())}"
        )
        try:
            payout = self._client.create_payout(
                amount_cents=amount_cents,
                destination_token=destination_bank_token,
                idempotency_key=idempotency_key,
            )
        except StripeError:
            raise

        if payout.status is StripePayoutStatus.FAILED:
            raise StripeError(
                f"payout failed: {payout.failure_reason or 'unknown'}"
            )
        return PayoutResult(payout_id=payout.payout_id)


# =============================================================================
# Coinbase — Task 4
# =============================================================================


@dataclass(frozen=True)
class CoinbaseOrder:
    order_id: str
    side: str                       # 'sell' (FTNS→USDC) or 'buy' (reverse)
    base_amount_wei: int            # FTNS wei sold
    quote_amount_micro: int         # USDC micro-dollars received (1 USDC = 1e6)
    status: str                     # 'filled' / 'rejected'
    fill_price_usd: Decimal         # USD per FTNS at fill time


class CoinbaseExchangeClient(Protocol):
    def place_market_sell(
        self,
        base_amount_wei: int,
        max_slippage_bps: int,
    ) -> CoinbaseOrder: ...

    def get_order(self, order_id: str) -> CoinbaseOrder: ...


@dataclass
class StubCoinbaseClient:
    """Deterministic Coinbase stub.

    `mid_price_usd` is the reference price. Stub fills at
    `mid_price_usd × (1 - actual_slippage_bps / 10000)`. Configure
    `actual_slippage_bps` to exercise the slippage-exceeded path.

    Setting `reject_next_order = True` makes the next
    place_market_sell raise CoinbaseOrderRejected — simulates a
    halted market or rate-limit rejection from the exchange.
    """

    mid_price_usd: Decimal = Decimal("2.00")
    actual_slippage_bps: int = 10  # 0.10% default
    reject_next_order: bool = False
    _orders: Dict[str, CoinbaseOrder] = field(default_factory=dict)

    def place_market_sell(
        self,
        base_amount_wei: int,
        max_slippage_bps: int,
    ) -> CoinbaseOrder:
        if self.reject_next_order:
            self.reject_next_order = False
            raise CoinbaseOrderRejected("stub: market halted")
        if self.actual_slippage_bps > max_slippage_bps:
            raise CoinbaseSlippageExceeded(
                f"actual {self.actual_slippage_bps} bps > max {max_slippage_bps} bps"
            )

        slippage_factor = Decimal(1) - Decimal(self.actual_slippage_bps) / Decimal(10000)
        fill_price = self.mid_price_usd * slippage_factor
        ftns_amount = Decimal(base_amount_wei) / Decimal(10**18)
        usdc_amount_usd = ftns_amount * fill_price
        quote_amount_micro = int(usdc_amount_usd * Decimal(10**6))

        order_id = f"cb_{secrets.token_hex(10)}"
        order = CoinbaseOrder(
            order_id=order_id,
            side="sell",
            base_amount_wei=base_amount_wei,
            quote_amount_micro=quote_amount_micro,
            status="filled",
            fill_price_usd=fill_price,
        )
        self._orders[order_id] = order
        return order

    def get_order(self, order_id: str) -> CoinbaseOrder:
        order = self._orders.get(order_id)
        if order is None:
            raise CoinbaseError(f"unknown order: {order_id}")
        return order


class CoinbaseSwapService:
    """SwapService Protocol impl wrapping a CoinbaseExchangeClient.

    Enforces slippage tolerance against the expected USD quote: if
    the fill's realised USD total differs from the expected quote by
    more than `max_expected_slippage_bps`, raises
    CoinbaseSlippageExceeded. Prevents the orchestrator from
    accepting a wildly-off-price fill.
    """

    def __init__(
        self,
        client: CoinbaseExchangeClient,
        *,
        max_exchange_slippage_bps: int = 200,         # 2% at the exchange level
        max_expected_quote_drift_bps: int = 500,      # 5% vs the oracle quote
    ) -> None:
        self._client = client
        self._max_exchange_slippage_bps = max_exchange_slippage_bps
        self._max_expected_drift_bps = max_expected_quote_drift_bps

    def swap_ftns_to_usdc(
        self,
        ftns_wei: int,
        *,
        expected_usd_cents: int,
    ) -> SwapResult:
        order = self._client.place_market_sell(
            base_amount_wei=ftns_wei,
            max_slippage_bps=self._max_exchange_slippage_bps,
        )
        if order.status != "filled":
            raise CoinbaseOrderRejected(
                f"order {order.order_id}: status={order.status}"
            )

        # Cross-check actual-vs-expected: exchange fill could have moved
        # since the oracle quote was taken.
        realised_cents = order.quote_amount_micro // 10_000  # micro → cents
        drift_bps = abs(realised_cents - expected_usd_cents) * 10_000 // max(expected_usd_cents, 1)
        if drift_bps > self._max_expected_drift_bps:
            raise CoinbaseSlippageExceeded(
                f"drift {drift_bps} bps > max {self._max_expected_drift_bps} bps "
                f"(expected {expected_usd_cents}c, got {realised_cents}c)"
            )

        return SwapResult(
            order_id=order.order_id,
            usdc_received=order.quote_amount_micro,
        )

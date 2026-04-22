"""FTNS → USD withdrawal orchestrator + state machine.

Per docs/2026-04-22-phase5-fiat-onramp-design-plan.md §3, §6 Task 5.

The orchestrator wires together KYC, oracle, FTNS→USDC swap, and Stripe
payout into a single state machine so failures at any stage surface
cleanly and retries are deterministic.

States:

    requested ─► kyc_check ─► oracle_quote ─► swap_executed ─► stripe_payout ─► completed
                    │               │                │                 │
                    ▼               ▼                ▼                 ▼
                kyc_failed     quote_failed    swap_failed      payout_failed

    Non-terminal states can cancel → canceled.
    Terminal failures require Foundation operator review (manual reset).

Design commitments:

  * The orchestrator is I/O-free. All external services (KYC, oracle,
    swap, payout) are injected as `Protocol` implementations; the real
    Stripe / Coinbase / Persona wiring is Tasks 2/3/4 in the Phase 5
    plan. Task 5 is the logic.
  * `advance()` is idempotent on terminal states (completed / *_failed /
    canceled) — repeatedly calling it is a safe retry mechanism the
    caller can drive from a background worker without risking state
    corruption.
  * The swap step is the "no-cancel boundary." Once FTNS has been
    converted to USDC, user-initiated cancel is rejected — funds have
    already moved and the only remaining path is forward through
    payout (or operator-initiated refund, which is out of scope here).
  * Every state transition updates `updated_at`. Persisted records are
    source-of-truth; the orchestrator is stateless and can be
    reconstructed from any SQLite snapshot.
"""

from __future__ import annotations

import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Protocol


logger = logging.getLogger(__name__)


__all__ = [
    "InMemoryWithdrawalStore",
    "KycResult",
    "KycService",
    "OracleService",
    "PayoutResult",
    "PayoutService",
    "SqliteWithdrawalStore",
    "SwapResult",
    "SwapService",
    "Withdrawal",
    "WithdrawalError",
    "WithdrawalNotFoundError",
    "WithdrawalOrchestrator",
    "WithdrawalState",
    "WithdrawalStateError",
    "WithdrawalStore",
]


# -----------------------------------------------------------------------------
# States
# -----------------------------------------------------------------------------


class WithdrawalState(str, Enum):
    REQUESTED = "requested"
    KYC_CHECK = "kyc_check"
    ORACLE_QUOTE = "oracle_quote"
    SWAP_EXECUTED = "swap_executed"
    STRIPE_PAYOUT = "stripe_payout"
    COMPLETED = "completed"

    # Terminal failures — require operator intervention.
    KYC_FAILED = "kyc_failed"
    QUOTE_FAILED = "quote_failed"
    SWAP_FAILED = "swap_failed"
    PAYOUT_FAILED = "payout_failed"

    # User-initiated.
    CANCELED = "canceled"


_TERMINAL_STATES = frozenset(
    {
        WithdrawalState.COMPLETED,
        WithdrawalState.KYC_FAILED,
        WithdrawalState.QUOTE_FAILED,
        WithdrawalState.SWAP_FAILED,
        WithdrawalState.PAYOUT_FAILED,
        WithdrawalState.CANCELED,
    }
)

# States from which user-initiated cancel is allowed. Once funds are
# in-flight (swap executed or later), cancel is rejected — the only
# recovery is operator-initiated refund outside this module.
_CANCELABLE_STATES = frozenset(
    {
        WithdrawalState.REQUESTED,
        WithdrawalState.KYC_CHECK,
        WithdrawalState.ORACLE_QUOTE,
    }
)


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class WithdrawalError(Exception):
    """Base class for orchestrator failures."""


class WithdrawalNotFoundError(WithdrawalError):
    """Withdrawal ID not found in the store."""


class WithdrawalStateError(WithdrawalError):
    """Operation not permitted from the current state."""


# -----------------------------------------------------------------------------
# External-service protocols + result types
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class KycResult:
    passed: bool
    reason: Optional[str] = None


class KycService(Protocol):
    def check(self, user_id: str) -> KycResult: ...


class OracleService(Protocol):
    """Returns USD cents for a given FTNS wei amount, or raises on
    oracle-unavailable / stale-price."""

    def quote_ftns_to_usd_cents(self, ftns_wei: int) -> int: ...


@dataclass(frozen=True)
class SwapResult:
    order_id: str
    usdc_received: int  # micro-USDC (1 USDC = 1,000,000)


class SwapService(Protocol):
    def swap_ftns_to_usdc(
        self, ftns_wei: int, *, expected_usd_cents: int
    ) -> SwapResult: ...


@dataclass(frozen=True)
class PayoutResult:
    payout_id: str


class PayoutService(Protocol):
    def payout_usd_to_bank(
        self, *, amount_cents: int, destination_bank_token: str
    ) -> PayoutResult: ...


# -----------------------------------------------------------------------------
# Withdrawal record + storage
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Withdrawal:
    id: str
    user_id: str
    ftns_wei: int
    destination_bank_token: str
    state: WithdrawalState
    created_at: int
    updated_at: int
    usd_quote_cents: int = 0
    coinbase_order_id: Optional[str] = None
    stripe_payout_id: Optional[str] = None
    error: Optional[str] = None


class WithdrawalStore(Protocol):
    def save(self, withdrawal: Withdrawal) -> None: ...
    def load(self, withdrawal_id: str) -> Optional[Withdrawal]: ...


class InMemoryWithdrawalStore:
    def __init__(self) -> None:
        self._rows: dict[str, Withdrawal] = {}

    def save(self, withdrawal: Withdrawal) -> None:
        self._rows[withdrawal.id] = withdrawal

    def load(self, withdrawal_id: str) -> Optional[Withdrawal]:
        return self._rows.get(withdrawal_id)


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS withdrawals (
    id                       TEXT PRIMARY KEY,
    user_id                  TEXT NOT NULL,
    ftns_wei                 INTEGER NOT NULL,
    destination_bank_token   TEXT NOT NULL,
    state                    TEXT NOT NULL,
    created_at               INTEGER NOT NULL,
    updated_at               INTEGER NOT NULL,
    usd_quote_cents          INTEGER NOT NULL DEFAULT 0,
    coinbase_order_id        TEXT,
    stripe_payout_id         TEXT,
    error                    TEXT
);
CREATE INDEX IF NOT EXISTS idx_withdrawals_user_id ON withdrawals(user_id);
CREATE INDEX IF NOT EXISTS idx_withdrawals_state ON withdrawals(state);
"""


class SqliteWithdrawalStore:
    """Postgres-portable schema; no SQLite-specific types."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SQLITE_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, withdrawal: Withdrawal) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO withdrawals "
                "(id, user_id, ftns_wei, destination_bank_token, state, "
                " created_at, updated_at, usd_quote_cents, "
                " coinbase_order_id, stripe_payout_id, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                " state = excluded.state, "
                " updated_at = excluded.updated_at, "
                " usd_quote_cents = excluded.usd_quote_cents, "
                " coinbase_order_id = excluded.coinbase_order_id, "
                " stripe_payout_id = excluded.stripe_payout_id, "
                " error = excluded.error",
                (
                    withdrawal.id,
                    withdrawal.user_id,
                    withdrawal.ftns_wei,
                    withdrawal.destination_bank_token,
                    withdrawal.state.value,
                    withdrawal.created_at,
                    withdrawal.updated_at,
                    withdrawal.usd_quote_cents,
                    withdrawal.coinbase_order_id,
                    withdrawal.stripe_payout_id,
                    withdrawal.error,
                ),
            )
            conn.commit()

    def load(self, withdrawal_id: str) -> Optional[Withdrawal]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM withdrawals WHERE id = ?",
                (withdrawal_id,),
            ).fetchone()
        if row is None:
            return None
        return Withdrawal(
            id=row["id"],
            user_id=row["user_id"],
            ftns_wei=row["ftns_wei"],
            destination_bank_token=row["destination_bank_token"],
            state=WithdrawalState(row["state"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            usd_quote_cents=row["usd_quote_cents"],
            coinbase_order_id=row["coinbase_order_id"],
            stripe_payout_id=row["stripe_payout_id"],
            error=row["error"],
        )


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------


class WithdrawalOrchestrator:
    def __init__(
        self,
        *,
        kyc: KycService,
        oracle: OracleService,
        swap: SwapService,
        payout: PayoutService,
        store: WithdrawalStore,
        clock: Callable[[], float] = time.time,
        id_generator: Callable[[], str] = lambda: secrets.token_hex(16),
    ) -> None:
        self._kyc = kyc
        self._oracle = oracle
        self._swap = swap
        self._payout = payout
        self._store = store
        self._clock = clock
        self._id_generator = id_generator

    # ---- public API ------------------------------------------------------

    def request(
        self,
        user_id: str,
        ftns_wei: int,
        destination_bank_token: str,
    ) -> Withdrawal:
        if ftns_wei <= 0:
            raise WithdrawalError("ftns_wei must be > 0")
        now = int(self._clock())
        withdrawal = Withdrawal(
            id=self._id_generator(),
            user_id=user_id,
            ftns_wei=ftns_wei,
            destination_bank_token=destination_bank_token,
            state=WithdrawalState.REQUESTED,
            created_at=now,
            updated_at=now,
        )
        self._store.save(withdrawal)
        return withdrawal

    def advance(self, withdrawal_id: str) -> Withdrawal:
        """Drive the state machine one step forward. Idempotent on
        terminal states — repeatedly calling on a completed / failed /
        canceled withdrawal returns it unchanged.
        """
        w = self._load_or_raise(withdrawal_id)
        if w.state in _TERMINAL_STATES:
            return w

        if w.state is WithdrawalState.REQUESTED:
            return self._step_to_kyc(w)
        if w.state is WithdrawalState.KYC_CHECK:
            return self._run_kyc(w)
        if w.state is WithdrawalState.ORACLE_QUOTE:
            return self._run_quote(w)
        if w.state is WithdrawalState.SWAP_EXECUTED:
            return self._run_payout(w)
        if w.state is WithdrawalState.STRIPE_PAYOUT:
            # The payout step is complete when advance re-enters from it —
            # treat as a polling pattern that marks completed once the
            # previous `_run_payout` has landed.
            return self._mark_completed(w)
        raise WithdrawalStateError(
            f"unhandled state in advance(): {w.state!r}"
        )

    def cancel(self, withdrawal_id: str) -> Withdrawal:
        w = self._load_or_raise(withdrawal_id)
        if w.state not in _CANCELABLE_STATES:
            raise WithdrawalStateError(
                f"cannot cancel from state {w.state.value!r}"
            )
        return self._transition(w, WithdrawalState.CANCELED)

    def get(self, withdrawal_id: str) -> Withdrawal:
        return self._load_or_raise(withdrawal_id)

    # ---- transition steps ------------------------------------------------

    def _step_to_kyc(self, w: Withdrawal) -> Withdrawal:
        return self._transition(w, WithdrawalState.KYC_CHECK)

    def _run_kyc(self, w: Withdrawal) -> Withdrawal:
        try:
            result = self._kyc.check(w.user_id)
        except Exception as exc:
            logger.exception("KYC service error for withdrawal %s", w.id)
            return self._transition(
                w, WithdrawalState.KYC_FAILED, error=f"kyc_error: {exc}"
            )
        if not result.passed:
            return self._transition(
                w,
                WithdrawalState.KYC_FAILED,
                error=result.reason or "kyc_rejected",
            )
        return self._transition(w, WithdrawalState.ORACLE_QUOTE)

    def _run_quote(self, w: Withdrawal) -> Withdrawal:
        try:
            cents = self._oracle.quote_ftns_to_usd_cents(w.ftns_wei)
        except Exception as exc:
            logger.exception("oracle error for withdrawal %s", w.id)
            return self._transition(
                w, WithdrawalState.QUOTE_FAILED, error=f"oracle_error: {exc}"
            )
        if cents <= 0:
            return self._transition(
                w, WithdrawalState.QUOTE_FAILED, error="nonpositive_quote"
            )

        w = replace(w, usd_quote_cents=cents, updated_at=int(self._clock()))
        # Execute the swap in the same advance() call so the state
        # transition and swap are coupled. If the swap fails, we mark
        # SWAP_FAILED; the quote is preserved in the record for audit.
        try:
            swap_result = self._swap.swap_ftns_to_usdc(
                w.ftns_wei, expected_usd_cents=cents
            )
        except Exception as exc:
            logger.exception("swap service error for withdrawal %s", w.id)
            return self._transition(
                w, WithdrawalState.SWAP_FAILED, error=f"swap_error: {exc}"
            )
        return self._transition(
            w,
            WithdrawalState.SWAP_EXECUTED,
            coinbase_order_id=swap_result.order_id,
        )

    def _run_payout(self, w: Withdrawal) -> Withdrawal:
        try:
            payout_result = self._payout.payout_usd_to_bank(
                amount_cents=w.usd_quote_cents,
                destination_bank_token=w.destination_bank_token,
            )
        except Exception as exc:
            logger.exception("payout service error for withdrawal %s", w.id)
            return self._transition(
                w, WithdrawalState.PAYOUT_FAILED, error=f"payout_error: {exc}"
            )
        return self._transition(
            w,
            WithdrawalState.STRIPE_PAYOUT,
            stripe_payout_id=payout_result.payout_id,
        )

    def _mark_completed(self, w: Withdrawal) -> Withdrawal:
        return self._transition(w, WithdrawalState.COMPLETED)

    # ---- internal helpers ------------------------------------------------

    def _transition(
        self,
        w: Withdrawal,
        new_state: WithdrawalState,
        *,
        coinbase_order_id: Optional[str] = None,
        stripe_payout_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Withdrawal:
        updated = replace(
            w,
            state=new_state,
            updated_at=int(self._clock()),
            coinbase_order_id=coinbase_order_id or w.coinbase_order_id,
            stripe_payout_id=stripe_payout_id or w.stripe_payout_id,
            error=error if error is not None else w.error,
        )
        self._store.save(updated)
        return updated

    def _load_or_raise(self, withdrawal_id: str) -> Withdrawal:
        w = self._store.load(withdrawal_id)
        if w is None:
            raise WithdrawalNotFoundError(withdrawal_id)
        return w

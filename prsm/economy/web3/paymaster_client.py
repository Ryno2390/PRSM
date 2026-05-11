"""Sprint 277 — Coinbase CDP paymaster adapter.

Per Vision §14 "Crypto-UX adoption barrier" mitigation: users
should never need to hold gas tokens or see "transaction fees."
This adapter sponsors gas via the Coinbase CDP paymaster
endpoint, deducting cost from an operator-funded sponsorship
budget (deposited paymaster reserve).

ERC-4337 conceptual model: client builds a UserOperation,
paymaster signs it (validating per-op limits), bundler submits
to the entry point. This v1 scaffold abstracts the
EntryPoint/Bundler plumbing behind a dependency-injected
backend — production backend wraps the real CDP endpoint;
tests use a fake.

PENDING_COMMISSION pattern (mirrors WaaS + offramp quote):
when paymaster env keys are absent, sponsor_user_op returns a
preview record without touching any external endpoint. Once
commissioned, behavior splits on dry_run:

  - dry_run=True  → call estimate_gas only, return ESTIMATED.
  - dry_run=False → call submit_sponsored, return SUBMITTED
                    on success or FAILED (with error string) if
                    the backend raises.

Per R-2026-05-08-1 (composer-only invariant): scoped to
coinbase_offramp_initiate. Gasless FTNS transfer is intra-
network user-authorized money movement — authorization is
implicit via the user's WaaS-managed key — same risk profile
as prsm_royalty_claim's execute path. The dry_run-by-default
default at the endpoint + MCP layer preserves the
"composer-first, execute-on-confirm" UX.

Operator env:
  - COINBASE_CDP_PAYMASTER_ENDPOINT   — paymaster RPC URL
  - COINBASE_CDP_PAYMASTER_API_KEY    — auth header for the URL
  - PRSM_PAYMASTER_POLICY_ID          — opt-in CDP policy id
                                        (per-op spend caps etc.)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class _PaymasterBackend(Protocol):
    """Dependency-injected backend. Production = CDP paymaster;
    tests = a fake."""

    def estimate_gas(
        self, user_op: Dict[str, Any],
    ) -> Dict[str, Any]: ...

    def submit_sponsored(
        self, user_op: Dict[str, Any],
    ) -> Dict[str, Any]: ...


@dataclass
class SponsorshipResult:
    status: str  # PENDING_COMMISSION | ESTIMATED | SUBMITTED | FAILED
    tx_hash: Optional[str] = None
    user_op_hash: Optional[str] = None
    gas_estimate_wei: Optional[int] = None
    sponsor_amount_wei: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PaymasterClient:
    """In-process paymaster adapter with cumulative spend telemetry."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        policy_id: Optional[str] = None,
        backend: Optional[_PaymasterBackend] = None,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._policy_id = policy_id
        self._backend = backend
        # Spend telemetry — non-persistent; resets per restart.
        # (Persistence is a follow-on if operators want lifetime
        # spend tracking across reboots.)
        self._total_sponsored_wei: int = 0
        self._sponsorship_count: int = 0

    @classmethod
    def from_env(
        cls, *, backend: Optional[_PaymasterBackend] = None,
    ) -> "PaymasterClient":
        endpoint = (
            os.environ.get("COINBASE_CDP_PAYMASTER_ENDPOINT") or None
        )
        api_key = (
            os.environ.get("COINBASE_CDP_PAYMASTER_API_KEY") or None
        )
        policy_id = (
            os.environ.get("PRSM_PAYMASTER_POLICY_ID") or None
        )
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            policy_id=policy_id,
            backend=backend,
        )

    def is_commissioned(self) -> bool:
        """True iff both paymaster env keys are present."""
        return bool(self._endpoint and self._api_key)

    def sponsor_user_op(
        self,
        user_op: Dict[str, Any],
        *,
        dry_run: bool = True,
    ) -> SponsorshipResult:
        """Sponsor a UserOperation.

        Pre-commission: returns PENDING_COMMISSION without backend
        call. Commissioned + dry_run: estimate-only. Commissioned +
        execute: full submit. Backend exceptions during execute
        produce FAILED records (fail-soft) and do not pollute the
        cumulative spend counter."""
        if not isinstance(user_op, dict):
            raise ValueError("user_op must be a dict")
        if not user_op.get("sender"):
            raise ValueError("user_op missing required 'sender'")
        if not user_op.get("to"):
            raise ValueError("user_op missing required 'to'")

        if not self.is_commissioned() or self._backend is None:
            return SponsorshipResult(status="PENDING_COMMISSION")

        try:
            estimate = self._backend.estimate_gas(user_op)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PaymasterClient: estimate_gas raised: %s", exc,
            )
            return SponsorshipResult(
                status="FAILED",
                error=f"estimate_gas: {exc}",
            )
        gas_estimate_wei = int(
            estimate.get("gas_estimate_wei", 0) or 0
        )

        if dry_run:
            return SponsorshipResult(
                status="ESTIMATED",
                gas_estimate_wei=gas_estimate_wei,
            )

        try:
            submit = self._backend.submit_sponsored(user_op)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PaymasterClient: submit_sponsored raised: %s", exc,
            )
            return SponsorshipResult(
                status="FAILED",
                gas_estimate_wei=gas_estimate_wei,
                error=f"submit_sponsored: {exc}",
            )

        sponsor_amount_wei = int(
            submit.get("sponsor_amount_wei", 0) or 0
        )
        self._total_sponsored_wei += sponsor_amount_wei
        self._sponsorship_count += 1

        return SponsorshipResult(
            status="SUBMITTED",
            tx_hash=submit.get("tx_hash"),
            user_op_hash=submit.get("user_op_hash"),
            gas_estimate_wei=gas_estimate_wei,
            sponsor_amount_wei=sponsor_amount_wei,
        )

    # ── Spend telemetry ──────────────────────────────────

    def total_sponsored_wei(self) -> int:
        return self._total_sponsored_wei

    def total_sponsorships(self) -> int:
        return self._sponsorship_count

    def spend_summary(self) -> Dict[str, Any]:
        return {
            "commissioned": self.is_commissioned(),
            "sponsorships": self._sponsorship_count,
            "total_sponsored_wei": self._total_sponsored_wei,
            "endpoint": self._endpoint,
            "policy_id": self._policy_id,
        }
